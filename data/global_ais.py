"""Zoomed-out global ship layer.

The layer contains real AISStream contacts only. If no API key is configured,
or the upstream connection has not produced a position report, the API returns
an empty contact list and an explicit status instead of synthetic substitutes.

This module never blocks the rest of the server on the real feed: run() is a
background task, snapshot() always returns immediately from whatever state
currently exists.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import aiohttp

from data.maritime_context import maritime_context

AISSTREAM_WS_URL = "wss://stream.aisstream.io/v0/stream"
CONNECT_TIMEOUT_S = 8.0
MAX_TRACKED = int(os.getenv("AEGIS_MAX_ACTIVE_VESSELS", "100000"))
DARK_AFTER_S = 45.0  # no fresh position report: render as a coasting contact
MAX_HISTORY = 20
MAX_TOMBSTONES = 50000
STATE_RETENTION_S = 24 * 60 * 60

# Busy maritime regions across every inhabited continent. A single world box
# can deliver thousands of messages per second and starve the API server; these
# boxes retain global operational coverage while keeping one process responsive.
REGIONAL_BOXES = [
    [[35, -10], [65, 30]],       # North Sea / Western Europe
    [[20, 100], [45, 145]],      # East Asia / Sea of Japan
    [[-40, 110], [-10, 155]],    # Australia
    [[25, -100], [50, -60]],     # US East Coast / Gulf / Great Lakes
    [[30, -130], [55, -115]],    # US and Canada West Coast
    [[-10, -55], [15, -30]],     # Brazil and tropical Atlantic
    [[25, -10], [45, 40]],       # Mediterranean
    [[5, 40], [30, 80]],         # Red Sea / Persian Gulf / India
    [[-10, 90], [20, 125]],      # Malacca / Indonesia
    [[-40, 10], [-20, 45]],      # Southern Africa
]

# ------------------------------------------------------------------- real feed

class GlobalAisFeed:
    """Background aisstream.io consumer. `run()` is started once as an
    asyncio task at server startup and never returns except on cancellation;
    reconnects with backoff on any failure. `live` only ever flips True after
    a real position report has actually been parsed -- never on connection
    open alone, so a server that connects but gets no traffic still reports
    itself honestly as not-yet-live."""

    def __init__(self, api_key: str, state_path: str | Path | None = None):
        self.api_key = api_key
        self.state_path = Path(state_path) if state_path else None
        self.vessels: dict[int, dict] = {}
        self.static_data: dict[int, dict] = {}
        self.pinned_mmsi: int | None = None
        self.live = False
        self._active_order: OrderedDict[int, None] = OrderedDict()
        self._dark_mmsi: set[int] = set()
        self.revision = 0
        self._removed: list[tuple[int, int]] = []
        self._delta_floor = 0
        self.connected = False
        self.messages_received = 0
        self.position_reports = 0
        self.static_reports = 0
        self.identity_switches = 0
        self.reconnects = 0
        self.last_message_at = 0.0
        self.last_error = ""
        self._load_state()

    def _load_state(self) -> None:
        """Restore real last-report timestamps saved during a prior run."""
        if self.state_path is None or not self.state_path.is_file():
            return
        now = time.time()
        try:
            with gzip.open(self.state_path, "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
            for saved in payload.get("vessels", ()):
                mmsi = int(saved["mmsi"])
                last_seen = float(saved["last_seen"])
                age_s = max(0.0, now - last_seen)
                if age_s > STATE_RETENTION_S:
                    continue
                row = {
                    key: value
                    for key, value in saved.items()
                    if not key.startswith("_")
                }
                self.revision += 1
                row["_revision"] = self.revision
                self.vessels[mmsi] = row
                if age_s >= DARK_AFTER_S:
                    self._dark_mmsi.add(mmsi)
                else:
                    self._active_order[mmsi] = None
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            # A bad cache must never prevent the real AIS connection starting.
            self.vessels.clear()
            self._active_order.clear()
            self._dark_mmsi.clear()
            self.revision = 0

    def save_state(self) -> None:
        """Atomically retain recent real contacts so silence age survives restarts."""
        if self.state_path is None:
            return
        cutoff = time.time() - STATE_RETENTION_S
        rows = [
            {
                key: value
                for key, value in vessel.items()
                if not key.startswith("_")
            }
            for vessel in self.vessels.values()
            if float(vessel.get("last_seen", 0.0)) >= cutoff
        ]
        rows.sort(key=lambda row: float(row.get("last_seen", 0.0)), reverse=True)
        rows = rows[:MAX_TRACKED]
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            json.dump({"saved_at": time.time(), "vessels": rows}, stream)
        temporary.replace(self.state_path)

    @staticmethod
    def _identity_changed(mmsi: int, previous: dict, values: dict) -> bool:
        return any(
            previous.get(field) not in (None, "", 0, f"MMSI {mmsi}")
            and values.get(field) not in (None, "", 0)
            and str(previous[field]).strip().upper()
            != str(values[field]).strip().upper()
            for field in ("name", "imo", "call_sign")
        )

    def _record(self, mmsi: int, fix: dict) -> None:
        received_at = time.time()
        becoming_active = mmsi not in self._active_order
        if becoming_active and len(self._active_order) >= MAX_TRACKED:
            oldest = next(
                (candidate for candidate in self._active_order
                 if candidate != self.pinned_mmsi),
                None,
            )
            if oldest is not None:
                self._active_order.pop(oldest, None)
                self.vessels.pop(oldest, None)
                self._record_removal(oldest)
        self._dark_mmsi.discard(mmsi)
        self._active_order.pop(mmsi, None)
        self._active_order[mmsi] = None
        was_positioned = mmsi in self.vessels
        previous = self.vessels.get(mmsi)
        if previous is None:
            previous = self.static_data.pop(mmsi, {})
        if was_positioned and self._identity_changed(mmsi, previous, fix):
            self.identity_switches += 1
        history = list(previous.get("history", []))
        history_samples = list(previous.get("history_samples", []))
        if "lat" in fix and "lon" in fix:
            point = [float(fix["lat"]), float(fix["lon"])]
            if not history or history[-1] != point:
                history.append(point)
                history = history[-MAX_HISTORY:]
                history_samples.append({
                    "lat": point[0],
                    "lon": point[1],
                    "time": received_at,
                    "course": fix.get("course"),
                    "speed_kn": fix.get("speed_kn"),
                })
                history_samples = history_samples[-MAX_HISTORY:]
        row = {
            **previous,
            **fix,
            "mmsi": mmsi,
            "history": history,
            "history_samples": history_samples,
            "last_seen": received_at,
        }
        row.pop("_annotation_key", None)
        row.pop("_annotation", None)
        self.vessels[mmsi] = row
        self._mark_changed(mmsi)
        self.live = True

    def _mark_changed(self, mmsi: int) -> None:
        self.revision += 1
        if mmsi in self.vessels:
            self.vessels[mmsi]["_revision"] = self.revision

    def _record_removal(self, mmsi: int) -> None:
        self.revision += 1
        self._removed.append((self.revision, mmsi))
        if len(self._removed) > MAX_TOMBSTONES:
            dropped_revision, _ = self._removed.pop(0)
            self._delta_floor = dropped_revision

    def pin(self, mmsi: int | None) -> None:
        """Protect one operator-selected contact from cache eviction."""
        self.pinned_mmsi = mmsi

    def _record_static(self, mmsi: int, values: dict) -> None:
        if mmsi in self.vessels:
            previous = self.vessels[mmsi]
            if self._identity_changed(mmsi, previous, values):
                self.identity_switches += 1
            row = {**self.vessels[mmsi], **values}
            row.pop("_annotation_key", None)
            row.pop("_annotation", None)
            self.vessels[mmsi] = row
            self._mark_changed(mmsi)
            return
        if len(self.static_data) >= MAX_TRACKED * 2:
            self.static_data.pop(next(iter(self.static_data)))
        self.static_data[mmsi] = {**self.static_data.get(mmsi, {}), **values}

    @staticmethod
    def _first(mapping: dict, *names: str, default=None):
        for name in names:
            value = mapping.get(name)
            if value not in (None, ""):
                return value
        return default

    def _handle_message(self, raw) -> None:
        # aisstream.io sends its JSON payloads as BINARY websocket frames
        # (confirmed empirically -- msg.type is WSMsgType.BINARY, not TEXT,
        # even though the payload itself is plain JSON text), so `raw` may
        # be str or bytes depending on which frame type carried it.
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        data = json.loads(raw)
        self.messages_received += 1
        self.last_message_at = time.time()
        meta = data.get("MetaData") or {}
        message_type = data.get("MessageType", "")
        message = data.get("Message") or {}
        mmsi = self._first(meta, "MMSI", "Mmsi")

        if message_type in ("ShipStaticData", "StaticDataReport"):
            static = message.get(message_type) or {}
            mmsi = self._first(
                meta,
                "MMSI",
                "Mmsi",
                default=self._first(static, "UserID", "UserId"),
            )
            if mmsi is None:
                return
            self._record_static(int(mmsi), {
                "name": str(
                    self._first(
                        static,
                        "Name",
                        "ShipName",
                        default=meta.get("ShipName", ""),
                    )
                ).strip(),
                "imo": self._first(static, "ImoNumber", "IMONumber", "Imo"),
                "call_sign": str(
                    self._first(static, "CallSign", "Callsign", default="")
                ).strip(),
                "ship_type": self._first(static, "Type", "ShipType", default=""),
                "destination": str(static.get("Destination", "")).strip(),
                "draught_m": self._first(
                    static, "MaximumStaticDraught", "Draught", default=0.0
                ),
            })
            self.static_reports += 1
            return

        lat = meta.get("latitude")
        lon = meta.get("longitude")
        if mmsi is None or lat is None or lon is None:
            report = (
                message.get("PositionReport")
                or message.get("StandardClassBPositionReport")
                or {}
            )
            lat = lat if lat is not None else report.get("Latitude")
            lon = lon if lon is not None else report.get("Longitude")
        if mmsi is None or lat is None or lon is None:
            return
        report = (
            message.get("PositionReport")
            or message.get("StandardClassBPositionReport")
            or {}
        )
        self._record(int(mmsi), {
            "mmsi": int(mmsi),
            "name": (
                meta.get("ShipName", "").strip()
                or self.vessels.get(int(mmsi), {}).get("name")
                or f"MMSI {mmsi}"
            ),
            "lat": round(float(lat), 4),
            "lon": round(float(lon), 4),
            "course": report.get("Cog"),
            "speed_kn": report.get("Sog"),
            "heading": report.get("TrueHeading"),
            "navigation_status": report.get("NavigationalStatus"),
            "rate_of_turn": report.get("RateOfTurn"),
        })
        self.position_reports += 1

    async def run(self) -> None:
        backoff = 2.0
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(AISSTREAM_WS_URL) as ws:
                        self.connected = True
                        self.last_error = ""
                        await ws.send_str(json.dumps({
                            "APIKey": self.api_key,
                            "BoundingBoxes": REGIONAL_BOXES,
                            "FilterMessageTypes": [
                                "PositionReport",
                                "ShipStaticData",
                            ],
                        }))
                        backoff = 2.0
                        since_yield = 0
                        async for msg in ws:
                            if msg.type in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                                try:
                                    self._handle_message(msg.data)
                                except (ValueError, TypeError, KeyError, UnicodeDecodeError):
                                    continue
                                since_yield += 1
                                if since_yield >= 100:
                                    since_yield = 0
                                    await asyncio.sleep(0)
                            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = type(exc).__name__
            finally:
                self.connected = False
                self.reconnects += 1
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

    def _snapshot_rows(self, since: int = 0, only_mmsi: int | None = None) -> list[dict]:
        now = time.time()
        vessels = []
        items = (
            [(only_mmsi, self.vessels[only_mmsi])]
            if only_mmsi in self.vessels else []
        ) if only_mmsi is not None else list(self.vessels.items())
        for mmsi, fix in items:
            if "lat" not in fix or "lon" not in fix:
                continue
            age_s = max(0.0, now - float(fix.get("last_seen", now)))
            dark = age_s >= DARK_AFTER_S
            if dark and mmsi not in self._dark_mmsi:
                self._dark_mmsi.add(mmsi)
                self._active_order.pop(mmsi, None)
                self._mark_changed(mmsi)
            if since and int(fix.get("_revision", 0)) <= since:
                continue
            row = {
                key: value
                for key, value in fix.items()
                if not key.startswith("_")
            }
            row["age_s"] = round(age_s, 1)
            row["dark"] = dark
            annotation_key = (row["lat"], row["lon"], dark)
            if fix.get("_annotation_key") != annotation_key:
                fix["_annotation"] = maritime_context().annotate(row)
                fix["_annotation_key"] = annotation_key
            annotated = dict(fix["_annotation"])
            annotated["age_s"] = row["age_s"]
            annotated["dark"] = dark
            vessels.append(annotated)
        return vessels

    def snapshot(self) -> list[dict]:
        return self._snapshot_rows()

    def vessel_snapshot(self, mmsi: int) -> dict | None:
        rows = self._snapshot_rows(only_mmsi=mmsi)
        return rows[0] if rows else None

    def snapshot_since(self, since: int = 0) -> dict:
        full = since <= 0 or since < self._delta_floor or since > self.revision
        effective_since = 0 if full else since
        vessels = self._snapshot_rows(effective_since)
        removed = [] if full else [
            mmsi for revision, mmsi in self._removed if revision > since
        ]
        return {
            "vessels": vessels,
            "removed": removed,
            "revision": self.revision,
            "full": full,
        }

    def status(self) -> dict:
        return {
            "configured": True,
            "connected": self.connected,
            "messages_received": self.messages_received,
            "position_reports": self.position_reports,
            "static_reports": self.static_reports,
            "identity_switches": self.identity_switches,
            "pinned_mmsi": self.pinned_mmsi,
            "reconnects": self.reconnects,
            "last_message_at": self.last_message_at,
            "last_error": self.last_error,
            "regions": len(REGIONAL_BOXES),
            "max_tracked": MAX_TRACKED,
            "dark_after_s": DARK_AFTER_S,
            "active_contacts": len(self._active_order),
            "dark_contacts": len(self._dark_mmsi),
            "total_contacts": len(self.vessels),
        }


def global_snapshot(feed: "GlobalAisFeed | None", since: int = 0) -> dict:
    """Return real AIS contacts, or an explicit empty/offline state."""
    if feed is not None and feed.live and feed.vessels:
        delta = feed.snapshot_since(since)
        return {"live": True, **delta, "status": feed.status()}
    return {
        "live": False,
        "vessels": [],
        "removed": [],
        "revision": feed.revision if feed is not None else 0,
        "full": since <= 0,
        "status": {**feed.status(), "configured": True} if feed is not None else {
            "configured": False,
            "connected": False,
            "messages_received": 0,
            "position_reports": 0,
            "static_reports": 0,
            "reconnects": 0,
            "last_message_at": 0.0,
            "last_error": "",
            "regions": len(REGIONAL_BOXES),
            "max_tracked": MAX_TRACKED,
            "dark_after_s": DARK_AFTER_S,
        },
    }

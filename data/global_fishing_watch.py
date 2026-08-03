"""On-demand Global Fishing Watch identity, events, and map layers."""

from __future__ import annotations

import asyncio
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
from urllib.parse import urlparse, urlunparse
from typing import Any

import aiohttp

GFW_VESSEL_SEARCH_URL = (
    "https://gateway.api.globalfishingwatch.org/v3/vessels/search"
)
GFW_DATASET = "public-global-vessel-identity:latest"
GFW_EVENTS_URL = "https://gateway.api.globalfishingwatch.org/v3/events"
GFW_GENERATE_PNG_URL = (
    "https://gateway.api.globalfishingwatch.org/v3/4wings/generate-png"
)
GFW_ALLOWED_TILE_HOSTS = {
    "gateway.api.globalfishingwatch.org",
    "gateway.api.prod.globalfishingwatch.org",
}
EVENT_DATASETS = (
    "public-global-fishing-events:latest",
    "public-global-encounters-events:latest",
    "public-global-loitering-events:latest",
    "public-global-port-visits-events:latest",
    "public-global-gaps-events:latest",
)
MAP_LAYERS = {
    "fishing": {
        "dataset": "public-global-fishing-effort:latest",
        "color": "#ffb454",
        "lag_days": 4,
        "days": 30,
        "label": "Apparent fishing effort",
        "unit": "fishing hours",
    },
    "sar": {
        "dataset": "public-global-sar-presence:latest",
        "color": "#db7cff",
        "lag_days": 5,
        "days": 30,
        "label": "SAR vessel detections",
        "unit": "satellite detections",
    },
}
CACHE_TTL_S = 3600.0
EVENT_CACHE_TTL_S = 900.0
STYLE_CACHE_TTL_S = 21600.0


def _latest(records: list[dict[str, Any]], mmsi: str) -> dict[str, Any]:
    matching = [row for row in records if str(row.get("ssvid", "")) == mmsi]
    candidates = matching or records
    return max(
        candidates,
        key=lambda row: str(row.get("transmissionDateTo", "")),
        default={},
    )


def parse_vessel_search(payload: dict[str, Any], mmsi: int) -> dict[str, Any]:
    """Reduce the large GFW response to fields used by the dashboard."""
    entries = payload.get("entries") or []
    if not entries:
        return {"matched": False, "mmsi": mmsi, "source": "Global Fishing Watch"}

    entry = entries[0]
    mmsi_text = str(mmsi)
    self_reported = _latest(entry.get("selfReportedInfo") or [], mmsi_text)
    registry = _latest(entry.get("registryInfo") or [], mmsi_text)
    combined = entry.get("combinedSourcesInfo") or []
    vessel_id = self_reported.get("id") or (
        combined[0].get("vesselId") if combined else None
    )
    shiptypes = sorted({
        item.get("name", "")
        for row in combined
        for item in row.get("shiptypes", [])
        if item.get("name")
    })
    geartypes = sorted({
        item.get("name", "")
        for row in combined
        for item in row.get("geartypes", [])
        if item.get("name")
    })
    return {
        "matched": True,
        "source": "Global Fishing Watch",
        "dataset": entry.get("dataset"),
        "vessel_id": vessel_id,
        "mmsi": mmsi,
        "name": self_reported.get("shipname") or registry.get("shipname"),
        "imo": self_reported.get("imo") or registry.get("imo"),
        "call_sign": self_reported.get("callsign") or registry.get("callsign"),
        "flag": self_reported.get("flag") or registry.get("flag"),
        "positions_count": self_reported.get("positionsCounter"),
        "transmission_from": self_reported.get("transmissionDateFrom"),
        "transmission_to": self_reported.get("transmissionDateTo"),
        "ship_types": shiptypes,
        "gear_types": geartypes,
        "registry_sources": registry.get("sourceCode") or [],
    }


def parse_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize heterogeneous GFW event records for the dashboard."""
    normalized = []
    for event in payload.get("entries") or []:
        position = event.get("position") or {}
        lat = position.get("lat")
        lon = position.get("lon")
        if lat is None or lon is None:
            continue
        event_type = str(event.get("type") or "").upper()
        start = event.get("start")
        end = event.get("end")
        detail = event.get(event_type.lower()) or {}
        duration_hours = detail.get("averageDurationHours")
        if duration_hours is None and start and end:
            try:
                start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
                duration_hours = (end_dt - start_dt).total_seconds() / 3600
            except ValueError:
                duration_hours = None
        normalized.append({
            "id": event.get("id"),
            "type": event_type,
            "start": start,
            "end": end,
            "lat": float(lat),
            "lon": float(lon),
            "duration_hours": (
                round(float(duration_hours), 2)
                if duration_hours is not None else None
            ),
            "regions": event.get("regions") or {},
            "distances": event.get("distances") or {},
            "port": event.get("port") or detail.get("port"),
            "encounter": event.get("encounter"),
            "fishing": event.get("fishing"),
            "loitering": event.get("loitering"),
            "gap": event.get("gap"),
            "source": "Global Fishing Watch",
            "classification": "modelled" if event_type == "FISHING" else "observed",
        })
    normalized.sort(key=lambda row: str(row.get("start") or ""), reverse=True)
    return normalized


def load_protected_area_index(path: Any) -> dict[str, dict[str, Any]]:
    """Load the supplied WDPA/WDOECM attributes without inventing geometry."""
    if not path:
        return {}
    source = Path(path).expanduser()
    if not source.is_file():
        return {}
    index = {}
    try:
        with source.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                site_id = str(row.get("SITE_ID") or "").strip()
                if not site_id:
                    continue
                index[site_id] = {
                    "id": site_id,
                    "name": row.get("NAME_ENG") or row.get("NAME"),
                    "designation": row.get("DESIG_ENG") or row.get("DESIG"),
                    "iucn_category": row.get("IUCN_CAT"),
                    "no_take": str(row.get("NO_TAKE") or "").strip(),
                    "status": row.get("STATUS"),
                    "status_year": row.get("STATUS_YR"),
                    "manager": row.get("MANG_AUTH"),
                    "country": row.get("ISO3") or row.get("PRNT_ISO3"),
                    "source": "Protected Planet WDPA/WDOECM August 2026",
                }
    except (OSError, csv.Error, UnicodeError):
        return {}
    return index


class GlobalFishingWatchClient:
    def __init__(self, token: str, protected_areas_path: Any = None):
        self.token = token
        self.protected_areas = load_protected_area_index(protected_areas_path)
        self._cache: dict[int, tuple[float, dict[str, Any]]] = {}
        self._event_cache: dict[int, tuple[float, dict[str, Any]]] = {}
        self._style_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._style_locks = {kind: asyncio.Lock() for kind in MAP_LAYERS}

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    async def vessel_identity(self, mmsi: int) -> dict[str, Any]:
        cached = self._cache.get(mmsi)
        now = time.time()
        if cached and now - cached[0] < CACHE_TTL_S:
            return cached[1]

        timeout = aiohttp.ClientTimeout(total=10)
        params = [
            ("query", str(mmsi)),
            ("datasets[0]", GFW_DATASET),
            ("includes[0]", "MATCH_CRITERIA"),
            ("includes[1]", "AUTHORIZATIONS"),
        ]
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    GFW_VESSEL_SEARCH_URL,
                    params=params,
                    headers=self.headers,
                ) as response:
                    if response.status != 200:
                        return {
                            "matched": False,
                            "mmsi": mmsi,
                            "source": "Global Fishing Watch",
                            "error": f"upstream_status_{response.status}",
                        }
                    result = parse_vessel_search(await response.json(), mmsi)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            return {
                "matched": False,
                "mmsi": mmsi,
                "source": "Global Fishing Watch",
                "error": "upstream_unavailable",
            }

        self._cache[mmsi] = (now, result)
        return result

    async def vessel_activity(self, mmsi: int) -> dict[str, Any]:
        cached = self._event_cache.get(mmsi)
        now = time.time()
        if cached and now - cached[0] < EVENT_CACHE_TTL_S:
            return cached[1]

        identity = await self.vessel_identity(mmsi)
        vessel_id = identity.get("vessel_id")
        if not vessel_id:
            return {
                "matched": False,
                "mmsi": mmsi,
                "events": [],
                "source": "Global Fishing Watch",
            }

        end = datetime.now(timezone.utc).date() + timedelta(days=1)
        start = end - timedelta(days=366)
        params: list[tuple[str, str]] = [
            ("vessels[0]", str(vessel_id)),
            ("start-date", start.isoformat()),
            ("end-date", end.isoformat()),
            ("limit", "100"),
            ("offset", "0"),
            ("sort", "-start"),
            ("include-regions", "true"),
        ]
        params.extend(
            (f"datasets[{index}]", dataset)
            for index, dataset in enumerate(EVENT_DATASETS)
        )
        timeout = aiohttp.ClientTimeout(total=18)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    GFW_EVENTS_URL,
                    params=params,
                    headers=self.headers,
                ) as response:
                    if response.status != 200:
                        return {
                            "matched": True,
                            "mmsi": mmsi,
                            "events": [],
                            "source": "Global Fishing Watch",
                            "error": f"upstream_status_{response.status}",
                        }
                    payload = await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            return {
                "matched": True,
                "mmsi": mmsi,
                "events": [],
                "source": "Global Fishing Watch",
                "error": "upstream_unavailable",
            }

        events = parse_events(payload)
        for event in events:
            mpa_ids = {
                str(item)
                for key in ("mpa", "mpaNoTake", "mpaNoTakePartial")
                for item in (event.get("regions", {}).get(key) or [])
            }
            event["protected_areas"] = [
                self.protected_areas[site_id]
                for site_id in sorted(mpa_ids)
                if site_id in self.protected_areas
            ]
        result = {
            "matched": True,
            "mmsi": mmsi,
            "vessel_id": vessel_id,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "events": events,
            "total": int(payload.get("total") or 0),
            "source": "Global Fishing Watch",
            "caveat": (
                "Activity is derived from AIS and satellite analysis. "
                "An event inside a managed area is not by itself evidence of illegality."
            ),
        }
        self._event_cache[mmsi] = (now, result)
        return result

    async def map_layers(self) -> dict[str, Any]:
        layers = []
        for key, config in MAP_LAYERS.items():
            style = await self._map_style(key)
            layers.append({
                "id": key,
                "label": config["label"],
                "unit": config["unit"],
                "from": style.get("from"),
                "to": style.get("to"),
                "available": bool(style.get("url")),
                "source": "Global Fishing Watch",
                "error": style.get("error"),
            })
        return {
            "configured": True,
            "layers": layers,
            "protected_areas": {
                "records": len(self.protected_areas),
                "source": "Protected Planet WDPA/WDOECM August 2026",
                "geometry_available": False,
            },
            "note": (
                f"Fishing effort is modelled from AIS. SAR shows satellite "
                f"detections. {len(self.protected_areas):,} protected-area "
                "records provide event context; activity alone does not "
                "establish illegality."
            ),
        }

    async def _map_style(self, kind: str) -> dict[str, Any]:
        if kind not in MAP_LAYERS:
            return {"error": "unknown_layer"}
        cached = self._style_cache.get(kind)
        now = time.time()
        if cached and now - cached[0] < STYLE_CACHE_TTL_S:
            return cached[1]

        async with self._style_locks[kind]:
            cached = self._style_cache.get(kind)
            now = time.time()
            if cached and now - cached[0] < STYLE_CACHE_TTL_S:
                return cached[1]

            config = MAP_LAYERS[kind]
            end = datetime.now(timezone.utc).date() - timedelta(days=config["lag_days"])
            start = end - timedelta(days=config["days"])
            params = [
                ("interval", "DAY"),
                ("datasets[0]", str(config["dataset"])),
                ("color", str(config["color"])),
                ("date-range", f"{start.isoformat()},{end.isoformat()}"),
            ]
            timeout = aiohttp.ClientTimeout(total=20)
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        GFW_GENERATE_PNG_URL,
                        params=params,
                        headers=self.headers,
                    ) as response:
                        if response.status != 200:
                            result = {
                                "error": f"upstream_status_{response.status}",
                                "from": start.isoformat(),
                                "to": end.isoformat(),
                            }
                        else:
                            payload = await response.json()
                            result = {
                                "url": payload.get("url"),
                                "from": start.isoformat(),
                                "to": end.isoformat(),
                            }
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
                result = {
                    "error": "upstream_unavailable",
                    "from": start.isoformat(),
                    "to": end.isoformat(),
                }
            self._style_cache[kind] = (now, result)
            return result

    async def map_tile(
        self,
        kind: str,
        z: int,
        x: int,
        y: int,
    ) -> tuple[int, bytes, str]:
        style = await self._map_style(kind)
        template = style.get("url")
        if not template:
            return 503, b"", "image/png"
        url = (
            str(template)
            .replace("{z}", str(z))
            .replace("{x}", str(x))
            .replace("{y}", str(y))
        )
        parsed = urlparse(url)
        if parsed.hostname not in GFW_ALLOWED_TILE_HOSTS:
            return 502, b"", "image/png"
        if parsed.hostname == "gateway.api.prod.globalfishingwatch.org":
            parsed = parsed._replace(netloc="gateway.api.globalfishingwatch.org")
            url = urlunparse(parsed)
        timeout = aiohttp.ClientTimeout(total=15)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=self.headers) as response:
                    return (
                        response.status,
                        await response.read(),
                        response.headers.get("Content-Type", "image/png"),
                    )
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return 503, b"", "image/png"

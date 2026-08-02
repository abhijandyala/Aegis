"""Scenario pack loader: AIS ingest, geofences, synthetic actors. Data prep only.

A scenario pack is a directory ``scenarios/<pack_id>/`` containing ``pack.json``
plus its GeoJSON layers and a cached AIS window CSV. ``load_scenario(pack_id)``
turns one into an in-memory :class:`Scenario`: fixed-interval frames of
unlabelled :class:`~data.contracts.Measurement` objects plus ENU geofences.

Boundary: this module does numerics and parsing only -- resampling,
projection, noise, ground-truth bookkeeping. Scripted event *application*
(``ais_off`` suppression, ``radar_contact`` injection, ``identity_change``)
is state mutation handled by ``aegis.driver.ScenarioDriver``. Events are parsed
here and attached to their frame, but
every frame carries the *full* unsuppressed AIS picture; the driver decides
what the tracker actually sees.

Identity handling: MMSI is stripped at ingest into ``Scenario.ground_truth``
keyed by ``meas_id``. Measurements are structurally incapable of carrying
identity, and :func:`_assert_no_identity_leak` verifies that at load time
rather than assuming it.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from data.contracts import (
    AIS_SIGMA_M,
    Frame,
    Geofence,
    Measurement,
    ScenarioEvent,
)

__all__ = [
    "Scenario",
    "load_scenario",
    "find_crossings",
    "true_position_at",
    "lla_to_enu",
    "enu_to_lla",
    "SCENARIOS_DIR",
]

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENARIOS_DIR = os.path.join(_REPO_ROOT, "scenarios")

_EARTH_R = 6_371_000.0  # metres
# Interpolating across a longer AIS gap than this invents positions; drop instead.
_MAX_INTERP_GAP_S = 600.0


# --------------------------------------------------------------------------- geo

def lla_to_enu(lat: float, lon: float, origin_lat: float, origin_lon: float) -> Tuple[float, float]:
    """Equirectangular lat/lon -> local ENU metres. Sub-metre error at bay scale."""
    x = math.radians(lon - origin_lon) * _EARTH_R * math.cos(math.radians(origin_lat))
    y = math.radians(lat - origin_lat) * _EARTH_R
    return x, y


def enu_to_lla(x: float, y: float, origin_lat: float, origin_lon: float) -> Tuple[float, float]:
    lat = origin_lat + math.degrees(y / _EARTH_R)
    lon = origin_lon + math.degrees(x / (_EARTH_R * math.cos(math.radians(origin_lat))))
    return lat, lon


def _parse_utc(s: str) -> float:
    """ISO timestamp -> POSIX seconds. Naive timestamps are treated as UTC
    (MarineCadastre BaseDateTime has no timezone suffix)."""
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


# ---------------------------------------------------------------------- scenario

@dataclass
class Scenario:
    """One fully materialised scenario: everything the replay engine needs."""

    pack_id: str
    name: str
    origin: Tuple[float, float]              # (lat, lon)
    frame_interval_s: float
    frames: List[Frame]
    geofences: List[Geofence]
    # meas_id -> true actor key ("mmsi:367..." or synthetic id). Scoring only.
    ground_truth: Dict[str, str]
    # actor key -> [(t, x, y)] true resampled positions (includes dark segments).
    true_tracks: Dict[str, List[Tuple[float, float, float]]]
    events: List[ScenarioEvent]
    expected: Dict[str, Any] = field(default_factory=dict)
    # meas_id counter continues where load stopped, so radar contacts injected
    # by the ScenarioDriver walker get collision-free ids.
    next_meas_seq: int = 0

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def duration_s(self) -> float:
        return self.n_frames * self.frame_interval_s

    @property
    def n_vessels(self) -> int:
        return len(self.true_tracks)

    def display_id(self, actor: str, t: float) -> str:
        """The identity ``actor`` broadcasts at time ``t``.

        Purely a function of ``identity_change`` events (``params.new_display_id``),
        never of measurements -- radar-source Measurements are structurally
        incapable of carrying identity, so this is the ONLY place a display
        identity can come from. An event at exactly ``t`` has already applied
        (inclusive ``<=``): s02_mmsi_spoof's two spoofing events both fire at
        ``t=0.0``, and ``display_id(actor, 0.0)`` must already reflect them, not
        wait for the next tick. With no identity_change event at or before
        ``t``, the actor key itself is the display identity (a synthetic
        ``gen:`` actor's own id, or a real vessel's own MMSI-as-key) -- most
        scenarios (s01) never touch identity at all, and this makes that the
        default rather than a special case.
        """
        latest_t = None
        latest_id = actor
        for ev in self.events:
            if ev.kind == "identity_change" and ev.actor == actor and ev.t <= t:
                if latest_t is None or ev.t >= latest_t:
                    latest_t = ev.t
                    latest_id = ev.params["new_display_id"]
        return latest_id

    def mint_meas_id(self, actor: str) -> str:
        """Reserve a fresh measurement id and bind its ground truth.

        Called by the ScenarioDriver walker when it injects a radar contact;
        keeping the counter and the ground-truth write here means the side
        table stays the single place identity ever lives.
        """
        mid = f"m{self.next_meas_seq:06d}"
        self.next_meas_seq += 1
        self.ground_truth[mid] = actor
        return mid


def _resolve_pack(name_or_path: str) -> Tuple[Dict[str, Any], str]:
    """Accept a pack id (dir under scenarios/), a pack dir, or a pack.json path."""
    candidates = [
        name_or_path,
        os.path.join(name_or_path, "pack.json"),
        os.path.join(SCENARIOS_DIR, name_or_path, "pack.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f), os.path.dirname(os.path.abspath(path))
    raise FileNotFoundError(
        f"scenario pack {name_or_path!r} not found (looked in {candidates})"
    )


def _load_geofences(pack: Dict[str, Any], pack_dir: str,
                    origin: Tuple[float, float]) -> List[Geofence]:
    fences: List[Geofence] = []
    for layer in pack.get("geofence_layers", []):
        path = os.path.join(pack_dir, layer["file"])
        with open(path, "r", encoding="utf-8") as f:
            gj = json.load(f)
        features = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
        for i, feat in enumerate(features):
            geom = feat["geometry"]
            if geom["type"] != "Polygon":
                raise ValueError(f"{path}: only Polygon geofences supported, got {geom['type']}")
            exterior = geom["coordinates"][0]  # GeoJSON is [lon, lat]
            ring = tuple(
                lla_to_enu(lat, lon, origin[0], origin[1]) for lon, lat in exterior
            )
            props = feat.get("properties", {})
            fences.append(Geofence(
                fence_id=props.get("id", f"{layer.get('kind', 'zone')}_{i}"),
                name=props.get("name", os.path.basename(path)),
                kind=layer.get("kind", props.get("kind", "zone")),
                ring=ring,
                geojson=feat,
            ))
    return fences


# -------------------------------------------------------------------- AIS ingest

def _ingest_ais(csv_path: str, pack: Dict[str, Any], t0: float, t1: float,
                interval: float, origin: Tuple[float, float]
                ) -> Dict[str, List[Tuple[float, float, float]]]:
    """Filter the cached AIS window to bbox+time, resample each vessel to the
    fixed frame grid, convert to ENU. Returns actor -> [(t_rel, x, y)]."""
    bbox = pack["bbox"]
    lon_min, lon_max = bbox["lon_min"], bbox["lon_max"]
    lat_min, lat_max = bbox["lat_min"], bbox["lat_max"]

    fixes: Dict[str, List[Tuple[float, float, float]]] = {}  # mmsi -> [(t_abs, lat, lon)]
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row["LAT"])
                lon = float(row["LON"])
            except (ValueError, KeyError):
                continue
            if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                continue
            t = _parse_utc(row["BaseDateTime"])
            if not (t0 <= t <= t1):
                continue
            fixes.setdefault(row["MMSI"], []).append((t, lat, lon))

    n_frames = int((t1 - t0) / interval) + 1
    frame_times = [t0 + k * interval for k in range(n_frames)]

    tracks: Dict[str, List[Tuple[float, float, float]]] = {}
    for mmsi, pts in fixes.items():
        pts.sort(key=lambda p: p[0])
        resampled: List[Tuple[float, float, float]] = []
        j = 0
        for ft in frame_times:
            while j + 1 < len(pts) and pts[j + 1][0] <= ft:
                j += 1
            if pts[j][0] > ft or j + 1 >= len(pts):
                # No bracketing pair; accept a fix within half a frame of the tick.
                if abs(pts[j][0] - ft) <= interval / 2:
                    _, lat, lon = pts[j]
                else:
                    continue
            else:
                ta, la, lo = pts[j]
                tb, lb, lo2 = pts[j + 1]
                if tb - ta > _MAX_INTERP_GAP_S:
                    continue
                w = 0.0 if tb == ta else (ft - ta) / (tb - ta)
                lat = la + w * (lb - la)
                lon = lo + w * (lo2 - lo)
            x, y = lla_to_enu(lat, lon, origin[0], origin[1])
            resampled.append((ft - t0, x, y))
        min_fixes = int(pack.get("min_fixes_per_vessel", 4))
        if len(resampled) >= min_fixes:
            tracks[f"mmsi:{mmsi}"] = resampled
    return tracks


# --------------------------------------------------------------- synthetic actors

def _gen_transit_then_loiter(params: Dict[str, Any], frame_times: Sequence[float],
                             origin: Tuple[float, float]
                             ) -> List[Tuple[float, float, float]]:
    """Straight transit at constant speed/heading, then a slow loiter circle
    around the point reached. Nav heading convention: 0 deg = north, clockwise."""
    x0, y0 = _actor_start(params, origin)
    speed = float(params["speed_mps"])
    hdg = math.radians(float(params["heading_deg"]))
    dx, dy = math.sin(hdg), math.cos(hdg)
    transit_s = float(params["transit_s"])
    r = float(params.get("loiter_radius_m", 150.0))
    # Loiter angular rate keeps tangential speed gentle (~2 m/s default).
    omega = float(params.get("loiter_speed_mps", 2.0)) / max(r, 1.0)
    t_start = float(params.get("t_start", 0.0))
    t_end = float(params.get("t_end", frame_times[-1] if frame_times else 0.0))

    cx = x0 + dx * speed * transit_s
    cy = y0 + dy * speed * transit_s

    out: List[Tuple[float, float, float]] = []
    for t in frame_times:
        if t < t_start or t > t_end:
            continue
        te = t - t_start
        if te <= transit_s:
            out.append((t, x0 + dx * speed * te, y0 + dy * speed * te))
        else:
            a = omega * (te - transit_s)
            # Enter the circle tangentially-ish: start at angle pointing back along track.
            out.append((t, cx + r * math.sin(a), cy - r * (1.0 - math.cos(a))))
    return out


def _gen_transit(params: Dict[str, Any], frame_times: Sequence[float],
                 origin: Tuple[float, float]) -> List[Tuple[float, float, float]]:
    """Constant-velocity straight line."""
    x0, y0 = _actor_start(params, origin)
    speed = float(params["speed_mps"])
    hdg = math.radians(float(params["heading_deg"]))
    dx, dy = math.sin(hdg), math.cos(hdg)
    t_start = float(params.get("t_start", 0.0))
    t_end = float(params.get("t_end", frame_times[-1] if frame_times else 0.0))
    return [
        (t, x0 + dx * speed * (t - t_start), y0 + dy * speed * (t - t_start))
        for t in frame_times
        if t_start <= t <= t_end
    ]


def _actor_start(params: Dict[str, Any], origin: Tuple[float, float]) -> Tuple[float, float]:
    if "start_latlon" in params:
        lat, lon = params["start_latlon"]
        return lla_to_enu(lat, lon, origin[0], origin[1])
    x, y = params["start"]
    return float(x), float(y)


GENERATORS: Dict[str, Callable[..., List[Tuple[float, float, float]]]] = {
    "gen:transit_then_loiter": _gen_transit_then_loiter,
    "gen:transit": _gen_transit,
}


# ------------------------------------------------------------------ truth query

def true_position_at(track: Sequence[Tuple[float, float, float]], t: float
                     ) -> Optional[Tuple[float, float]]:
    """Linear interpolation along a resampled true track.

    Used by the ScenarioDriver walker to place injected radar contacts on the
    actor's actual (possibly dark) position.
    """
    if not track or t < track[0][0] or t > track[-1][0]:
        return None
    for i in range(len(track) - 1):
        ta, xa, ya = track[i]
        tb, xb, yb = track[i + 1]
        if ta <= t <= tb:
            w = 0.0 if tb == ta else (t - ta) / (tb - ta)
            return xa + w * (xb - xa), ya + w * (yb - ya)
    return track[-1][1], track[-1][2]


# ----------------------------------------------------------------------- loading

def load_scenario(name_or_path: str) -> Scenario:
    """Parse a pack, ingest AIS, generate synthetic actors, apply scripted
    events, and return fully materialised frames plus geofences."""
    pack, pack_dir = _resolve_pack(name_or_path)

    origin = (float(pack["origin"]["lat"]), float(pack["origin"]["lon"]))
    interval = float(pack.get("frame_interval_s", 30.0))
    t0 = _parse_utc(pack["time_window"]["start"])
    t1 = _parse_utc(pack["time_window"]["end"])
    if t1 <= t0:
        raise ValueError(f"pack {pack['id']}: time_window end must be after start")

    n_frames = int((t1 - t0) / interval) + 1
    frame_times = [k * interval for k in range(n_frames)]

    geofences = _load_geofences(pack, pack_dir, origin)

    # --- true tracks: real AIS + synthetic actors, one namespace -------------
    true_tracks: Dict[str, List[Tuple[float, float, float]]] = {}
    ais_csv = pack.get("ais_csv")
    if ais_csv:
        csv_path = os.path.join(pack_dir, ais_csv)
        if os.path.isfile(csv_path):
            true_tracks.update(_ingest_ais(csv_path, pack, t0, t1, interval, origin))
        elif pack.get("ais_required", True):
            raise FileNotFoundError(
                f"pack {pack['id']}: AIS window {csv_path} missing; run "
                f"scripts/extract_window.py or set ais_required=false"
            )

    for actor in pack.get("synthetic_actors", []):
        gen = GENERATORS.get(actor["track"])
        if gen is None:
            raise ValueError(f"unknown track generator {actor['track']!r} "
                             f"(have {sorted(GENERATORS)})")
        track = gen(actor.get("params", {}), frame_times, origin)
        if actor["actor_id"] in true_tracks:
            raise ValueError(f"duplicate actor id {actor['actor_id']!r}")
        true_tracks[actor["actor_id"]] = track

    events = [
        ScenarioEvent(
            event_id=ev.get("id", f"ev{i:03d}"),
            t=float(ev["t"]),
            kind=ev["kind"],
            actor=ev["actor"],
            params=ev.get("params", {}),
        )
        for i, ev in enumerate(pack.get("events", []))
    ]
    # --- assemble frames ------------------------------------------------------
    # Every frame carries the FULL unsuppressed AIS picture. Applying events
    # (suppression, radar injection, identity display swaps) belongs to
    # aegis.driver.ScenarioDriver.
    rng = np.random.default_rng(int(pack.get("seed", 20260726)))
    ais_sigma = float(pack.get("ais_noise_sigma_m", AIS_SIGMA_M))
    ground_truth: Dict[str, str] = {}
    events_by_frame: Dict[int, List[ScenarioEvent]] = {}
    for ev in events:
        events_by_frame.setdefault(int(round(ev.t / interval)), []).append(ev)

    # Deterministic measurement order (sorted actors) so meas ids are stable
    # across loads regardless of dict insertion order in the CSV.
    actor_order = sorted(true_tracks)
    track_pos: Dict[str, int] = {a: 0 for a in actor_order}

    frames: List[Frame] = []
    next_meas = 0
    for k, ft in enumerate(frame_times):
        meas: List[Measurement] = []
        for actor in actor_order:
            track = true_tracks[actor]
            i = track_pos[actor]
            while i < len(track) and track[i][0] < ft - 1e-9:
                i += 1
            track_pos[actor] = i
            if i >= len(track) or abs(track[i][0] - ft) > 1e-6:
                continue
            _, tx, ty = track[i]
            mid = f"m{next_meas:06d}"
            next_meas += 1
            nx, ny = rng.normal(0.0, ais_sigma, 2)
            meas.append(Measurement(
                meas_id=mid, t=ft, x=tx + nx, y=ty + ny,
                source="ais", sigma=ais_sigma,
            ))
            ground_truth[mid] = actor

        frames.append(Frame(
            idx=k, t=ft,
            measurements=tuple(meas),
            events=tuple(events_by_frame.get(k, ())),
        ))

    scenario = Scenario(
        pack_id=pack["id"],
        name=pack.get("name", pack["id"]),
        origin=origin,
        frame_interval_s=interval,
        frames=frames,
        geofences=geofences,
        ground_truth=ground_truth,
        true_tracks=true_tracks,
        events=events,
        expected=pack.get("expected", {}),
        next_meas_seq=next_meas,
    )
    _assert_no_identity_leak(scenario)
    return scenario


_MEASUREMENT_FIELDS = frozenset({"meas_id", "t", "x", "y", "source", "sigma"})


def _assert_no_identity_leak(scenario: Scenario) -> None:
    """No measurement carries identity -- asserted, not assumed.

    Two layers: the Measurement shape must be exactly the frozen contract (so a
    future 'mmsi' field fails loudly here), and no field value may contain an
    actor key from ground truth.
    """
    actors = set(scenario.true_tracks)
    for frame in scenario.frames:
        for m in frame.measurements:
            fields = set(m.__dataclass_fields__)  # type: ignore[attr-defined]
            assert fields == _MEASUREMENT_FIELDS, (
                f"Measurement contract violated: unexpected fields "
                f"{fields ^ _MEASUREMENT_FIELDS}"
            )
            assert m.source in ("ais", "radar"), f"bad source {m.source!r}"
            for v in (m.meas_id, m.source):
                assert v not in actors and not str(v).startswith("mmsi:"), (
                    f"identity leaked into measurement {m.meas_id}: {v!r}"
                )


# -------------------------------------------------------------------- crossings

def find_crossings(true_tracks: Dict[str, List[Tuple[float, float, float]]],
                   interval: float,
                   max_dist_m: float = 500.0) -> List[Dict[str, Any]]:
    """Locate pairs of vessels within ``max_dist_m`` with converging headings.

    Converging means the pair's range is decreasing at the flagged frame
    (closing speed > 0.5 m/s) -- both moving matters less than actually
    closing. Consecutive flagged frames for a pair merge into one crossing.
    Returns [{pair, t_start, t_end, min_dist_m, closing_mps}] sorted by t.
    """
    # actor -> {t: (x, y)} for O(1) frame lookup.
    by_time = {
        a: {round(t, 3): (x, y) for t, x, y in trk}
        for a, trk in true_tracks.items()
    }
    actors = sorted(by_time)
    hits: Dict[Tuple[str, str], List[Tuple[float, float, float]]] = {}

    all_times = sorted({t for trk in by_time.values() for t in trk})
    for t in all_times:
        present = [(a, by_time[a][t]) for a in actors if t in by_time[a]]
        for i in range(len(present)):
            ai, (xi, yi) = present[i]
            for j in range(i + 1, len(present)):
                aj, (xj, yj) = present[j]
                d = math.hypot(xi - xj, yi - yj)
                if d > max_dist_m:
                    continue
                tp = round(t + interval, 3)
                if tp in by_time[ai] and tp in by_time[aj]:
                    xi2, yi2 = by_time[ai][tp]
                    xj2, yj2 = by_time[aj][tp]
                    d2 = math.hypot(xi2 - xj2, yi2 - yj2)
                    closing = (d - d2) / interval
                else:
                    closing = 0.0
                if closing > 0.5:
                    hits.setdefault((ai, aj), []).append((t, d, closing))

    crossings: List[Dict[str, Any]] = []
    for pair, pts in hits.items():
        pts.sort()
        start = prev = pts[0][0]
        seg = [pts[0]]
        for p in pts[1:] + [(float("inf"), 0.0, 0.0)]:
            if p[0] - prev > interval * 1.5:
                crossings.append({
                    "pair": pair,
                    "t_start": start,
                    "t_end": prev,
                    "min_dist_m": min(s[1] for s in seg),
                    "closing_mps": max(s[2] for s in seg),
                })
                start = p[0]
                seg = []
            seg.append(p)
            prev = p[0]
    crossings.sort(key=lambda c: c["t_start"])
    return crossings

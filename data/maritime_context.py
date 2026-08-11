"""Runtime context from every bundled maritime reference dataset."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from shapely.geometry import Point, shape
from shapely.ops import unary_union
from shapely.prepared import prep

from data.navigable_water import NavigableWaterMask
from financial import response_plan

ROOT = Path(__file__).resolve().parent.parent
GEO_DIR = ROOT / "geo"

_LAYER_SPECS = {
    "coastline": {
        "file": "ca_coastline_10m.geojson",
        "name": "California coastline",
        "kind": "land",
        "style": {"color": "#64748b", "weight": 1, "fillOpacity": 0.08},
    },
    "cables": {
        "file": "ca_submarine_cables.geojson",
        "name": "Submarine cables",
        "kind": "cable",
        "style": {"color": "#a78bfa", "weight": 2, "dashArray": "5 5"},
    },
    "sanctuary": {
        "file": "monterey_bay_nms_boundary.geojson",
        "name": "Monterey Bay National Marine Sanctuary",
        "kind": "sanctuary",
        "style": {"color": "#34d399", "weight": 2, "fillOpacity": 0.08},
    },
    "port": {
        "file": "port_of_sf_geofence.geojson",
        "name": "Port of San Francisco · NGA World Port Index",
        "kind": "port",
        "style": {"color": "#38bdf8", "weight": 2, "fillOpacity": 0.08},
    },
}

_OFAC_PATHS = (
    GEO_DIR / "ofac_sdn_vessels_subset.csv",
    ROOT / "scenarios" / "s03_ghost_fleet" / "ofac_sdn_vessels_subset.csv",
)
_MMSI_RE = re.compile(r"\bMMSI\s+(\d{9})\b", re.IGNORECASE)
_IMO_RE = re.compile(r"\bIMO\s+(\d{7})\b", re.IGNORECASE)


def _normalise(value: object) -> str:
    return "".join(ch for ch in str(value or "").upper() if ch.isalnum())


class MaritimeContext:
    """Load static layers once and annotate live AIS fixes cheaply."""

    def __init__(self) -> None:
        self._water_mask = NavigableWaterMask(
            ROOT / ".aegis" / "cache" / "navigable_water"
        )
        self._layers: dict[str, dict[str, Any]] = {}
        self._geometries: dict[str, Any] = {}
        self._prepared: dict[str, Any] = {}
        self._cables: list[tuple[str, Any]] = []
        for layer_id, spec in _LAYER_SPECS.items():
            path = GEO_DIR / spec["file"]
            raw = json.loads(path.read_text(encoding="utf-8"))
            geometries = [
                shape(feature["geometry"])
                for feature in raw.get("features", [])
                if feature.get("geometry")
            ]
            geometry = unary_union(geometries)
            self._layers[layer_id] = {**spec, "geojson": raw}
            self._geometries[layer_id] = geometry
            if layer_id != "cables":
                self._prepared[layer_id] = prep(geometry)
            else:
                self._cables = [
                    (
                        feature.get("properties", {}).get("name", "Cable"),
                        shape(feature["geometry"]),
                    )
                    for feature in raw.get("features", [])
                    if feature.get("geometry")
                ]

        self._ofac_by_mmsi: dict[str, dict[str, str]] = {}
        self._ofac_by_imo: dict[str, dict[str, str]] = {}
        self._ofac_by_call_sign: dict[str, dict[str, str]] = {}
        self._ofac_by_name: dict[str, dict[str, str]] = {}
        self._load_ofac()

    def _load_ofac(self) -> None:
        seen: set[str] = set()
        for path in _OFAC_PATHS:
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    record_id = str(row.get("ent_num", ""))
                    if not record_id or record_id in seen:
                        continue
                    seen.add(record_id)
                    remarks = row.get("Remarks", "")
                    record = {
                        "record_id": record_id,
                        "name": row.get("SDN_Name", ""),
                        "program": row.get("Program", ""),
                        "vessel_type": row.get("Vess_type", ""),
                        "flag": row.get("Vess_flag", ""),
                    }
                    for mmsi in _MMSI_RE.findall(remarks):
                        self._ofac_by_mmsi[mmsi] = record
                    for imo in _IMO_RE.findall(remarks):
                        self._ofac_by_imo[imo] = record
                    call_sign = _normalise(row.get("Call_Sign"))
                    if call_sign and call_sign != "0":
                        self._ofac_by_call_sign[call_sign] = record
                    name = _normalise(row.get("SDN_Name"))
                    if name:
                        self._ofac_by_name[name] = record

    def layer_payloads(self) -> list[dict[str, Any]]:
        return [
            {
                "layer_id": layer_id,
                "name": layer["name"],
                "kind": layer["kind"],
                "style": layer["style"],
                "geojson": layer["geojson"],
            }
            for layer_id, layer in self._layers.items()
        ]

    def _ofac_match(self, vessel: dict[str, Any]) -> dict[str, str] | None:
        candidates = (
            ("mmsi", self._ofac_by_mmsi, str(vessel.get("mmsi", ""))),
            ("imo", self._ofac_by_imo, str(vessel.get("imo", ""))),
            (
                "call_sign",
                self._ofac_by_call_sign,
                _normalise(vessel.get("call_sign")),
            ),
            ("name", self._ofac_by_name, _normalise(vessel.get("name"))),
        )
        for basis, index, value in candidates:
            if value and value in index:
                return {**index[value], "match_basis": basis}
        return None

    def terrain_status(self, lat: float, lon: float) -> dict[str, bool]:
        """Report land intersection from cached global water tiles or local data."""
        global_status = self._water_mask.status(lat, lon)
        if global_status["available"]:
            return global_status
        available = -126.0 <= lon <= -120.5 and 34.5 <= lat <= 38.5
        return {
            "available": available,
            "on_land": (
                bool(self._prepared["coastline"].covers(Point(lon, lat)))
                if available else False
            ),
        }

    def prime_navigable_water(
        self,
        lat: float,
        lon: float,
        radius_km: float,
    ) -> dict[str, Any]:
        return self._water_mask.prime_region(lat, lon, radius_km)

    def segment_crosses_land(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> bool:
        if self._water_mask.has_tiles:
            return self._water_mask.segment_crosses_land(start, end)
        # Preserve the bundled California fallback if global tiles are unavailable.
        mean_lat = math.radians((start[0] + end[0]) / 2)
        north_m = (end[0] - start[0]) * 111_320
        east_m = (
            (end[1] - start[1])
            * 111_320
            * max(0.1, math.cos(mean_lat))
        )
        checks = max(1, math.ceil(math.hypot(east_m, north_m) / 1500))
        for index in range(1, checks + 1):
            fraction = index / checks
            point = (
                start[0] + (end[0] - start[0]) * fraction,
                start[1] + (end[1] - start[1]) * fraction,
            )
            status = self.terrain_status(*point)
            if status["available"] and status["on_land"]:
                return True
        return False

    def clip_water_polygon(
        self,
        polygon: list[list[float]],
    ) -> list[dict[str, Any]] | None:
        return self._water_mask.clip_polygon(polygon)

    def annotate(self, vessel: dict[str, Any]) -> dict[str, Any]:
        row = dict(vessel)
        lat = float(row.get("lat", 0.0))
        lon = float(row.get("lon", 0.0))
        point = Point(lon, lat)

        # The bundled geometry covers California. Bounds checks in GEOS make
        # these predicates cheap for the rest of the global stream.
        in_california_extent = -126.0 <= lon <= -120.5 and 34.5 <= lat <= 38.5
        on_land = (
            self._prepared["coastline"].covers(point)
            if in_california_extent
            else False
        )
        in_sanctuary = (
            self._prepared["sanctuary"].covers(point)
            if in_california_extent
            else False
        )
        port_distance_km = (
            self._geometries["port"].distance(point) * 111.32
            if in_california_extent
            else float("inf")
        )
        in_port = port_distance_km <= 12.0
        nearest_cables: list[dict[str, Any]] = []
        if in_california_extent:
            for name, geometry in self._cables:
                distance_km = geometry.distance(point) * 111.32
                if distance_km <= 25.0:
                    nearest_cables.append({
                        "name": name,
                        "distance_km": round(distance_km, 1),
                    })
        nearest_cables.sort(key=lambda cable: cable["distance_km"])

        ofac = self._ofac_match(row)
        context = {
            "on_land": bool(on_land),
            "in_sanctuary": bool(in_sanctuary),
            "in_port": bool(in_port),
            "port": {
                "name": "SAN FRANCISCO",
                "distance_km": round(port_distance_km, 1),
                "source": "NGA World Port Index",
            } if in_port else None,
            "near_cables": nearest_cables[:3],
            "ofac": ofac,
        }
        row["context"] = context
        row["risk"] = self._risk(row, context)
        return row

    @staticmethod
    def _risk(vessel: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        signals: list[str] = []
        severity = "warning"
        if context["ofac"]:
            signals.append("Sanctions-list identity match")
            severity = "critical"
        if vessel.get("dark") and context["in_sanctuary"]:
            signals.append("AIS-silent contact in protected water")
            severity = "critical"
        elif vessel.get("dark"):
            signals.append("AIS-silent contact")
        if context["in_port"]:
            signals.append("Contact near a monitored port")
        if context["near_cables"]:
            signals.append("Contact near submarine infrastructure")
        if context["on_land"]:
            signals.append("Reported position plots on land")

        include_on_water = bool(vessel.get("dark") or context["on_land"])
        include_air = bool(
            vessel.get("dark")
            and context["in_sanctuary"]
        )
        return response_plan(
            signals,
            severity=severity,
            track_id=str(vessel.get("mmsi", "")),
            include_on_water=include_on_water,
            include_air=include_air,
        )


_CONTEXT: MaritimeContext | None = None


def maritime_context() -> MaritimeContext:
    global _CONTEXT
    if _CONTEXT is None:
        _CONTEXT = MaritimeContext()
    return _CONTEXT

"""Viewport and proximity queries against the NGA World Port Index."""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any

import aiohttp

WPI_URL = (
    "https://services9.arcgis.com/j1CY4yzWfwptbTWN/arcgis/rest/services/"
    "WorldPortIndex_WFL1/FeatureServer/0/query"
)
CACHE_TTL_S = 300.0
OUT_FIELDS = ",".join((
    "INDEX_NO",
    "PORT_NAME",
    "COUNTRY",
    "HARBORSIZE",
    "HARBORTYPE",
    "PORTOFENTR",
    "PILOT_REQD",
    "TUG_ASSIST",
    "MAX_VESSEL",
))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    value = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return radius_km * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def parse_ports(
    payload: dict[str, Any],
    center: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    ports = []
    for feature in payload.get("features") or []:
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []
        if len(coordinates) < 2:
            continue
        lon, lat = float(coordinates[0]), float(coordinates[1])
        properties = feature.get("properties") or {}
        port = {
            "id": properties.get("INDEX_NO"),
            "name": properties.get("PORT_NAME") or "Unnamed port",
            "country": properties.get("COUNTRY"),
            "lat": lat,
            "lon": lon,
            "harbor_size": properties.get("HARBORSIZE"),
            "harbor_type": properties.get("HARBORTYPE"),
            "port_of_entry": properties.get("PORTOFENTR") == "Y",
            "pilot_required": properties.get("PILOT_REQD") == "Y",
            "tug_assist": properties.get("TUG_ASSIST") == "Y",
            "max_vessel": properties.get("MAX_VESSEL"),
            "source": "NGA World Port Index",
        }
        if center is not None:
            port["distance_km"] = round(
                haversine_km(center[0], center[1], lat, lon),
                1,
            )
        ports.append(port)
    ports.sort(key=lambda row: (
        row.get("distance_km", float("inf")),
        str(row["name"]),
    ))
    return ports


class WorldPortIndexClient:
    def __init__(self) -> None:
        self._cache: dict[tuple[Any, ...], tuple[float, list[dict[str, Any]]]] = {}

    async def ports(
        self,
        west: float,
        south: float,
        east: float,
        north: float,
        *,
        center: tuple[float, float] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        key = (
            round(west, 2),
            round(south, 2),
            round(east, 2),
            round(north, 2),
            round(center[0], 2) if center else None,
            round(center[1], 2) if center else None,
            limit,
        )
        cached = self._cache.get(key)
        now = time.time()
        if cached and now - cached[0] < CACHE_TTL_S:
            return cached[1]

        params = {
            "where": "1=1",
            "geometry": f"{west},{south},{east},{north}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": OUT_FIELDS,
            "returnGeometry": "true",
            "outSR": "4326",
            "resultRecordCount": str(max(1, min(limit, 100))),
            "f": "geojson",
        }
        timeout = aiohttp.ClientTimeout(total=12)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(WPI_URL, params=params) as response:
                    if response.status != 200:
                        return []
                    result = parse_ports(await response.json(), center)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            return []
        self._cache[key] = (now, result)
        return result

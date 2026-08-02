"""Rate-limited world place search using OpenStreetMap Nominatim."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import aiohttp

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
CACHE_TTL_S = 86400.0


def parse_places(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    places = []
    for row in payload:
        try:
            lat = float(row["lat"])
            lon = float(row["lon"])
            raw_bounds = row.get("boundingbox") or []
            bounds = (
                [
                    [float(raw_bounds[0]), float(raw_bounds[2])],
                    [float(raw_bounds[1]), float(raw_bounds[3])],
                ]
                if len(raw_bounds) == 4
                else None
            )
        except (KeyError, TypeError, ValueError):
            continue
        places.append({
            "name": row.get("display_name") or row.get("name") or "Place",
            "lat": lat,
            "lon": lon,
            "bounds": bounds,
            "type": row.get("type"),
            "category": row.get("category"),
            "source": "OpenStreetMap Nominatim",
        })
    return places


class GeocodingClient:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._lock = asyncio.Lock()
        self._last_request = 0.0

    async def search(self, query: str) -> list[dict[str, Any]]:
        normalized = " ".join(query.split()).casefold()
        if len(normalized) < 2:
            return []
        cached = self._cache.get(normalized)
        now = time.time()
        if cached and now - cached[0] < CACHE_TTL_S:
            return cached[1]

        async with self._lock:
            cached = self._cache.get(normalized)
            now = time.time()
            if cached and now - cached[0] < CACHE_TTL_S:
                return cached[1]
            wait_s = max(0.0, 1.05 - (time.monotonic() - self._last_request))
            if wait_s:
                await asyncio.sleep(wait_s)
            timeout = aiohttp.ClientTimeout(total=12)
            headers = {
                "User-Agent": (
                    "Aegis-Maritime-Dashboard/1.0 "
                    "(https://github.com/abhijandyala/Aegis)"
                ),
                "Accept-Language": "en",
            }
            params = {
                "q": query,
                "format": "jsonv2",
                "limit": "5",
                "addressdetails": "1",
            }
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(
                        NOMINATIM_URL,
                        params=params,
                        headers=headers,
                    ) as response:
                        self._last_request = time.monotonic()
                        if response.status != 200:
                            return []
                        result = parse_places(await response.json())
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
                self._last_request = time.monotonic()
                return []
            self._cache[normalized] = (time.time(), result)
            return result

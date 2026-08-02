"""Cached NOAA GFS wind forcing for AIS-silent vessel prediction."""

from __future__ import annotations

import csv
import io
import math
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

WIND_DATASET = "NCEP_Global_Best"
WIND_ENDPOINT = (
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap/"
    f"{WIND_DATASET}.csv"
)


class WeatherConditionsClient:
    """Fetch the nearest current-hour NOAA GFS 10-meter wind vector."""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, int, str], tuple[float, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _cache_key(lat: float, lon: float) -> tuple[int, int, str]:
        now = datetime.now(timezone.utc)
        hour = now.replace(minute=0, second=0, microsecond=0).isoformat()
        return (round(lat * 2), round(lon * 2), hour)

    def cached_conditions(self, lat: float, lon: float) -> dict[str, Any] | None:
        cached = self._cache.get(self._cache_key(lat, lon))
        if cached and time.monotonic() - cached[0] < 1800:
            return cached[1]
        return None

    def current_conditions(self, lat: float, lon: float) -> dict[str, Any]:
        key = self._cache_key(lat, lon)
        cached = self._cache.get(key)
        if cached and time.monotonic() - cached[0] < 1800:
            return cached[1]
        with self._lock:
            cached = self._cache.get(key)
            if cached and time.monotonic() - cached[0] < 1800:
                return cached[1]
            payload = self._fetch(lat, lon)
            self._cache[key] = (time.monotonic(), payload)
            return payload

    def _fetch(self, lat: float, lon: float) -> dict[str, Any]:
        now = datetime.now(timezone.utc).replace(microsecond=0)
        longitude = lon % 360
        timestamp = now.isoformat().replace("+00:00", "Z")
        selectors = f"[(%s)][(%s)][(%s)]" % (
            timestamp,
            f"{lat:.4f}",
            f"{longitude:.4f}",
        )
        query = ",".join(
            f"{variable}{selectors}"
            for variable in ("ugrd10m", "vgrd10m")
        )
        url = f"{WIND_ENDPOINT}?{urllib.parse.quote(query, safe=',[]()')}"
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Aegis maritime safety dashboard"},
            )
            with urllib.request.urlopen(request, timeout=8) as response:
                text = response.read().decode("utf-8")
            rows = list(csv.DictReader(io.StringIO(text)))
            data_rows = [
                row
                for row in rows
                if row.get("time") != "UTC"
            ]
            if not data_rows:
                raise ValueError("no_wind_cell")
            row = data_rows[0]
            east = float(row["ugrd10m"])
            north = float(row["vgrd10m"])
            return {
                "configured": True,
                "available": True,
                "source": "NOAA Global Forecast System",
                "dataset": WIND_DATASET,
                "valid_at": row["time"],
                "center": {
                    "lat": float(row["latitude"]),
                    "lon": ((float(row["longitude"]) + 180) % 360) - 180,
                    "east_mps": round(east, 4),
                    "north_mps": round(north, 4),
                    "speed_mps": round(math.hypot(east, north), 4),
                    "bearing_deg": round(
                        math.degrees(math.atan2(east, north)) % 360,
                        1,
                    ),
                },
            }
        except Exception as exc:
            return {
                "configured": True,
                "available": False,
                "source": "NOAA Global Forecast System",
                "dataset": WIND_DATASET,
                "error": type(exc).__name__,
            }

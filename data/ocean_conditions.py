"""Small, cached Copernicus Marine current fields for dark-vessel prediction."""

from __future__ import annotations

import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any

CURRENT_DATASET = "cmems_mod_glo_phy_anfc_merged-uv_PT1H-i"
WAVE_DATASET = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"


def _rounded_or_none(value: object, digits: int) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


class OceanConditionsClient:
    """Fetch a compact surface-current grid only when an operator selects a vessel."""

    def __init__(self, username: str = "", password: str = "") -> None:
        self.username = username or os.environ.get("COPERNICUS_MARINE_USERNAME", "")
        self.password = password or os.environ.get("COPERNICUS_MARINE_PASSWORD", "")
        self._cache: dict[tuple[int, int, str], tuple[float, dict[str, Any]]] = {}
        self._cache_lock = threading.Lock()
        self._region_locks: dict[tuple[int, int, str], threading.Lock] = {}

    @property
    def configured(self) -> bool:
        return bool(self.username and self.password)

    @staticmethod
    def _cache_key(lat: float, lon: float) -> tuple[int, int, str]:
        now = datetime.now(timezone.utc)
        return (round(lat * 4), round(lon * 4), now.date().isoformat())

    def cached_current_grid(self, lat: float, lon: float) -> dict[str, Any] | None:
        with self._cache_lock:
            cached = self._cache.get(self._cache_key(lat, lon))
        if cached and time.monotonic() - cached[0] < 1800:
            return cached[1]
        return None

    def current_grid(self, lat: float, lon: float) -> dict[str, Any]:
        if not self.configured:
            return {
                "configured": False,
                "available": False,
                "source": "Copernicus Marine Service",
            }
        now = datetime.now(timezone.utc)
        key = self._cache_key(lat, lon)
        with self._cache_lock:
            cached = self._cache.get(key)
            region_lock = self._region_locks.setdefault(key, threading.Lock())
        if cached and time.monotonic() - cached[0] < 1800:
            return cached[1]
        with region_lock:
            with self._cache_lock:
                cached = self._cache.get(key)
            if cached and time.monotonic() - cached[0] < 1800:
                return cached[1]
            payload = self._fetch(lat, lon, now)
            with self._cache_lock:
                self._cache[key] = (time.monotonic(), payload)
            return payload

    def _fetch(self, lat: float, lon: float, now: datetime) -> dict[str, Any]:
        executor: ThreadPoolExecutor | None = None
        try:
            import copernicusmarine

            executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="aegis-ocean",
            )
            current_future = executor.submit(
                copernicusmarine.read_dataframe,
                dataset_id=CURRENT_DATASET,
                username=self.username,
                password=self.password,
                variables=[
                    "uo",
                    "vo",
                    "utide",
                    "vtide",
                    "vsdx",
                    "vsdy",
                    "utotal",
                    "vtotal",
                ],
                minimum_longitude=max(-179.99, lon - 0.35),
                maximum_longitude=min(179.99, lon + 0.35),
                minimum_latitude=max(-89.9, lat - 0.35),
                maximum_latitude=min(89.9, lat + 0.35),
                start_datetime=now - timedelta(hours=3),
                end_datetime=now + timedelta(hours=3),
                coordinates_selection_method="nearest",
                disable_progress_bar=True,
            )
            wave_future = executor.submit(
                copernicusmarine.read_dataframe,
                dataset_id=WAVE_DATASET,
                username=self.username,
                password=self.password,
                variables=["VHM0", "VTPK", "VMDR", "VSDX", "VSDY"],
                minimum_longitude=max(-179.99, lon - 0.35),
                maximum_longitude=min(179.99, lon + 0.35),
                minimum_latitude=max(-89.9, lat - 0.35),
                maximum_latitude=min(89.9, lat + 0.35),
                start_datetime=now - timedelta(hours=3),
                end_datetime=now + timedelta(hours=3),
                coordinates_selection_method="nearest",
                disable_progress_bar=True,
            )
            frame = current_future.result().reset_index()
            frame = frame.dropna(subset=["utotal", "vtotal"])
            if frame.empty:
                return {
                    "configured": True,
                    "available": False,
                    "source": "Copernicus Marine Service",
                    "dataset": CURRENT_DATASET,
                    "error": "no_ocean_cell",
                }
            times = frame["time"]
            if times.dt.tz is None:
                times = times.dt.tz_localize("UTC")
            else:
                times = times.dt.tz_convert("UTC")
            frame["_time_delta"] = (times - now).abs()
            nearest_time = frame.loc[frame["_time_delta"].idxmin(), "time"]
            frame = frame[frame["time"] == nearest_time]
            vectors = []
            for row in frame.itertuples():
                east = float(row.utotal)
                north = float(row.vtotal)
                vectors.append({
                    "lat": round(float(row.latitude), 5),
                    "lon": round(float(row.longitude), 5),
                    "east_mps": round(east, 4),
                    "north_mps": round(north, 4),
                    "speed_mps": round(math.hypot(east, north), 4),
                    "bearing_deg": round(math.degrees(math.atan2(east, north)) % 360, 1),
                    "components": {
                        "ocean_east_mps": _rounded_or_none(row.uo, 4),
                        "ocean_north_mps": _rounded_or_none(row.vo, 4),
                        "tide_east_mps": _rounded_or_none(row.utide, 4),
                        "tide_north_mps": _rounded_or_none(row.vtide, 4),
                        "stokes_east_mps": _rounded_or_none(row.vsdx, 4),
                        "stokes_north_mps": _rounded_or_none(row.vsdy, 4),
                    },
                })
            center = min(
                vectors,
                key=lambda vector: (
                    (vector["lat"] - lat) ** 2
                    + (vector["lon"] - lon) ** 2
                ),
            )
            vectors.sort(
                key=lambda vector: (
                    (vector["lat"] - lat) ** 2
                    + (vector["lon"] - lon) ** 2
                )
            )
            wave = {
                "available": False,
                "source": "Copernicus Marine Service",
                "dataset": WAVE_DATASET,
            }
            try:
                wave_frame = wave_future.result().reset_index()
                wave_frame = wave_frame.dropna(subset=["VHM0", "VMDR"])
                if not wave_frame.empty:
                    wave_times = wave_frame["time"]
                    if wave_times.dt.tz is None:
                        wave_times = wave_times.dt.tz_localize("UTC")
                    else:
                        wave_times = wave_times.dt.tz_convert("UTC")
                    wave_frame["_time_delta"] = (wave_times - now).abs()
                    wave_row = wave_frame.loc[
                        wave_frame["_time_delta"].idxmin()
                    ]
                    wave = {
                        "available": True,
                        "source": "Copernicus Marine Service",
                        "dataset": WAVE_DATASET,
                        "valid_at": wave_row["time"].isoformat(),
                        "height_m": round(float(wave_row.VHM0), 3),
                        "period_s": _rounded_or_none(wave_row.VTPK, 3),
                        "from_direction_deg": round(float(wave_row.VMDR), 1),
                        "stokes_east_mps": _rounded_or_none(wave_row.VSDX, 4),
                        "stokes_north_mps": _rounded_or_none(wave_row.VSDY, 4),
                    }
            except Exception as exc:
                wave["error"] = type(exc).__name__

            return {
                "configured": True,
                "available": True,
                "source": "Copernicus Marine Service",
                "dataset": CURRENT_DATASET,
                "valid_at": nearest_time.isoformat(),
                "center": center,
                "vectors": vectors[:25],
                "wave": wave,
            }
        except Exception as exc:
            return {
                "configured": True,
                "available": False,
                "source": "Copernicus Marine Service",
                "dataset": CURRENT_DATASET,
                "error": type(exc).__name__,
            }
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)

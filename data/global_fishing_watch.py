"""On-demand Global Fishing Watch vessel identity enrichment."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import aiohttp

GFW_VESSEL_SEARCH_URL = (
    "https://gateway.api.globalfishingwatch.org/v3/vessels/search"
)
GFW_DATASET = "public-global-vessel-identity:latest"
CACHE_TTL_S = 3600.0


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


class GlobalFishingWatchClient:
    def __init__(self, token: str):
        self.token = token
        self._cache: dict[int, tuple[float, dict[str, Any]]] = {}

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
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    GFW_VESSEL_SEARCH_URL,
                    params=params,
                    headers=headers,
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

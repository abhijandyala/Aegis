"""Cached OSM-derived navigable-water mask for trajectory constraints."""

from __future__ import annotations

import math
import os
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mapbox_vector_tile
from google.protobuf.message import DecodeError
from shapely.geometry import LineString, Point, Polygon, shape
from shapely.ops import transform, unary_union
from shapely.prepared import prep

EARTH_RADIUS_M = 6_371_008.8
DEFAULT_TILE_URL = (
    "https://tiles.openfreemap.org/planet/"
    "20260513_001001_pt/{z}/{x}/{y}.pbf"
)


@dataclass(frozen=True)
class _WaterTile:
    available: bool
    extent: int
    local_water: Any
    global_water: Any
    prepared_water: Any


def _world_coordinates(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    count = 2**zoom
    x = (lon + 180.0) / 360.0 * count
    clamped_lat = max(-85.05112878, min(85.05112878, lat))
    y = (
        1
        - math.asinh(math.tan(math.radians(clamped_lat))) / math.pi
    ) / 2 * count
    return x, y


def _latitude_from_world_y(world_y: float, zoom: int) -> float:
    count = 2**zoom
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * world_y / count))))


def _tile_ids(
    lat: float,
    lon: float,
    radius_km: float,
    zoom: int,
) -> list[tuple[int, int, int]]:
    count = 2**zoom
    lat_delta = radius_km / 111.32
    lon_delta = lat_delta / max(0.08, math.cos(math.radians(lat)))
    south = max(-85.0, lat - lat_delta)
    north = min(85.0, lat + lat_delta)
    west = lon - lon_delta
    east = lon + lon_delta
    _, north_y = _world_coordinates(north, lon, zoom)
    _, south_y = _world_coordinates(south, lon, zoom)
    y_start = max(0, int(math.floor(north_y)))
    y_end = min(count - 1, int(math.floor(south_y)))

    x_values: list[int] = []
    x_start = int(math.floor((west + 180.0) / 360.0 * count))
    x_end = int(math.floor((east + 180.0) / 360.0 * count))
    for raw_x in range(x_start, x_end + 1):
        x_values.append(raw_x % count)
    return [
        (zoom, x, y)
        for y in range(y_start, y_end + 1)
        for x in x_values
    ]


class NavigableWaterMask:
    """Fetch and cache OpenMapTiles water polygons without blocking simulations."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.tile_url = os.getenv("AEGIS_WATER_TILE_URL", DEFAULT_TILE_URL)
        self._tiles: dict[tuple[int, int, int], _WaterTile] = {}
        self._union_cache: OrderedDict[tuple[tuple[int, int, int], ...], Any] = (
            OrderedDict()
        )
        self._lock = threading.Lock()
        self._remote_disabled_until = 0.0

    def _cache_path(self, tile_id: tuple[int, int, int]) -> Path:
        zoom, x, y = tile_id
        return self.cache_dir / str(zoom) / str(x) / f"{y}.pbf"

    @staticmethod
    def _to_global_geometry(
        geometry: Any,
        tile_id: tuple[int, int, int],
        extent: int,
    ) -> Any:
        zoom, tile_x, tile_y = tile_id
        count = 2**zoom

        def convert(x: float, y: float, z: object | None = None):
            world_x = tile_x + x / extent
            # mapbox-vector-tile decodes coordinates with an upward Y axis;
            # Web Mercator tile rows increase downward.
            world_y = tile_y + (extent - y) / extent
            lon = world_x / count * 360.0 - 180.0
            lat = _latitude_from_world_y(world_y, zoom)
            return lon, lat

        return transform(convert, geometry)

    def _decode_tile(
        self,
        tile_id: tuple[int, int, int],
        payload: bytes,
    ) -> _WaterTile:
        decoded = mapbox_vector_tile.decode(payload)
        water_layer = decoded.get("water") or {}
        extent = int(water_layer.get("extent") or 4096)
        geometries = [
            shape(feature["geometry"])
            for feature in water_layer.get("features", [])
            if feature.get("geometry")
        ]
        waterway_layer = decoded.get("waterway") or {}
        waterway_geometries = [
            shape(feature["geometry"])
            for feature in waterway_layer.get("features", [])
            if feature.get("geometry")
        ]
        if waterway_geometries:
            zoom, _, y = tile_id
            center_lat = _latitude_from_world_y(y + 0.5, zoom)
            tile_width_m = (
                2 * math.pi * EARTH_RADIUS_M
                * max(0.08, math.cos(math.radians(center_lat)))
                / 2**zoom
            )
            channel_buffer = max(2.0, 25.0 / tile_width_m * extent)
            geometries.extend(
                geometry.buffer(channel_buffer, cap_style=2, join_style=2)
                for geometry in waterway_geometries
                if isinstance(geometry, LineString)
            )
        local_water = unary_union(geometries) if geometries else Polygon()
        # A small tolerance keeps valid AIS fixes at docks and breakwaters in water.
        zoom, _, y = tile_id
        center_lat = _latitude_from_world_y(y + 0.5, zoom)
        tile_width_m = (
            2 * math.pi * EARTH_RADIUS_M
            * max(0.08, math.cos(math.radians(center_lat)))
            / 2**zoom
        )
        tolerance = max(1.0, 15.0 / tile_width_m * extent)
        local_water = local_water.buffer(tolerance)
        global_water = self._to_global_geometry(local_water, tile_id, extent)
        return _WaterTile(
            available=True,
            extent=extent,
            local_water=local_water,
            global_water=global_water,
            prepared_water=prep(local_water),
        )

    def _load_tile(self, tile_id: tuple[int, int, int]) -> _WaterTile:
        with self._lock:
            existing = self._tiles.get(tile_id)
        if existing is not None:
            return existing
        path = self._cache_path(tile_id)
        try:
            if path.exists():
                payload = path.read_bytes()
            else:
                with self._lock:
                    remote_disabled = time.monotonic() < self._remote_disabled_until
                if remote_disabled:
                    raise urllib.error.URLError("water tile source temporarily unavailable")
                url = self.tile_url.format(
                    z=tile_id[0],
                    x=tile_id[1],
                    y=tile_id[2],
                )
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Aegis-Maritime/1.0"},
                )
                try:
                    with urllib.request.urlopen(request, timeout=8) as response:
                        payload = response.read()
                except (OSError, urllib.error.URLError):
                    with self._lock:
                        self._remote_disabled_until = time.monotonic() + 60
                    raise
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(payload)
                temporary.replace(path)
            tile = self._decode_tile(tile_id, payload)
        except (
            OSError,
            ValueError,
            urllib.error.URLError,
            DecodeError,
        ):
            tile = _WaterTile(
                available=False,
                extent=4096,
                local_water=Polygon(),
                global_water=Polygon(),
                prepared_water=None,
            )
        if tile.available:
            with self._lock:
                self._tiles[tile_id] = tile
        return tile

    def prime_region(
        self,
        lat: float,
        lon: float,
        radius_km: float,
    ) -> dict[str, int | bool | str]:
        radius = max(10.0, min(700.0, float(radius_km)))
        plan = (
            (11, min(radius, 20.0)),
            (10, min(radius, 75.0)),
            (8, radius),
        )
        tile_ids: list[tuple[int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()
        for zoom, level_radius in plan:
            for tile_id in _tile_ids(lat, lon, level_radius, zoom):
                if tile_id not in seen:
                    seen.add(tile_id)
                    tile_ids.append(tile_id)
        with ThreadPoolExecutor(max_workers=12) as executor:
            tiles = list(executor.map(self._load_tile, tile_ids))
        available = sum(tile.available for tile in tiles)
        return {
            "available": available > 0,
            "tiles_available": available,
            "tiles_requested": len(tile_ids),
            "source": "OpenStreetMap · OpenMapTiles · OpenFreeMap",
        }

    def status(self, lat: float, lon: float) -> dict[str, Any]:
        for zoom in (11, 10, 8):
            world_x, world_y = _world_coordinates(lat, lon, zoom)
            tile_id = (zoom, int(world_x) % 2**zoom, int(world_y))
            with self._lock:
                tile = self._tiles.get(tile_id)
            if tile is None or not tile.available:
                continue
            point = Point(
                (world_x - math.floor(world_x)) * tile.extent,
                (1 - (world_y - math.floor(world_y))) * tile.extent,
            )
            return {
                "available": True,
                "on_land": not tile.prepared_water.covers(point),
                "source": "OpenStreetMap · OpenMapTiles · OpenFreeMap",
                "zoom": zoom,
            }
        return {"available": False, "on_land": False}

    @property
    def has_tiles(self) -> bool:
        with self._lock:
            return any(tile.available for tile in self._tiles.values())

    def segment_crosses_land(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> bool:
        if not self.has_tiles:
            return False
        line = LineString([
            (start[1], start[0]),
            (end[1], end[0]),
        ])
        for zoom in (11, 10, 8):
            start_x, start_y = _world_coordinates(start[0], start[1], zoom)
            end_x, end_y = _world_coordinates(end[0], end[1], zoom)
            count = 2**zoom
            x_start = math.floor(min(start_x, end_x))
            x_end = math.floor(max(start_x, end_x))
            y_start = max(0, math.floor(min(start_y, end_y)))
            y_end = min(count - 1, math.floor(max(start_y, end_y)))
            tile_ids = tuple(
                (zoom, x % count, y)
                for y in range(y_start, y_end + 1)
                for x in range(x_start, x_end + 1)
            )
            with self._lock:
                tiles = [self._tiles.get(tile_id) for tile_id in tile_ids]
                cached_union = self._union_cache.get(tile_ids)
            if not tile_ids or not all(tile and tile.available for tile in tiles):
                continue
            if cached_union is None:
                cached_union = unary_union([
                    tile.global_water
                    for tile in tiles
                    if tile is not None
                ])
                with self._lock:
                    self._union_cache[tile_ids] = cached_union
                    self._union_cache.move_to_end(tile_ids)
                    while len(self._union_cache) > 256:
                        self._union_cache.popitem(last=False)
            else:
                with self._lock:
                    self._union_cache.move_to_end(tile_ids)
            return not cached_union.covers(line)

        mean_lat = math.radians((start[0] + end[0]) / 2)
        north_m = (end[0] - start[0]) * 111_320
        east_m = (
            (end[1] - start[1])
            * 111_320
            * max(0.08, math.cos(mean_lat))
        )
        midpoint = (
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2,
        )
        coverage = [
            self.status(*point)
            for point in (start, midpoint, end)
        ]
        if any(
            status.get("available") and status.get("on_land")
            for status in coverage[1:]
        ):
            return True
        zooms = [
            int(status["zoom"])
            for status in coverage
            if status.get("available") and status.get("zoom")
        ]
        if not zooms:
            return False
        interval_m = 125 if max(zooms) >= 11 else 300 if max(zooms) >= 10 else 750
        checks = max(1, math.ceil(math.hypot(east_m, north_m) / interval_m))
        for index in range(1, checks + 1):
            fraction = index / checks
            point = (
                start[0] + (end[0] - start[0]) * fraction,
                start[1] + (end[1] - start[1]) * fraction,
            )
            status = self.status(*point)
            if status["available"] and status["on_land"]:
                return True
        return False

    def clip_polygon(
        self,
        polygon: list[list[float]],
    ) -> list[dict[str, Any]] | None:
        if len(polygon) < 3:
            return None
        confidence = Polygon([(point[1], point[0]) for point in polygon])
        if confidence.is_empty or not confidence.is_valid:
            return None
        min_lon, min_lat, max_lon, max_lat = confidence.bounds
        selected: list[Any] = []
        with self._lock:
            tile_items = list(self._tiles.items())
        for zoom in (11, 10, 8):
            level_geometries: list[Any] = []
            expected = _tile_ids(
                (min_lat + max_lat) / 2,
                (min_lon + max_lon) / 2,
                max(
                    (max_lat - min_lat) * 111.32 / 2,
                    (max_lon - min_lon)
                    * 111.32
                    * max(0.08, math.cos(math.radians((min_lat + max_lat) / 2)))
                    / 2,
                    1,
                ),
                zoom,
            )
            # Large open-water envelopes can span hundreds of detailed vector
            # tiles. Omitting that band is safer and far faster than drawing an
            # un-clipped hull over land.
            if len(expected) > 36:
                if zoom == 8:
                    return []
                continue
            available_ids = {
                tile_id
                for tile_id, tile in tile_items
                if tile_id[0] == zoom and tile.available
            }
            if expected and all(tile_id in available_ids for tile_id in expected):
                level_geometries = [
                    tile.global_water
                    for tile_id, tile in tile_items
                    if tile_id in set(expected) and tile.available
                ]
            if level_geometries:
                selected = level_geometries
                break
        if not selected:
            return None
        polygons: list[Any] = []
        for water_geometry in selected:
            if not water_geometry.intersects(confidence):
                continue
            clipped = confidence.intersection(water_geometry)
            if clipped.geom_type == "Polygon":
                polygons.append(clipped)
            else:
                polygons.extend(
                    item
                    for item in getattr(clipped, "geoms", [])
                    if item.geom_type == "Polygon"
                )
        result: list[dict[str, Any]] = []
        for item in polygons:
            if item.is_empty or item.geom_type != "Polygon":
                continue
            exterior = [[lat, lon] for lon, lat in item.exterior.coords]
            holes = [
                [[lat, lon] for lon, lat in ring.coords]
                for ring in item.interiors
            ]
            result.append({"exterior": exterior, "holes": holes})
        return result

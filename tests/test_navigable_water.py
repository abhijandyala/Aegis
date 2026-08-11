from pathlib import Path

import mapbox_vector_tile
from shapely.geometry import box, mapping

from data.navigable_water import (
    NavigableWaterMask,
    _latitude_from_world_y,
)


def _coordinate(
    tile_id: tuple[int, int, int],
    local_x: float,
    local_y: float,
) -> tuple[float, float]:
    zoom, tile_x, tile_y = tile_id
    count = 2**zoom
    lon = (tile_x + local_x / 4096) / count * 360 - 180
    lat = _latitude_from_world_y(tile_y + (4096 - local_y) / 4096, zoom)
    return lat, lon


def test_vector_water_tile_blocks_land_and_clips_confidence(tmp_path: Path):
    mask = NavigableWaterMask(tmp_path)
    tile_id = (8, 128, 128)
    payload = mapbox_vector_tile.encode({
        "name": "water",
        "features": [{
            "geometry": mapping(box(0, 0, 2048, 4096)),
            "properties": {"class": "ocean"},
        }],
    })
    mask._tiles[tile_id] = mask._decode_tile(tile_id, payload)

    water = _coordinate(tile_id, 1024, 2048)
    land = _coordinate(tile_id, 3072, 2048)
    assert mask.status(*water)["on_land"] is False
    assert mask.status(*land)["on_land"] is True
    assert mask.segment_crosses_land(water, land) is True

    confidence = [
        list(_coordinate(tile_id, 500, 1000)),
        list(_coordinate(tile_id, 3500, 1000)),
        list(_coordinate(tile_id, 3500, 3000)),
        list(_coordinate(tile_id, 500, 3000)),
    ]
    clipped = mask.clip_polygon(confidence)
    assert clipped
    water_edge_lon = _coordinate(tile_id, 2052, 2048)[1]
    assert max(
        point[1]
        for polygon in clipped
        for point in polygon["exterior"]
    ) <= water_edge_lon

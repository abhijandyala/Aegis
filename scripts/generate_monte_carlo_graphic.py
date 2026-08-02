"""Generate the compact Monte Carlo trajectory graphic used by Aegis."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data.dark_prediction as prediction_module


WIDTH = 360
HEIGHT = 203
EXPORT_SCALE = 4
OUTPUT = Path(__file__).resolve().parents[1] / "web" / "assets" / "monte-carlo-simulation.svg"


def _capture_prediction() -> tuple[dict, list[dict]]:
    captured: list[dict] = []
    original_cluster = prediction_module._cluster

    def capture_cluster(samples, count, origin):
        captured.extend(samples)
        return original_cluster(samples, count, origin)

    prediction_module._cluster = capture_cluster
    try:
        result = prediction_module.predict_dark_vessel({
            "mmsi": 123456789,
            "lat": 37.7,
            "lon": -123.0,
            "last_seen": 1_700_000_000.0,
            "age_s": 180.0,
            "dark": True,
            "course": 95.0,
            "heading": 96.0,
            "speed_kn": 12.0,
            "rate_of_turn": 0.1,
            "history": [
                [37.7, -123.02],
                [37.7, -123.01],
                [37.7, -123.0],
            ],
        })
    finally:
        prediction_module._cluster = original_cluster
    return result, captured


def _projector(samples: list[dict]):
    origin_lat, origin_lon = samples[0]["path"][0]
    cos_lat = math.cos(math.radians(origin_lat))
    bearing = math.radians(95)

    def cross_track_meters(point):
        lat, lon = point
        east = (lon - origin_lon) * 111_320 * cos_lat
        north = (lat - origin_lat) * 111_320
        return east * math.cos(bearing) - north * math.sin(bearing)

    offsets = [
        cross_track_meters(point)
        for sample in samples
        for point in sample["path"]
    ]
    extent = max(1000, max(abs(offset) for offset in offsets) * 1.12)
    max_step = max(len(sample["path"]) for sample in samples) - 1

    left, right, top, bottom = 4, WIDTH - 4, 4, HEIGHT - 4

    def project(point, step):
        x = left + step / max_step * (right - left)
        y = (top + bottom) / 2 - cross_track_meters(point) / extent * (bottom - top) / 2
        return x, y

    return project, extent / 1852


def _polyline(points, project) -> str:
    return " ".join(
        f"{x:.1f},{y:.1f}"
        for x, y in (project(point, index) for index, point in enumerate(points))
    )


def build_svg() -> str:
    _, samples = _capture_prediction()
    project, _ = _projector(samples)
    ensemble_palette = (
        "#2f6fff", "#ec3f68", "#21a875", "#8f4fd4", "#f28b24",
        "#00a6c7", "#d737b4", "#729d21", "#7657ff", "#de5454",
    )

    raw_paths = []
    for index, sample in enumerate(samples):
        color = ensemble_palette[index % len(ensemble_palette)]
        raw_paths.append(
            f'<polyline points="{_polyline(sample["path"], project)}" '
            f'fill="none" stroke="{color}" stroke-width=".38" '
            'stroke-opacity=".24" stroke-linecap="round"/>'
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH * EXPORT_SCALE}" height="{HEIGHT * EXPORT_SCALE}" viewBox="0 0 {WIDTH} {HEIGHT}" shape-rendering="geometricPrecision">
<defs>
  <linearGradient id="paper" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#ffffff"/>
    <stop offset="1" stop-color="#f1f3f5"/>
  </linearGradient>
</defs>
<rect width="360" height="203" fill="url(#paper)"/>
<g stroke="#86919c" stroke-opacity=".26" stroke-width=".42">
  <path d="M4 4V199 M74.4 4V199 M144.8 4V199 M215.2 4V199 M285.6 4V199 M356 4V199"/>
  <path d="M4 4H356 M4 43H356 M4 82H356 M4 121H356 M4 160H356 M4 199H356"/>
</g>
<g>{''.join(raw_paths)}</g>
</svg>"""


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build_svg(), encoding="utf-8")
    print(OUTPUT)

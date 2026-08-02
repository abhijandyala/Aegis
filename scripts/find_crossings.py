"""Locate vessel crossings in an extracted AIS window, or scan candidate
time windows so you can pick the demo window fast instead of eyeballing.

A crossing = two vessels within 500 m whose range is actually closing.

Report crossings for a pack's configured window:

    .venv/bin/python scripts/find_crossings.py --pack s01_dark_in_sanctuary

Scan a whole extracted day/window CSV for the best 2 h slice:

    .venv/bin/python scripts/find_crossings.py --pack s01_dark_in_sanctuary --scan
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from data.scenario import (  # noqa: E402
    SCENARIOS_DIR,
    _ingest_ais,
    _parse_utc,
    find_crossings,
)


def _fmt(t_abs: float) -> str:
    return datetime.fromtimestamp(t_abs, tz=timezone.utc).strftime("%H:%M:%SZ")


def report(pack: dict, pack_dir: str, t0: float, t1: float, label: str = "") -> int:
    interval = float(pack.get("frame_interval_s", 30.0))
    origin = (float(pack["origin"]["lat"]), float(pack["origin"]["lon"]))
    csv_path = os.path.join(pack_dir, pack["ais_csv"])
    tracks = _ingest_ais(csv_path, pack, t0, t1, interval, origin)
    crossings = find_crossings(tracks, interval)
    print(f"{label or 'window'} {_fmt(t0)}–{_fmt(t1)}: "
          f"{len(tracks)} vessels, {len(crossings)} crossings")
    for c in crossings:
        a, b = c["pair"]
        print(f"  {a} x {b}  t={c['t_start']:6.0f}–{c['t_end']:6.0f}s  "
              f"min {c['min_dist_m']:5.0f} m  closing {c['closing_mps']:4.1f} m/s")
    return len(crossings)


def scan(pack: dict, pack_dir: str, window_s: float, step_s: float) -> None:
    """Slide a window across the whole extracted CSV; rank by crossing count."""
    csv_path = os.path.join(pack_dir, pack["ais_csv"])
    interval = float(pack.get("frame_interval_s", 30.0))
    origin = (float(pack["origin"]["lat"]), float(pack["origin"]["lon"]))

    # Full extent of the extracted file, regardless of the pack's window.
    import csv as _csv
    with open(csv_path, newline="", encoding="utf-8") as f:
        times = [_parse_utc(r["BaseDateTime"]) for r in _csv.DictReader(f)]
    lo, hi = min(times), max(times)

    results = []
    t = lo
    while t + window_s <= hi:
        tracks = _ingest_ais(csv_path, pack, t, t + window_s, interval, origin)
        crossings = find_crossings(tracks, interval)
        results.append((len(crossings), len(tracks), t))
        print(f"  {_fmt(t)}  vessels {len(tracks):3d}  crossings {len(crossings):3d}")
        t += step_s
    results.sort(reverse=True)
    best = results[0]
    print(f"\nbest window: start {_fmt(best[2])} "
          f"({best[1]} vessels, {best[0]} crossings)")
    print(f'  "time_window": {{"start": '
          f'"{datetime.fromtimestamp(best[2], tz=timezone.utc).isoformat()}", '
          f'"end": "{datetime.fromtimestamp(best[2] + window_s, tz=timezone.utc).isoformat()}"}}')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pack", required=True)
    p.add_argument("--scan", action="store_true",
                   help="slide a window over the whole CSV instead of using the pack window")
    p.add_argument("--window-min", type=float, default=120.0)
    p.add_argument("--step-min", type=float, default=30.0)
    args = p.parse_args()

    pack_path = args.pack
    if not os.path.isfile(pack_path):
        pack_path = os.path.join(SCENARIOS_DIR, args.pack, "pack.json")
    with open(pack_path, encoding="utf-8") as f:
        pack = json.load(f)
    pack_dir = os.path.dirname(os.path.abspath(pack_path))

    if args.scan:
        scan(pack, pack_dir, args.window_min * 60.0, args.step_min * 60.0)
    else:
        report(pack, pack_dir,
               _parse_utc(pack["time_window"]["start"]),
               _parse_utc(pack["time_window"]["end"]))


if __name__ == "__main__":
    main()

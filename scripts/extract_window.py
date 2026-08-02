"""Extract a scenario pack's AIS window from a raw MarineCadastre daily zip.

Streams the CSV straight out of the zip (never inflates 4 GB to disk), keeps
only rows inside the pack's bbox and time window, and writes the small cached
CSV the pack points at via ``ais_csv``. Run once per pack; the loader then
never touches the raw file again.

    .venv/bin/python scripts/extract_window.py \
        --zip data/raw/AIS_2024_06_15.zip --pack s01_dark_in_sanctuary
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
import zipfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from data.scenario import SCENARIOS_DIR, _parse_utc  # noqa: E402

# Columns the pipeline needs. Dropping the rest (VesselName, IMO, CallSign,
# dimensions, cargo) keeps the cached window small and avoids carrying more
# identity than necessary; MMSI stays because ingest strips it into ground truth.
KEEP = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading"]


def extract(zip_path: str, pack_path: str, pad_s: float = 0.0) -> str:
    with open(pack_path, "r", encoding="utf-8") as f:
        pack = json.load(f)
    pack_dir = os.path.dirname(os.path.abspath(pack_path))
    bbox = pack["bbox"]
    t0 = _parse_utc(pack["time_window"]["start"]) - pad_s
    t1 = _parse_utc(pack["time_window"]["end"]) + pad_s
    out_path = os.path.join(pack_dir, pack.get("ais_csv", "ais_window.csv"))

    lon_min, lon_max = bbox["lon_min"], bbox["lon_max"]
    lat_min, lat_max = bbox["lat_min"], bbox["lat_max"]

    n_in = n_out = 0
    start = time.perf_counter()
    with zipfile.ZipFile(zip_path) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(member) as raw, open(out_path, "w", newline="", encoding="utf-8") as out:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
            writer = csv.DictWriter(out, fieldnames=KEEP, extrasaction="ignore")
            writer.writeheader()
            for row in reader:
                n_in += 1
                try:
                    lat = float(row["LAT"])
                    lon = float(row["LON"])
                except (ValueError, KeyError):
                    continue
                if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                    continue
                if not (t0 <= _parse_utc(row["BaseDateTime"]) <= t1):
                    continue
                writer.writerow(row)
                n_out += 1

    print(f"{n_out} of {n_in} rows kept in {time.perf_counter() - start:.0f}s "
          f"-> {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--zip", required=True, help="raw MarineCadastre daily zip")
    p.add_argument("--pack", required=True,
                   help="pack id under scenarios/ or path to pack.json")
    p.add_argument("--pad-s", type=float, default=300.0,
                   help="extra seconds either side of the window (interp headroom)")
    args = p.parse_args()

    pack_path = args.pack
    if not os.path.isfile(pack_path):
        pack_path = os.path.join(SCENARIOS_DIR, args.pack, "pack.json")
    extract(args.zip, pack_path, pad_s=args.pad_s)


if __name__ == "__main__":
    main()

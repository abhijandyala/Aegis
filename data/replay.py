"""Replay engine: pace a materialised Scenario through wall time.

The full scenario is materialised once by ``load_scenario`` (all frames in
memory), so ``seek`` is an index assignment and ``reset`` never re-parses.
The engine yields plain :class:`~data.contracts.Frame` objects; whoever
consumes them (the Aegis pipeline, a test, a benchmark) decides what a
frame means. Speed only changes the sleep between yields, never the frames.
"""

from __future__ import annotations

import time
from typing import Iterator, Optional

from data.contracts import Frame
from data.scenario import Scenario, load_scenario

__all__ = ["ReplayEngine", "SPEEDS"]

SPEEDS = (1.0, 10.0, 60.0)


class ReplayEngine:
    """Frame cursor over a Scenario with wall-clock pacing.

    - ``speed``: replay multiplier (1x / 10x / 60x, any positive float works).
    - ``seek(frame_idx)``: instant; just moves the cursor.
    - ``reset()``: rewind to frame 0 without re-parsing anything.
    - ``play()``: generator yielding frames paced by ``speed``. Pass
      ``realtime=False`` to iterate as fast as possible (tests, benchmarks).
    """

    def __init__(self, scenario: Scenario, speed: float = 1.0) -> None:
        self.scenario = scenario
        self.speed = speed
        self._cursor = 0

    @classmethod
    def from_pack(cls, name_or_path: str, speed: float = 1.0) -> "ReplayEngine":
        return cls(load_scenario(name_or_path), speed)

    # ------------------------------------------------------------- properties

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, value: float) -> None:
        value = float(value)
        if value <= 0.0:
            raise ValueError(f"speed must be positive, got {value}")
        self._speed = value

    @property
    def cursor(self) -> int:
        """Index of the next frame ``play()``/``step()`` will emit."""
        return self._cursor

    @property
    def n_frames(self) -> int:
        return self.scenario.n_frames

    @property
    def finished(self) -> bool:
        return self._cursor >= self.n_frames

    # ---------------------------------------------------------------- control

    def seek(self, frame_idx: int) -> Frame:
        """Jump the cursor. Instant: frames are pre-materialised, nothing is
        re-parsed. Returns the frame at the new cursor."""
        if not 0 <= frame_idx < self.n_frames:
            raise IndexError(
                f"frame {frame_idx} out of range [0, {self.n_frames})"
            )
        self._cursor = frame_idx
        return self.scenario.frames[frame_idx]

    def reset(self) -> None:
        """Rewind to the start. No I/O, no parsing."""
        self._cursor = 0

    def step(self) -> Optional[Frame]:
        """Emit the next frame immediately (no pacing), or None when finished."""
        if self.finished:
            return None
        frame = self.scenario.frames[self._cursor]
        self._cursor += 1
        return frame

    def play(self, realtime: bool = True) -> Iterator[Frame]:
        """Yield frames from the cursor to the end.

        With ``realtime=True`` the gap between yields is
        ``frame_interval_s / speed`` (re-read every frame, so changing
        ``engine.speed`` mid-flight takes effect immediately). ``seek()`` from
        another coroutine/thread also takes effect on the next yield because
        the cursor is re-read each iteration.
        """
        interval = self.scenario.frame_interval_s
        while not self.finished:
            start = time.monotonic()
            frame = self.scenario.frames[self._cursor]
            self._cursor += 1
            yield frame
            if realtime and not self.finished:
                delay = interval / self._speed - (time.monotonic() - start)
                if delay > 0:
                    time.sleep(delay)


def _main() -> None:
    """Smoke run: load a pack, print acceptance-relevant numbers, dry-run replay."""
    import argparse

    parser = argparse.ArgumentParser(description="Replay a scenario pack (dry run)")
    parser.add_argument("pack", help="pack id under scenarios/ or path to pack.json")
    parser.add_argument("--speed", type=float, default=60.0)
    parser.add_argument("--realtime", action="store_true",
                        help="pace frames by wall clock instead of dry-running")
    args = parser.parse_args()

    t_load = time.perf_counter()
    engine = ReplayEngine.from_pack(args.pack, speed=args.speed)
    load_s = time.perf_counter() - t_load
    s = engine.scenario

    n_meas = sum(len(f.measurements) for f in s.frames)
    n_radar = sum(1 for f in s.frames for m in f.measurements if m.source == "radar")
    print(f"pack        {s.pack_id} — {s.name}")
    print(f"load        {load_s:.2f} s ({s.n_frames} frames, {s.duration_s/60:.0f} min window)")
    print(f"vessels     {s.n_vessels}  measurements {n_meas} ({n_radar} radar)")
    print(f"geofences   {[f.fence_id for f in s.geofences]}")
    print(f"events      {[(e.t, e.kind, e.actor) for e in s.events]}")

    t_seek = time.perf_counter()
    engine.seek(s.n_frames // 2)
    engine.reset()
    print(f"seek+reset  {(time.perf_counter() - t_seek)*1e6:.0f} us")

    t_run = time.perf_counter()
    for frame in engine.play(realtime=args.realtime):
        if args.realtime:
            print(f"  t={frame.t:7.0f}s  frame {frame.idx:4d}  "
                  f"{len(frame.measurements):3d} meas  "
                  f"{[e.kind for e in frame.events] or ''}")
    print(f"full replay {time.perf_counter() - t_run:.3f} s at "
          f"{'realtime x' + str(args.speed) if args.realtime else 'unpaced'}")


if __name__ == "__main__":
    _main()

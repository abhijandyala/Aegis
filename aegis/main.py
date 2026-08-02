"""Pure-Python Aegis runtime pipeline."""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from data.replay import ReplayEngine
from data.scenario import load_scenario

from .alerts import AlertScribe, Briefer
from .driver import ScenarioDriver
from .fusion import Fusion, assoc_provenance
from .graph import Mission, alerts_of, build_mission, tracks_of
from .render import Renderer, frame_delta


def run_pack(
    pack_id: str,
    max_frames: int = -1,
    seed_offset: int = 1,
    assoc_mode: str = "global",
) -> dict[str, Any]:
    scenario = load_scenario(pack_id)
    engine = ReplayEngine(scenario)
    mission = build_mission(scenario)
    mission.meta["rng"] = np.random.default_rng(
        int(scenario.expected.get("seed", 0)) + 20260726 + seed_offset
    )
    deltas = []
    for frame in engine.play(realtime=False):
        if max_frames >= 0 and frame.idx >= max_frames:
            break
        deltas.append(
            run_frame(mission, scenario, frame, assoc_mode=assoc_mode)
        )
    return {
        "pack_id": scenario.pack_id,
        "deltas": deltas,
        "mission": mission,
        "scenario": scenario,
        "assoc_source": assoc_provenance(assoc_mode),
    }


def run_frame(
    mission: Mission,
    scenario: Any,
    frame: Any,
    assoc_mode: str = "global",
) -> dict[str, Any]:
    delta, _pairs = run_frame_scored(
        mission, scenario, frame, assoc_mode=assoc_mode
    )
    return delta


def run_frame_scored(
    mission: Mission,
    scenario: Any,
    frame: Any,
    assoc_mode: str = "global",
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    driver = ScenarioDriver(scenario=scenario, frame=frame).run(mission)
    fusion = Fusion(
        t=frame.t,
        dt=scenario.frame_interval_s,
        meas=driver.effective,
        assoc_mode=assoc_mode,
    ).run(mission)
    for alert in fusion.new_alerts:
        AlertScribe().compose(alert)
    renderer = Renderer(origin=scenario.origin).run(mission)
    delta = frame_delta(
        mission, driver, fusion, renderer, include_zones=frame.idx == 0
    )
    pairs = [
        (track_id, scenario.ground_truth[assignment[0].meas_id])
        for track_id, assignment in fusion.assigned.items()
    ]
    return delta, pairs


def summarize(result: dict[str, Any]) -> None:
    mission = result["mission"]
    alerts = sorted(alerts_of(mission), key=lambda alert: alert.t)
    print(f"pack        {result['pack_id']}")
    print(f"associator  {result['assoc_source']}")
    print(f"frames      {len(result['deltas'])}")
    print(f"tracks live {len(tracks_of(mission))}")
    print(f"alerts      {len(alerts)}")
    for alert in alerts[:12]:
        print(
            f"  {alert.severity.upper():8s} {alert.headline}  "
            f"[{alert.model_used}]"
        )
    brief = Briefer().assemble(mission).brief
    print(f"brief       [{brief.model_used}] {brief.text[:300]}")


def main() -> None:
    pack = os.environ.get("AEGIS_PACK", "s02_synthetic_demo")
    max_frames = int(os.environ.get("AEGIS_FRAMES", "-1"))
    started = time.perf_counter()
    result = run_pack(pack, max_frames=max_frames)
    summarize(result)
    print(
        f"pipeline    {len(result['deltas'])} frames in "
        f"{time.perf_counter() - started:.2f} s"
    )


if __name__ == "__main__":
    main()

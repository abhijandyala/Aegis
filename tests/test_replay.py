"""Phase 1 acceptance tests for the replay engine (data/replay.py)."""

import time

import pytest

from data.replay import ReplayEngine
from data.scenario import load_scenario


@pytest.fixture(scope="module")
def scenario():
    return load_scenario("s02_synthetic_demo")


def test_full_unpaced_replay_yields_every_frame(scenario):
    engine = ReplayEngine(scenario)
    frames = list(engine.play(realtime=False))
    assert [f.idx for f in frames] == list(range(scenario.n_frames))
    assert engine.finished


def test_seek_is_instant_and_does_not_reparse(scenario):
    engine = ReplayEngine(scenario)
    targets = [100, 3, 77, 0, scenario.n_frames - 1] * 200
    start = time.perf_counter()
    for idx in targets:
        frame = engine.seek(idx)
        assert frame.idx == idx
    elapsed = time.perf_counter() - start
    # 1000 seeks in well under 10 ms means no per-seek parsing whatsoever.
    assert elapsed < 0.01


def test_reset_under_two_seconds(scenario):
    engine = ReplayEngine(scenario)
    list(engine.play(realtime=False))
    start = time.perf_counter()
    engine.reset()
    assert time.perf_counter() - start < 2.0
    assert engine.cursor == 0
    assert engine.step().idx == 0


def test_seek_bounds(scenario):
    engine = ReplayEngine(scenario)
    with pytest.raises(IndexError):
        engine.seek(-1)
    with pytest.raises(IndexError):
        engine.seek(scenario.n_frames)


def test_speed_paces_wall_clock(scenario):
    # 5 frames of a 30 s interval at 300x -> four 0.1 s gaps ~ 0.4 s wall time.
    engine = ReplayEngine(scenario, speed=300.0)
    start = time.perf_counter()
    for i, _ in enumerate(engine.play(realtime=True)):
        if i == 4:
            break
    elapsed = time.perf_counter() - start
    assert 0.3 < elapsed < 1.2


def test_speed_change_mid_flight(scenario):
    engine = ReplayEngine(scenario, speed=1.0)
    gen = engine.play(realtime=True)
    next(gen)                       # first frame yields immediately
    engine.speed = 600.0            # 30 s interval -> 50 ms gaps
    start = time.perf_counter()
    next(gen)
    assert time.perf_counter() - start < 1.0


def test_invalid_speed_rejected(scenario):
    with pytest.raises(ValueError):
        ReplayEngine(scenario, speed=0.0)
    engine = ReplayEngine(scenario)
    with pytest.raises(ValueError):
        engine.speed = -5.0


def test_measurements_feed_the_tracker_contract(scenario):
    """End-to-end shape check: replay frames drive Raj's filter untouched."""
    from tracker.kalman import CVKalman

    ghost = [
        m for f in scenario.frames for m in f.measurements
        if scenario.ground_truth[m.meas_id] == "ghost_1" and m.source == "ais"
    ]
    kf = CVKalman.init_two_point(ghost[0].z(), ghost[0].t,
                                 ghost[1].z(), ghost[1].t, ghost[1].R())
    for m in ghost[2:]:
        kf.predict(m.t - kf.t)
        kf.update(m.z(), m.R())
    # Converged track should sit within a few metres of the last noisy fix.
    err = ((kf.state[0] - ghost[-1].x) ** 2 + (kf.state[1] - ghost[-1].y) ** 2) ** 0.5
    assert err < 30.0

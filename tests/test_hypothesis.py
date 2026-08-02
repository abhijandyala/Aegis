"""Phase 3 acceptance tests for ambiguity detection and hypothesis lifecycle."""

from __future__ import annotations

import math

import numpy as np
import pytest

import aegis.hypothesis as hypothesis
from tracker.ambiguity import (
    AMBIGUITY_DELTA_NATS,
    AmbiguityReport,
    detect_ambiguity,
    solve_with_forbidden,
    total_cost,
)
from tracker.assoc import DEFAULT_PARAMS, Track, associate_global, big_cost
from tracker.kalman import CVKalman

# Radar-grade measurement noise, 50 m 1-sigma.
R_RADAR = np.diag([2500.0, 2500.0])


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_track(tid: int, x: float, y: float, pos_var: float = 400.0) -> Track:
    """A converged, stationary track at (x, y) with an isotropic position cov."""
    kf = CVKalman(q=0.5)
    kf.state[:] = [x, y, 0.0, 0.0]
    kf.cov[:] = np.diag([pos_var, pos_var, 25.0, 25.0])
    return Track(id=tid, kf=kf)


def crossing_frame() -> tuple[list[Track], np.ndarray]:
    """Two vessels at the moment of crossing.

    The tracks are 20 m apart and the two measurements straddle them
    symmetrically, so "T0 took the northern plot" and "T0 took the southern
    plot" are within a nat of each other -- exactly the geometry that swaps
    track IDs for good in a single-hypothesis tracker.
    """
    tracks = [make_track(0, 0.0, -10.0), make_track(1, 0.0, 10.0)]
    zs = np.array([[0.0, -40.0], [0.0, 40.0]])
    return tracks, zs


def separated_frame() -> tuple[list[Track], np.ndarray]:
    """Two vessels 5 km apart, each sitting on its own plot: no contest."""
    tracks = [make_track(0, 0.0, 0.0), make_track(1, 5000.0, 0.0)]
    zs = np.array([[0.0, 0.0], [5000.0, 0.0]])
    return tracks, zs


@pytest.fixture(scope="module")
def hyp():
    return hypothesis


def weights_sum(mod) -> float:
    return float(mod.live_weight_sum())


# ===========================================================================
# tracker/ambiguity.py
# ===========================================================================

def test_crossing_frame_is_ambiguous_with_a_small_margin():
    tracks, zs = crossing_frame()
    rep = detect_ambiguity(tracks, zs, R_RADAR)

    assert isinstance(rep, AmbiguityReport)
    assert rep.ambiguous is True
    assert rep.margin < AMBIGUITY_DELTA_NATS
    assert rep.margin >= 0.0
    # Both interpretations assign both measurements; they differ in which.
    assert len(rep.best) == 2
    assert len(rep.second) == 2
    assert rep.best != rep.second


def test_separated_frame_is_unambiguous_with_a_large_margin():
    tracks, zs = separated_frame()
    rep = detect_ambiguity(tracks, zs, R_RADAR)

    assert rep.ambiguous is False
    # Forbidding the winning pair forces a miss + a birth, which is worth
    # several nats: the runner-up is not a live possibility.
    assert rep.margin > AMBIGUITY_DELTA_NATS
    assert rep.margin > 4.0
    assert rep.weights[0] > 0.95


def test_weights_are_normalized_and_ordered():
    for build in (crossing_frame, separated_frame):
        tracks, zs = build()
        rep = detect_ambiguity(tracks, zs, R_RADAR)
        w_best, w_second = rep.weights
        assert w_best + w_second == pytest.approx(1.0, abs=1e-12)
        assert 0.0 <= w_second <= w_best <= 1.0
        # The weights ARE the margin, re-expressed: w_second/w_best = e^-margin.
        if rep.weights[1] > 0.0:
            assert math.log(w_best / w_second) == pytest.approx(rep.margin, abs=1e-9)


def test_symmetric_frame_splits_belief_evenly():
    """Perfectly symmetric geometry -> a genuine coin flip, 0.5/0.5."""
    tracks = [make_track(0, 0.0, 0.0), make_track(1, 0.0, 0.0)]
    zs = np.array([[0.0, -50.0], [0.0, 50.0]])
    rep = detect_ambiguity(tracks, zs, R_RADAR)

    assert rep.ambiguous is True
    assert rep.margin == pytest.approx(0.0, abs=1e-9)
    assert rep.weights[0] == pytest.approx(0.5, abs=1e-9)
    assert rep.weights[1] == pytest.approx(0.5, abs=1e-9)


def test_margin_is_never_negative_and_second_is_never_cheaper():
    tracks, zs = crossing_frame()
    rep = detect_ambiguity(tracks, zs, R_RADAR)
    assert rep.cost_second >= rep.cost_best - 1e-12
    assert rep.margin == pytest.approx(rep.cost_second - rep.cost_best, abs=1e-9)


@pytest.mark.parametrize(
    "n_tracks, zs",
    [
        (0, np.zeros((0, 2))),                 # no tracks, no measurements
        (0, np.array([[0.0, 0.0]])),           # measurement with nothing to hit
        (2, np.zeros((0, 2))),                 # tracks with no measurements
    ],
)
def test_degenerate_frames_do_not_crash(n_tracks, zs):
    tracks = [make_track(i, 100.0 * i, 0.0) for i in range(n_tracks)]
    rep = detect_ambiguity(tracks, zs, R_RADAR)

    assert rep.ambiguous is False
    assert rep.margin == math.inf
    assert rep.cost_second == math.inf
    assert rep.second == []
    assert rep.weights == (1.0, 0.0)
    assert sum(rep.weights) == pytest.approx(1.0)


def test_all_measurements_gated_out_gives_no_second_solution():
    """Nothing is assignable -> nothing to forbid -> not ambiguous, infinite margin."""
    tracks = [make_track(0, 0.0, 0.0)]
    zs = np.array([[50000.0, 50000.0]])
    rep = detect_ambiguity(tracks, zs, R_RADAR)

    assert rep.best == []
    assert rep.ambiguous is False
    assert rep.margin == math.inf


def test_empty_forbidden_solve_agrees_with_associate_global():
    """The margin must not depend on two solvers agreeing -- but they do.

    ``detect_ambiguity`` runs both of its solves through
    ``solve_with_forbidden``; this pins that path to the shipped
    ``associate_global`` so a divergence in padding or tie-breaking shows up
    here rather than as a quietly wrong margin.
    """
    for build in (crossing_frame, separated_frame):
        tracks, zs = build()
        mine = solve_with_forbidden(tracks, zs, R_RADAR, ())
        theirs = associate_global(tracks, zs, R_RADAR)
        assert mine.assignments == theirs.assignments
        assert mine.unassigned_tracks == theirs.unassigned_tracks
        assert mine.unassigned_meas == theirs.unassigned_meas
        np.testing.assert_allclose(mine.cost, theirs.cost)


def test_forbidden_pair_never_appears_in_the_solution():
    tracks, zs = crossing_frame()
    base = solve_with_forbidden(tracks, zs, R_RADAR, ())
    assert base.assignments, "precondition: the frame assigns something"

    for pair in base.assignments:
        again = solve_with_forbidden(tracks, zs, R_RADAR, [pair])
        assert pair not in again.assignments
        # And no returned pair sits at the BIG sentinel.
        for t, m in again.assignments:
            assert again.cost[t, m] < big_cost(DEFAULT_PARAMS)


def test_forbidden_out_of_range_pairs_are_ignored():
    tracks, zs = crossing_frame()
    ok = solve_with_forbidden(tracks, zs, R_RADAR, ())
    junk = solve_with_forbidden(tracks, zs, R_RADAR, [(99, 99), (-1, 0)])
    assert junk.assignments == ok.assignments


def test_total_cost_charges_misses_and_births():
    """A solution that assigns fewer pairs is not automatically cheaper."""
    tracks, zs = separated_frame()
    res = solve_with_forbidden(tracks, zs, R_RADAR, ())
    pairs_only = sum(float(res.cost[t, m]) for t, m in res.assignments)
    assert res.assignments and not res.unassigned_tracks
    assert total_cost(res, res.cost, DEFAULT_PARAMS) == pytest.approx(pairs_only)

    # Forbid one pair: one track is missed and one measurement is born, so the
    # total must rise even though the pair sum falls.
    forced = solve_with_forbidden(tracks, zs, R_RADAR, [res.assignments[0]])
    forced_pairs_only = sum(float(res.cost[t, m]) for t, m in forced.assignments)
    assert forced_pairs_only < pairs_only
    assert total_cost(forced, res.cost, DEFAULT_PARAMS) > total_cost(
        res, res.cost, DEFAULT_PARAMS
    )


def test_detection_is_deterministic():
    tracks, zs = crossing_frame()
    a = detect_ambiguity(tracks, zs, R_RADAR)
    b = detect_ambiguity(tracks, zs, R_RADAR)
    assert (a.best, a.second, a.margin, a.weights) == (b.best, b.second, b.margin, b.weights)


def test_truth_ids_do_not_influence_the_verdict():
    tracks, zs = crossing_frame()
    plain = detect_ambiguity(tracks, zs, R_RADAR)
    for t, name in zip(tracks, ("244010101", "244020202")):
        t.truth_id = name
    scrambled = detect_ambiguity(tracks, zs, R_RADAR)
    assert plain.best == scrambled.best
    assert plain.margin == scrambled.margin


# ===========================================================================
# aegis/hypothesis.py
# ===========================================================================

def test_reset_starts_clean_and_normalized(hyp):
    hyp.reset(1)
    assert hyp.live_ids() == ["h_00"]
    assert hyp.get_log() == []
    assert weights_sum(hyp) == pytest.approx(1.0, abs=1e-9)

    hyp.reset(4)
    assert hyp.live_count() == 4
    assert hyp.get_log() == []          # reset clears the log, not just the roots
    assert weights_sum(hyp) == pytest.approx(1.0, abs=1e-9)


def test_visible_fork_then_collapse_back_to_one(hyp):
    """Acceptance: an ambiguous crossing forks, both hypotheses live for three
    frames, then the N-scan window closes and prunes back to one."""
    hyp.reset(1)

    hyp.run_frame(True, 0.61, 0.39)          # the crossing
    assert hyp.live_count() == 2
    assert hyp.live_ids() == ["h_00a", "h_00b"]

    hyp.run_frame(False)                     # no new evidence
    assert hyp.live_count() == 2

    hyp.run_frame(False)                     # depth 3 -> collapse
    assert hyp.live_count() == 1
    assert hyp.live_ids() == ["h_00a"]       # the heavier branch survives
    assert weights_sum(hyp) == pytest.approx(1.0, abs=1e-9)


def test_log_has_a_split_line_then_a_pruned_line_in_the_exact_format(hyp):
    hyp.reset(1)
    hyp.run_frame(True, 0.61, 0.39)
    hyp.run_frame(False)
    hyp.run_frame(False)

    log = hyp.get_log()
    assert log[0] == "[Hypothesis] split h_00 -> h_00a (0.61) / h_00b (0.39)"
    assert log[-1] == "[Hypothesis] pruned h_00b"
    assert [ln for ln in log if ln.startswith("[Hypothesis] split ")]
    assert [ln for ln in log if ln.startswith("[Hypothesis] pruned ")]

    # ASCII only: this has to print on a Windows cp1252 console.
    for line in log:
        line.encode("cp1252")
        assert "->" in line or line.startswith("[Hypothesis] pruned ")
        assert all(ord(ch) < 128 for ch in line)


def test_split_weights_are_formatted_to_two_decimals(hyp):
    """0.6 must print as (0.60), not (0.6) -- a 0.61/0.39 split cannot tell the
    difference, so assert on weights that can."""
    hyp.reset(1)
    hyp.run_frame(True, 0.6, 0.4)
    assert hyp.get_log()[0] == "[Hypothesis] split h_00 -> h_00a (0.60) / h_00b (0.40)"


def test_nscan_collapse_fires_at_depth_3(hyp):
    hyp.reset(1)
    for frame in range(1, 4):
        hyp.run_frame(frame == 1, 0.7, 0.3)   # fork on frame 1, then age
        if frame < 3:
            assert hyp.live_count() == 2, f"frame {frame} collapsed too early"
            assert hyp.max_live_depth() == frame

    assert hyp.max_live_depth() == hyp.NSCAN_DEPTH == 3
    assert hyp.live_count() == 1
    assert any(ln.startswith("[Hypothesis] pruned") for ln in hyp.get_log())


def test_collapse_keeps_the_highest_weight_leaf(hyp):
    hyp.reset(1)
    hyp.run_frame(True, 0.3, 0.7)     # the "b" child is the heavy one here
    hyp.run_frame(False)
    hyp.run_frame(False)
    assert hyp.live_ids() == ["h_00b"]
    assert "[Hypothesis] pruned h_00a" in hyp.get_log()


def test_weight_kill_removes_anything_below_one_percent(hyp):
    hyp.reset(1)
    hyp.run_frame(True, 0.995, 0.005)          # 0.005 < WEIGHT_FLOOR
    assert hyp.WEIGHT_FLOOR == 0.01
    assert hyp.live_ids() == ["h_00a"]
    assert "[Hypothesis] pruned h_00b" in hyp.get_log()
    # The killed leaf's weight is redistributed by the final normalize.
    assert weights_sum(hyp) == pytest.approx(1.0, abs=1e-9)


def test_weight_kill_never_empties_the_tree(hyp):
    """Even a pathological split leaves one survivor."""
    hyp.reset(1)
    hyp.run_frame(True, 0.5, 0.5)
    for _ in range(4):
        hyp.run_frame(True, 0.999, 0.001)
        assert hyp.live_count() >= 1
        assert weights_sum(hyp) == pytest.approx(1.0, abs=1e-9)


def test_hard_cap_holds_at_eight_across_a_full_replay(hyp):
    """Acceptance: drive enough forks to try to breach the cap, every frame."""
    assert hyp.MAX_LIVE == 8
    hyp.reset(6)                                   # 6 roots -> 12 leaves on a fork
    assert hyp.live_count() == 6

    seen_at_cap = False
    for frame in range(8):
        hyp.run_frame(True, 0.6, 0.4, True)        # fork EVERY live leaf
        n = hyp.live_count()
        assert n <= hyp.MAX_LIVE, f"cap breached at frame {frame}: {n} live"
        assert weights_sum(hyp) == pytest.approx(1.0, abs=1e-9)
        seen_at_cap = seen_at_cap or n == hyp.MAX_LIVE

    assert seen_at_cap, "the cap never actually bound; the test proves nothing"


def test_cap_kills_the_lowest_weight_first(hyp):
    hyp.reset(6)
    hyp.run_frame(True, 0.6, 0.4, True)            # 12 leaves -> cap to 8
    assert hyp.live_count() == 8

    pruned = [ln.split()[-1] for ln in hyp.get_log()
              if ln.startswith("[Hypothesis] pruned ")]
    assert len(pruned) == 4
    # Every "b" child carries 0.4 of its parent and every "a" child 0.6, so the
    # four that die must all be "b" children.
    assert all(pid.endswith("b") for pid in pruned), pruned
    assert all(pid not in hyp.live_ids() for pid in pruned)


def test_live_weights_sum_to_one_after_every_operation(hyp):
    """Frame boundaries only -- by construction the sum is NOT 1 between the
    steps of maintain(); that is what the final normalize is for."""
    script = [
        (True, 0.61, 0.39, False),
        (False, 0.5, 0.5, False),
        (True, 0.5, 0.5, True),
        (True, 0.9, 0.1, True),
        (False, 0.5, 0.5, False),
        (True, 0.55, 0.45, True),
        (False, 0.5, 0.5, False),
        (True, 0.99, 0.01, True),
    ]
    hyp.reset(3)
    assert weights_sum(hyp) == pytest.approx(1.0, abs=1e-9)

    for i, (amb, w1, w2, wide) in enumerate(script):
        hyp.run_frame(amb, w1, w2, wide)
        assert weights_sum(hyp) == pytest.approx(1.0, abs=1e-9), f"frame {i}"
        assert 1 <= hyp.live_count() <= hyp.MAX_LIVE
        for _hid, w, _d in hyp.live_state():
            assert w >= hyp.WEIGHT_FLOOR or hyp.live_count() == 1


def test_unambiguous_frame_ages_without_branching(hyp):
    hyp.reset(1)
    before = hyp.live_ids()
    hyp.run_frame(False)
    assert hyp.live_ids() == before          # same node, no fork
    assert hyp.max_live_depth() == 1
    assert hyp.get_log() == []               # no lifecycle event to report


def test_ambiguity_report_drives_the_fork_end_to_end(hyp):
    """The detector and hypothesis lifecycle, wired together the way the
    tracker wires them: numbers out of numpy, structure into the graph."""
    tracks, zs = crossing_frame()
    rep = detect_ambiguity(tracks, zs, R_RADAR)
    assert rep.ambiguous

    hyp.reset(1)
    hyp.run_frame(rep.ambiguous, rep.weights[0], rep.weights[1])
    assert hyp.live_count() == 2
    assert weights_sum(hyp) == pytest.approx(1.0, abs=1e-9)

    live = dict((hid, w) for hid, w, _d in hyp.live_state())
    assert live["h_00a"] == pytest.approx(rep.weights[0], abs=1e-9)
    assert live["h_00b"] == pytest.approx(rep.weights[1], abs=1e-9)

    # A well-separated frame must NOT fork.
    tracks, zs = separated_frame()
    calm = detect_ambiguity(tracks, zs, R_RADAR)
    hyp.reset(1)
    hyp.run_frame(calm.ambiguous, calm.weights[0], calm.weights[1])
    assert hyp.live_count() == 1

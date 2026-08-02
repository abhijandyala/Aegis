"""Acceptance + unit tests for tracker.assoc and tracker.metrics (Phase 2).

Four acceptance criteria are covered here:
  AC1 CROSSING TARGETS   -- test_ac1_deterministic_trap_greedy_mispairs_and_global_does_not
                            test_ac1_deterministic_trap_scores_id_switches
                            test_ac1_crossing_statistical_advantage
  AC2 ID-SWITCH COUNTER  -- test_ac2_* (metrics API tested directly, no simulation)
  AC3 PERFORMANCE        -- test_ac3_global_40x40_under_20ms
  AC4 DEGENERATE INPUTS  -- test_ac4_*

Everything else is supporting coverage of the association contract: the cost
formula, the gate, the identity holdout, the assignment partition invariant,
global-vs-greedy optimality, and the track lifecycle state machine.

`beta_fa` is tuned outside this file, so it is never hard-coded -- every test
reads it from DEFAULT_PARAMS or builds its own AssocParams.
"""

import json
import math
import time
from collections import deque

import numpy as np
import pytest
from scipy.linalg import solve as sp_solve

from tracker.kalman import CVKalman
from tracker.assoc import (
    CHI2_GATE_2DOF_99,
    AssocParams,
    AssocResult,
    DEFAULT_PARAMS,
    Track,
    associate_global,
    associate_greedy,
    big_cost,
    build_cost_matrix,
    max_assignable_sigma,
)
from tracker.metrics import TrackingMetrics, count_id_switches

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
SEED = 20260726

R_AIS = np.diag([25.0, 25.0])          # sigma =  5 m
R_RADAR = np.diag([2500.0, 2500.0])    # sigma = 50 m

ALGOS = [("global", associate_global), ("greedy", associate_greedy)]

# --- AC1 crossing scenario ------------------------------------------------
# Realistic maritime kinematics: 18 knots, 30-second revisit, 25 m position
# noise. (An earlier draft of this file used 150 m/s, which is ~290 knots --
# an aircraft, not a vessel. The quantity that actually sets the difficulty is
# the per-frame travel v*dt against sigma, so a slow ship on a long revisit is
# the same problem as a fast target on a short one, and it is the honest one.)
#
# The meeting is placed HALF a frame off the sampling grid (CROSS_FRAME = 11.5).
# At an exactly-sampled intersection both tracks predict to the same point and
# no association rule beats a coin flip, which would make the comparison pure
# noise rather than a measurement of the algorithms.
KNOT = 0.514444
CROSS_DT = 30.0                # s, realistic AIS/radar revisit
CROSS_N_FRAMES = 24            # 23 association frames, ~11 before / ~12 after
CROSS_FRAME = 11.5
CROSS_SPEED = 18.0 * KNOT      # m/s (9.26), identical for both targets
CROSS_SIGMA = 25.0             # m, per-axis measurement noise
CROSS_R = np.diag([CROSS_SIGMA ** 2, CROSS_SIGMA ** 2])
CROSS_SIGMA_V0 = 5.0           # m/s, initial velocity uncertainty
CROSS_Q = 0.01                 # m^2/s^3; vessels on steady courses barely manoeuvre

# Unscreened. range() on purpose -- see test_ac1_crossing_statistical_advantage
# for why the aggregate claim is a RATIO rather than "global never fails".
CROSSING_SEEDS = tuple(range(200))

# --- AC1 deterministic trap -----------------------------------------------
# Geometry, not a lucky noise draw. Two facts drive it:
#   * The log-det terms CANCEL in the joint comparison (each track appears
#     exactly once on both sides), so global picks the correct pairing iff
#         d00 + d11  <  d01 + d10.
#   * Greedy commits the lowest-log-det track first -- its whole row is
#     cheapest -- and that track takes its own nearest plot, so greedy mis-pairs
#     iff  d01 < d00.
# Both hold at once when a converged track sits slightly nearer its neighbour's
# plot while the neighbour, loose after a dark period, would have to be dragged
# much further to accept the swap. Realistically: a well-tracked ferry crossing
# a vessel that has just been re-acquired.
TRAP_SIGMA_TIGHT = 20.0        # m, converged ferry
TRAP_SIGMA_LOOSE = 200.0       # m, just re-acquired after going dark
TRAP_R = np.diag([25.0 ** 2, 25.0 ** 2])
TRAP_B_Y = 300.0               # m, how far north the loose track predicts


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _make_track(track_id, pos, vel, P, truth_id=None, params=DEFAULT_PARAMS, q=0.5):
    """Build a Track whose filter sits at a chosen state with a chosen covariance.

    Seeding through init_two_point would drag in a large two-point-differencing
    transient (velocity sigma = sqrt(2) * sigma_meas / dt), which dominates the
    early frames and would make the crossing scenario measure the initialiser
    rather than the association rule. test_kalman.py already writes ``_P``
    directly for the same reason.
    """
    kf = CVKalman(q=q)
    kf.state[:2] = np.asarray(pos, dtype=float).reshape(2)
    kf.state[2:] = np.asarray(vel, dtype=float).reshape(2)
    kf.cov[:] = np.asarray(P, dtype=float).reshape(4, 4)
    return Track(id=track_id, kf=kf, truth_id=truth_id, params=params)


def _simple_track(track_id, pos, pos_cov, truth_id=None, params=DEFAULT_PARAMS):
    """Track at ``pos``, zero velocity, position block ``pos_cov``, identity velocity block."""
    P = np.eye(4, dtype=float)
    P[:2, :2] = np.asarray(pos_cov, dtype=float).reshape(2, 2)
    return _make_track(track_id, pos, (0.0, 0.0), P, truth_id=truth_id, params=params)


def _assert_partition(res, n_tracks, n_meas):
    """Every index appears exactly once: assignments + unassigned form a partition."""
    assert isinstance(res, AssocResult)

    tracks_used = [t for t, _ in res.assignments]
    meas_used = [m for _, m in res.assignments]

    # Assignments are sorted by track index and reference no index twice.
    assert tracks_used == sorted(tracks_used), (
        f"assignments not sorted by track_idx: {res.assignments}"
    )
    assert len(set(tracks_used)) == len(tracks_used), (
        f"a track index is assigned twice: {res.assignments}"
    )
    assert len(set(meas_used)) == len(meas_used), (
        f"a measurement index is assigned twice: {res.assignments}"
    )

    # Exact partition of both index sets.
    assert sorted(tracks_used + list(res.unassigned_tracks)) == list(range(n_tracks)), (
        f"track indices are not a partition: assigned={sorted(tracks_used)}, "
        f"unassigned={list(res.unassigned_tracks)}, N={n_tracks}"
    )
    assert sorted(meas_used + list(res.unassigned_meas)) == list(range(n_meas)), (
        f"measurement indices are not a partition: assigned={sorted(meas_used)}, "
        f"unassigned={list(res.unassigned_meas)}, M={n_meas}"
    )

    # Matrices must always be present and correctly shaped, even when empty.
    for name in ("cost", "gate", "d2"):
        arr = np.asarray(getattr(res, name))
        assert arr.shape == (n_tracks, n_meas), (
            f"{name} has shape {arr.shape}, expected {(n_tracks, n_meas)}"
        )
    assert np.asarray(res.gate).dtype == np.bool_


def _crossing_truth():
    """Noiseless truth for both targets plus their constant velocities."""
    t = (np.arange(CROSS_N_FRAMES, dtype=float) - CROSS_FRAME) * CROSS_DT
    zeros = np.zeros(CROSS_N_FRAMES, dtype=float)
    truth_a = np.column_stack([CROSS_SPEED * t, zeros])          # west -> east, y = 0
    truth_b = np.column_stack([zeros, CROSS_SPEED * t])          # south -> north, x = 0
    vel_a = np.array([CROSS_SPEED, 0.0])
    vel_b = np.array([0.0, CROSS_SPEED])
    return truth_a, truth_b, vel_a, vel_b


def _crossing_measurements(seed):
    """One noisy measurement per target per frame. Generated ONCE per seed and
    handed to both algorithms, so neither sees a different noise realisation."""
    truth_a, truth_b, _, _ = _crossing_truth()
    rng = np.random.default_rng(seed)
    meas_a = truth_a + rng.normal(0.0, CROSS_SIGMA, size=truth_a.shape)
    meas_b = truth_b + rng.normal(0.0, CROSS_SIGMA, size=truth_b.shape)
    return meas_a, meas_b


def _run_crossing(associate_fn, meas_a, meas_b):
    """Drive the crossing scenario end to end with a pluggable association rule.

    Identical in every respect except the ``associate_fn`` call: same
    measurements (passed in, not regenerated), freshly constructed tracks and
    filters, same predict/update/lifecycle bookkeeping, same metric.

    Ground truth is the held-out label: ``truth_id`` is attached to the Track
    only so the scorer can name it, and the truth label of a *measurement* is
    its column index (column 0 is always target "A"). Nothing about identity is
    ever passed into ``associate_fn``.

    CLEAR-MOT note: the scored pair is (track.id, truth label of the measurement
    the track was matched to this frame). Pairing a track with its own static
    ``truth_id`` would make the metric identically zero by construction and
    could not detect a swap.

    Returns ``(switches, metrics, n_frames)``.
    """
    truth_a, truth_b, vel_a, vel_b = _crossing_truth()
    P0 = np.diag([CROSS_SIGMA ** 2, CROSS_SIGMA ** 2,
                  CROSS_SIGMA_V0 ** 2, CROSS_SIGMA_V0 ** 2])

    tracks = [
        _make_track(1, truth_a[0], vel_a, P0, truth_id="A", q=CROSS_Q),
        _make_track(2, truth_b[0], vel_b, P0, truth_id="B", q=CROSS_Q),
    ]
    meas_truth_label = ("A", "B")

    frames = []
    metrics = TrackingMetrics()

    for k in range(1, CROSS_N_FRAMES):
        for tr in tracks:
            tr.kf.predict(CROSS_DT)

        zs = np.array([meas_a[k], meas_b[k]], dtype=float)
        res = associate_fn(tracks, zs, CROSS_R)
        _assert_partition(res, len(tracks), zs.shape[0])

        pairs = []
        for ti, mi in res.assignments:
            assert bool(res.gate[ti, mi]), "a gated-out pair was assigned"
            tr = tracks[ti]
            _, S = tr.kf.innovation(zs[mi], CROSS_R)
            logdet_S = float(np.log(np.linalg.det(np.asarray(S, dtype=float))))
            tr.kf.update(zs[mi], CROSS_R)
            tr.register_hit(float(res.d2[ti, mi]), logdet_S)
            pairs.append((tr.id, meas_truth_label[mi]))

        for ti in res.unassigned_tracks:
            tracks[ti].register_miss()

        frames.append(pairs)
        metrics.update(pairs)

    switches = count_id_switches(frames)
    assert switches == metrics.id_switches, (
        "count_id_switches and TrackingMetrics disagree on the same frame stream: "
        f"{switches} vs {metrics.id_switches}"
    )
    return switches, metrics, len(frames)


def _random_fixture(seed, n_tracks, n_meas, spread=400.0, pos_var=2500.0):
    """Random tracks/measurements in a box, sized so a healthy fraction gates in."""
    rng = np.random.default_rng(seed)
    tracks = [
        _simple_track(i, rng.uniform(-spread, spread, size=2),
                      np.diag([pos_var, pos_var]), truth_id=f"truth-{i}")
        for i in range(n_tracks)
    ]
    zs = rng.uniform(-spread, spread, size=(n_meas, 2))
    return tracks, zs


def _expected_cost(track, z, R, params):
    """Cost from the written formula, computed independently of build_cost_matrix."""
    nu, S = track.kf.innovation(z, R)
    S = np.asarray(S, dtype=float)
    d2 = float(np.asarray(nu, dtype=float) @ sp_solve(S, np.asarray(nu, dtype=float),
                                                      assume_a="pos"))
    sign, logdet = np.linalg.slogdet(S)
    assert sign > 0
    cost = (0.5 * d2 + math.log(2.0 * math.pi) + 0.5 * float(logdet)
            - math.log(params.p_detect))
    return d2, cost


# --------------------------------------------------------------------------
# ACCEPTANCE CRITERION 1 -- TWO TRACKS CROSSING AT 90 DEGREES
# --------------------------------------------------------------------------
def _trap_frame(tight_pos, loose_pos):
    """Build the deterministic trap frame: two tracks, two plots, no noise.

    Track 0 is a converged ferry heading east; track 1 was dark and has just been
    re-acquired, heading north. Measurement 0 belongs to track 0, measurement 1 to
    track 1. See the TRAP_* constants for the algebra this is built to satisfy.
    """
    P_tight = np.eye(4)
    P_tight[:2, :2] = np.diag([TRAP_SIGMA_TIGHT ** 2, TRAP_SIGMA_TIGHT ** 2])
    P_loose = np.eye(4)
    P_loose[:2, :2] = np.diag([TRAP_SIGMA_LOOSE ** 2, TRAP_SIGMA_LOOSE ** 2])

    tracks = [
        _make_track(0, tight_pos, (CROSS_SPEED, 0.0), P_tight, truth_id="A", q=CROSS_Q),
        _make_track(1, loose_pos, (0.0, CROSS_SPEED), P_loose, truth_id="B", q=CROSS_Q),
    ]
    zs = np.array([[0.0, -35.0], [0.0, 30.0]])   # plot 0 -> track 0, plot 1 -> track 1
    return tracks, zs


def test_ac1_deterministic_trap_greedy_mispairs_and_global_does_not(capsys):
    """AC1: on a fixed, noiseless 90-degree crossing geometry, greedy mis-pairs and
    global does not.

    This is the headline claim and it is deterministic -- no seed, no noise draw,
    no screening. Greedy commits the converged track to its nearest plot, which is
    its neighbour's; global compares the two whole-frame interpretations and keeps
    both identities.
    """
    tracks_g, zs = _trap_frame((0.0, 0.0), (0.0, TRAP_B_Y))
    tracks_r, _ = _trap_frame((0.0, 0.0), (0.0, TRAP_B_Y))

    cost, gate, d2 = build_cost_matrix(tracks_g, zs, TRAP_R)
    res_g = associate_global(tracks_g, zs, TRAP_R)
    res_r = associate_greedy(tracks_r, zs, TRAP_R)

    with capsys.disabled():
        print(f"\n[AC1 TRAP] d2 = [[{d2[0,0]:.2f}, {d2[0,1]:.2f}], "
              f"[{d2[1,0]:.2f}, {d2[1,1]:.2f}]]  "
              f"correct {d2[0,0]+d2[1,1]:.3f} < swapped {d2[0,1]+d2[1,0]:.3f}")
        print(f"[AC1 TRAP] global = {res_g.assignments}, greedy = {res_r.assignments}")

    # It is a genuine contest: every pairing, including both wrong ones, is in-gate.
    assert gate.all(), (
        "the trap is a gating walkover, not a contest -- some pairing gated out"
    )
    # The correct pairing really is the maximum-likelihood one, so global is not
    # merely being credited for a labelling convention.
    assert d2[0, 0] + d2[1, 1] < d2[0, 1] + d2[1, 0]
    # ... and greedy's first pick really is the wrong cell.
    assert d2[0, 1] < d2[0, 0], "greedy's trap condition (d01 < d00) does not hold"

    assert res_g.assignments == [(0, 0), (1, 1)], (
        f"global mis-paired on the deterministic trap: {res_g.assignments}"
    )
    assert res_r.assignments == [(0, 1), (1, 0)], (
        f"greedy was expected to take the cross pairing, got {res_r.assignments}"
    )


def test_ac1_deterministic_trap_scores_id_switches(capsys):
    """AC1: greedy >= 1 ID switch, global 0, counted by the metric on the trap.

    Frame 1 is unambiguous so both algorithms start correct; frame 2 is the trap.
    Scored through count_id_switches, so this is the acceptance number itself and
    not a proxy for it.
    """
    results = {}
    for name, assoc_fn in ALGOS:
        frames = []
        # Frame 1: plots far apart, both algorithms trivially correct.
        tracks, _ = _trap_frame((0.0, 0.0), (0.0, TRAP_B_Y))
        easy = np.array([[0.0, 0.0], [0.0, TRAP_B_Y]])
        res = assoc_fn(tracks, easy, TRAP_R)
        frames.append([(tracks[ti].id, ("A", "B")[mi]) for ti, mi in res.assignments])

        # Frame 2: the trap, on freshly built tracks so the geometry is exact.
        tracks, zs = _trap_frame((0.0, 0.0), (0.0, TRAP_B_Y))
        res = assoc_fn(tracks, zs, TRAP_R)
        frames.append([(tracks[ti].id, ("A", "B")[mi]) for ti, mi in res.assignments])

        results[name] = count_id_switches(frames)

    with capsys.disabled():
        print(f"\n[AC1 TRAP] ID switches -- global = {results['global']}, "
              f"greedy = {results['greedy']}")

    assert results["global"] == 0, (
        f"global lost identity on the deterministic trap: {results['global']} switch(es)"
    )
    assert results["greedy"] >= 1, (
        f"greedy held identity on the deterministic trap: {results['greedy']} switch(es)"
    )


def test_uncertain_track_stops_being_assignable_above_the_covariance_ceiling(capsys):
    """Phase 4 warning, pinned as a test: beta_fa caps how uncertain a track may get.

    Above max_assignable_sigma, cost_assign exceeds cost_miss + cost_birth even at
    d2 = 0, so a measurement sitting exactly on the prediction is still declared a
    birth. This is the constraint that governs dark-vessel re-association, and it
    is set by beta_fa rather than by the chi-squared gate.
    """
    ceiling = max_assignable_sigma(DEFAULT_PARAMS)

    def assigned_at(sigma_pos):
        # Measurement placed exactly on the prediction: the most favourable case
        # possible, so a failure here is the cost model talking, not the geometry.
        track = _simple_track(0, (0.0, 0.0), np.diag([sigma_pos ** 2, sigma_pos ** 2]))
        zs = np.array([[0.0, 0.0]])
        res = associate_global([track], zs, np.diag([1.0, 1.0]))
        gate = np.asarray(res.gate)
        return len(res.assignments) == 1, bool(gate[0, 0])

    below_assigned, below_gated = assigned_at(0.5 * ceiling)
    above_assigned, above_gated = assigned_at(2.0 * ceiling)

    with capsys.disabled():
        print(f"\n[CEILING] max_assignable_sigma = {ceiling:.0f} m "
              f"(beta_fa = {DEFAULT_PARAMS.beta_fa:g})")
        print(f"    sigma = {0.5*ceiling:>7.0f} m -> assigned={below_assigned}, "
              f"in-gate={below_gated}")
        print(f"    sigma = {2.0*ceiling:>7.0f} m -> assigned={above_assigned}, "
              f"in-gate={above_gated}")

    assert below_assigned, "a track below the ceiling must still be assignable"
    assert not above_assigned, (
        "a track above the ceiling was assigned; max_assignable_sigma is wrong"
    )
    # The point of the warning: the gate is NOT what rejects it. The pair is
    # comfortably inside the gate and still loses to miss+birth.
    assert above_gated, (
        "expected the over-ceiling pair to be in-gate but out-competed by "
        "miss+birth; if it gated out, this test is proving the wrong thing"
    )


def test_ac1_crossing_statistical_advantage(capsys):
    """AC1: over UNSCREENED seeds, global switches identity materially less than greedy.

    The honest form of the claim. In a symmetric two-target crossing the two failure
    probabilities are tied together by the geometry -- global mis-pairs when noise
    flips the sign of the separation projected onto (pA - pB), greedy whenever a
    cross term becomes the single cheapest cell -- so there is no configuration where
    global is clean on every seed while greedy visibly fails. Quoting a "global: 0"
    from a screened seed set would overstate what the algorithm does.

    What is true, and what this asserts, is the ratio: on the same noise draws,
    driven through the same loop, global loses identity substantially less often.
    An earlier draft of this file pre-screened 10 seeds out of 1500; range(200) here
    is deliberate.

    Checked for window-dependence over four disjoint 200-seed windows (0-200,
    200-400, 400-600, 600-800). Global 2/7/9/4 switches against greedy 30/30/34/22,
    so the per-window ratio moves between 3.8x and 15x and the assertion below holds
    with room in every window. Pooled over all 800 seeds: global 22, greedy 116,
    i.e. ~5x -- that pooled figure is the one to quote, not any single window's.
    """
    global_total = greedy_total = 0
    global_bad = greedy_bad = 0

    for seed in CROSSING_SEEDS:
        meas_a, meas_b = _crossing_measurements(seed)
        g_sw, _, _ = _run_crossing(associate_global, meas_a, meas_b)
        r_sw, _, _ = _run_crossing(associate_greedy, meas_a, meas_b)
        global_total += g_sw
        greedy_total += r_sw
        global_bad += g_sw > 0
        greedy_bad += r_sw > 0

    n = len(CROSSING_SEEDS)
    with capsys.disabled():
        print(f"\n[AC1 CROSSING] {n} unscreened seeds, {CROSS_N_FRAMES - 1} frames each, "
              f"{CROSS_SPEED / KNOT:.0f} kn, {CROSS_DT:.0f} s revisit, "
              f"sigma = {CROSS_SIGMA:.0f} m")
        print(f"    global: {global_bad}/{n} runs with a switch, {global_total} switches total")
        print(f"    greedy: {greedy_bad}/{n} runs with a switch, {greedy_total} switches total")
        if global_total:
            print(f"    greedy loses identity {greedy_total / global_total:.2f}x as often")

    assert greedy_total > global_total, (
        f"greedy ({greedy_total}) did not lose identity more than global "
        f"({global_total}) over {n} unscreened seeds"
    )
    assert global_total <= 0.6 * greedy_total, (
        f"global's advantage is too thin to claim: {global_total} switches vs "
        f"greedy's {greedy_total} over {n} seeds"
    )


def test_ac1_crossing_is_a_genuine_contest_not_a_gating_walkover(capsys):
    """AC1 support: near the crossing the WRONG pairing must be inside the gate.

    If every cross pairing gated out, global would be right for free, greedy
    could never make the mistake, and the acceptance criterion would be vacuous.
    Counted over the whole seed set, because whether a given frame is confusing
    depends on the noise draw.
    """
    P0 = np.diag([CROSS_SIGMA ** 2, CROSS_SIGMA ** 2,
                  CROSS_SIGMA_V0 ** 2, CROSS_SIGMA_V0 ** 2])

    cross_gated_frames = 0      # a track sees the OTHER target's plot in its gate
    fully_gated_frames = 0      # all four pairings live: a true 2x2 contest

    for seed in CROSSING_SEEDS:
        meas_a, meas_b = _crossing_measurements(seed)
        truth_a, truth_b, vel_a, vel_b = _crossing_truth()
        tracks = [
            _make_track(1, truth_a[0], vel_a, P0, truth_id="A", q=CROSS_Q),
            _make_track(2, truth_b[0], vel_b, P0, truth_id="B", q=CROSS_Q),
        ]
        for k in range(1, CROSS_N_FRAMES):
            for tr in tracks:
                tr.kf.predict(CROSS_DT)
            zs = np.array([meas_a[k], meas_b[k]], dtype=float)
            _, gate, _ = build_cost_matrix(tracks, zs, CROSS_R)
            gate = np.asarray(gate)
            if bool(gate[0, 1]) or bool(gate[1, 0]):
                cross_gated_frames += 1
            if bool(gate.all()):
                fully_gated_frames += 1
            res = associate_global(tracks, zs, CROSS_R)
            for ti, mi in res.assignments:
                tracks[ti].kf.update(zs[mi], CROSS_R)

    with capsys.disabled():
        print(f"\n[AC1 CROSSING] over {len(CROSSING_SEEDS)} seeds: "
              f"{cross_gated_frames} frame(s) where the wrong pairing is in-gate, "
              f"{fully_gated_frames} where all four pairings are")

    assert cross_gated_frames >= 1, (
        "no frame ever put the other target's plot inside a track's gate, so the "
        "crossing never actually confuses either algorithm"
    )
    assert fully_gated_frames >= 1, (
        "no frame presented a full 2x2 contest (all four pairings in-gate)"
    )


# --------------------------------------------------------------------------
# ACCEPTANCE CRITERION 2 -- ID-SWITCH COUNTER
# --------------------------------------------------------------------------
# (frames, expected switches, description)
_ID_SWITCH_CASES = [
    ([], 0, "empty stream"),
    ([[]], 0, "one empty frame"),
    ([[(1, "A")]], 0, "first sighting is never a switch"),
    ([[(1, "A")], [(1, "A")]], 0, "A->1, A->1"),
    ([[(1, "A")], [(2, "A")]], 1, "A->1, A->2"),
    ([[(1, "A")], [], [(2, "A")]], 1, "A->1, gap, A->2 (switches count across gaps)"),
    ([[(1, "A")], [(2, "A")], [(1, "A")]], 2, "A->1, A->2, A->1"),
    ([[(1, "A"), (2, "B")], [(1, "A"), (2, "B")]], 0, "two objects, both stable"),
    ([[(1, "A"), (2, "B")], [(2, "A"), (1, "B")]], 2, "two objects, swapped"),
    ([[(1, "A")], [(1, "B")]], 0, "different truths, same track id, no switch"),
    ([[(1, "A")], [(2, "B")], [(2, "A")]], 1, "interleaved objects, one switch on A"),
]


@pytest.mark.parametrize(
    "frames,expected,label",
    _ID_SWITCH_CASES,
    ids=[c[2] for c in _ID_SWITCH_CASES],
)
def test_ac2_count_id_switches_matches_clear_mot(frames, expected, label):
    """AC2: count_id_switches implements the CLEAR-MOT identity-switch rule."""
    assert count_id_switches(frames) == expected, f"case: {label}"


@pytest.mark.parametrize(
    "frames,expected,label",
    _ID_SWITCH_CASES,
    ids=[c[2] for c in _ID_SWITCH_CASES],
)
def test_ac2_tracking_metrics_agrees_with_count_id_switches(frames, expected, label):
    """AC2: the stateful accumulator must agree with the one-shot function everywhere."""
    tm = TrackingMetrics()
    for frame in frames:
        tm.update(frame)

    assert tm.id_switches == expected, f"case: {label}"
    assert tm.id_switches == count_id_switches(frames)
    assert tm.frames == len(frames)
    assert tm.total_matches == sum(len(f) for f in frames)


def test_ac2_tracking_metrics_as_dict_is_json_serialisable():
    """AC2: as_dict() must survive a json.dumps/loads round-trip."""
    tm = TrackingMetrics()
    tm.update([(1, "A"), (2, "B")])
    tm.update([(2, "A"), (1, "B")])
    tm.update([])

    d = tm.as_dict()
    assert isinstance(d, dict)

    text = json.dumps(d)          # raises TypeError on sets / numpy scalars
    round_tripped = json.loads(text)
    assert json.dumps(round_tripped, sort_keys=True) == json.dumps(d, sort_keys=True), (
        "as_dict() does not survive a JSON round-trip unchanged"
    )
    assert round_tripped["id_switches"] == 2
    assert round_tripped["frames"] == 3
    assert round_tripped["total_matches"] == 4


def test_ac2_tracking_metrics_tracks_and_truths_seen():
    """AC2 support: the accumulator reports how many distinct ids it observed."""
    tm = TrackingMetrics()
    tm.update([(1, "A"), (2, "B")])
    tm.update([(3, "A")])

    # Both are counts of distinct ids, not the id collections themselves.
    assert tm.truths_seen == 2      # "A", "B"
    assert tm.tracks_seen == 3      # 1, 2, 3


def test_ac2_tracking_metrics_reset_returns_to_a_clean_slate():
    """AC2 support: reset() must clear the last-seen map too, not just the counters."""
    tm = TrackingMetrics()
    tm.update([(1, "A")])
    tm.update([(2, "A")])
    assert tm.id_switches == 1

    tm.reset()
    assert tm.id_switches == 0
    assert tm.frames == 0
    assert tm.total_matches == 0

    # If the last-seen map survived reset(), this would count a bogus switch.
    tm.update([(3, "A")])
    assert tm.id_switches == 0, "reset() left the previous track_id for truth 'A' behind"


# --------------------------------------------------------------------------
# ACCEPTANCE CRITERION 3 -- PERFORMANCE
# --------------------------------------------------------------------------
def test_ac3_global_40x40_under_20ms(capsys):
    """AC3: one 40x40 associate_global solve must complete in under 20 ms.

    time.perf_counter() is mandatory -- time.time() has ~15 ms resolution on
    Windows, which is the same order as the entire budget. Warm-up first, then
    best of three.

    The fixture is 10 tight clusters of 4 tracks and 4 measurements. A solve
    where everything gates out is trivially fast and proves nothing, so the
    gate population is asserted first.
    """
    n = 40
    clusters = 10
    per_cluster = n // clusters
    budget_ms = 20.0

    rng = np.random.default_rng(SEED + 3)
    pos_var = 2500.0
    R = np.diag([2500.0, 2500.0])      # S = diag(5000, 5000) -> gate radius ~214 m

    tracks = []
    zs = np.empty((n, 2), dtype=float)
    for c in range(clusters):
        centre = np.array([5000.0 * c, -3000.0 * c])
        for j in range(per_cluster):
            idx = c * per_cluster + j
            tracks.append(
                _simple_track(idx, centre + rng.normal(0.0, 25.0, size=2),
                              np.diag([pos_var, pos_var]), truth_id=f"truth-{idx}")
            )
            zs[idx] = centre + rng.normal(0.0, 25.0, size=2)

    cost, gate, d2 = build_cost_matrix(tracks, zs, R)
    n_gated = int(np.asarray(gate).sum())
    assert n_gated > 0, "every pair gated out -- the timed solve would be trivial"
    assert n_gated >= 2 * n, (
        f"only {n_gated} of {n * n} pairs are in-gate; the fixture is too sparse "
        f"to exercise the assignment solver"
    )

    # Warm-up (untimed): import-time work, allocator, and any lazy scipy setup.
    for _ in range(3):
        associate_global(tracks, zs, R)

    timings_ms = []
    for _ in range(3):
        t0 = time.perf_counter()
        res = associate_global(tracks, zs, R)
        t1 = time.perf_counter()
        timings_ms.append((t1 - t0) * 1000.0)

    best_ms = min(timings_ms)
    _assert_partition(res, n, n)

    with capsys.disabled():
        print(f"\n[AC3 PERFORMANCE] 40x40 associate_global: best {best_ms:.3f} ms of "
              f"[{', '.join(f'{t:.3f}' for t in timings_ms)}] ms "
              f"(budget {budget_ms:.0f} ms, {n_gated}/{n * n} pairs in-gate, "
              f"{len(res.assignments)} assigned)")

    assert best_ms < budget_ms, (
        f"40x40 associate_global took {best_ms:.3f} ms (best of "
        f"{', '.join(f'{t:.3f}' for t in timings_ms)} ms); budget is "
        f"{budget_ms:.0f} ms"
    )


# --------------------------------------------------------------------------
# ACCEPTANCE CRITERION 4 -- DEGENERATE INPUTS
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name,fn", ALGOS)
def test_ac4_zero_measurements(name, fn):
    """AC4: N > 0, M = 0 -- every track must be reported as unassigned, no exception."""
    tracks = [
        _simple_track(i, (100.0 * i, 0.0), R_RADAR, truth_id=f"truth-{i}")
        for i in range(3)
    ]
    zs = np.zeros((0, 2), dtype=float)

    res = fn(tracks, zs, R_RADAR)
    _assert_partition(res, 3, 0)

    assert res.assignments == []
    assert sorted(res.unassigned_tracks) == [0, 1, 2]
    assert sorted(res.unassigned_meas) == []


@pytest.mark.parametrize("name,fn", ALGOS)
def test_ac4_zero_tracks(name, fn):
    """AC4: N = 0, M > 0 -- every measurement must be reported as a birth candidate."""
    zs = np.array([[0.0, 0.0], [10.0, 10.0], [-5.0, 3.0], [7.0, -2.0]])

    res = fn([], zs, R_RADAR)
    _assert_partition(res, 0, 4)

    assert res.assignments == []
    assert sorted(res.unassigned_tracks) == []
    assert sorted(res.unassigned_meas) == [0, 1, 2, 3]


@pytest.mark.parametrize("name,fn", ALGOS)
def test_ac4_zero_tracks_and_zero_measurements(name, fn):
    """AC4: N = 0, M = 0 -- a legal no-op that must still return a shaped result."""
    zs = np.zeros((0, 2), dtype=float)

    res = fn([], zs, R_RADAR)
    _assert_partition(res, 0, 0)

    assert res.assignments == []
    assert sorted(res.unassigned_tracks) == []
    assert sorted(res.unassigned_meas) == []


@pytest.mark.parametrize("name,fn", ALGOS)
def test_ac4_all_pairs_gated_out(name, fn):
    """AC4: measurements kilometres away must produce miss + birth, never an assignment.

    This is the test BIG exists for. Because the sentinel is finite, an
    assignment solver run on the raw cost matrix will happily return a full
    permutation; the implementation has to drop those pairs itself.
    """
    tracks = [
        _simple_track(i, (0.0, 100.0 * i), R_RADAR, truth_id=f"truth-{i}")
        for i in range(3)
    ]
    zs = np.array([[50_000.0, 0.0], [0.0, -80_000.0]])     # tens of km away

    res = fn(tracks, zs, R_RADAR)
    _assert_partition(res, 3, 2)

    assert not np.asarray(res.gate).any(), "a 50 km measurement should never gate in"
    assert res.assignments == [], (
        f"{name} assigned a gated-out pair: {res.assignments}"
    )
    assert sorted(res.unassigned_tracks) == [0, 1, 2]
    assert sorted(res.unassigned_meas) == [0, 1]

    BIG = big_cost(DEFAULT_PARAMS)
    np.testing.assert_allclose(np.asarray(res.cost), np.full((3, 2), BIG),
                               rtol=0, atol=1e-12)
    assert np.all(np.isfinite(np.asarray(res.cost))), "BIG must be a finite sentinel"


# --------------------------------------------------------------------------
# IDENTITY HOLDOUT
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name,fn", ALGOS)
def test_identity_holdout_scrambled_truth_ids_change_nothing(name, fn):
    """Supporting: truth_id must never influence association -- structural proof.

    Same geometry, same noise, only the held-out labels differ. If any of them
    leaked into the cost, the gate, or the assignment, this would diverge.
    """
    tracks_a, zs = _random_fixture(SEED + 11, n_tracks=8, n_meas=9)
    res_a = fn(tracks_a, zs, R_RADAR)

    # Identical fixture, every truth_id scrambled to a different string.
    tracks_b, zs_b = _random_fixture(SEED + 11, n_tracks=8, n_meas=9)
    scrambled = ["zulu", "yankee", "xray", "whiskey", "victor", "uniform", "tango", "sierra"]
    for tr, label in zip(tracks_b, scrambled):
        assert tr.truth_id != label
        tr.truth_id = label
    np.testing.assert_array_equal(zs, zs_b)
    res_b = fn(tracks_b, zs_b, R_RADAR)

    assert [tr.truth_id for tr in tracks_a] != [tr.truth_id for tr in tracks_b], (
        "the fixtures were not actually relabelled -- the test would be vacuous"
    )
    assert res_a.assignments == res_b.assignments, (
        f"{name} produced different assignments after relabelling identities: "
        f"{res_a.assignments} vs {res_b.assignments}"
    )
    assert sorted(res_a.unassigned_tracks) == sorted(res_b.unassigned_tracks)
    assert sorted(res_a.unassigned_meas) == sorted(res_b.unassigned_meas)
    np.testing.assert_array_equal(np.asarray(res_a.gate), np.asarray(res_b.gate))
    np.testing.assert_allclose(np.asarray(res_a.cost), np.asarray(res_b.cost),
                               rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(res_a.d2), np.asarray(res_b.d2),
                               rtol=0, atol=0)


# --------------------------------------------------------------------------
# PARTITION INVARIANT AND GATE DISCIPLINE
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name,fn", ALGOS)
@pytest.mark.parametrize("n_tracks,n_meas", [(1, 1), (3, 1), (1, 4), (5, 5), (7, 4), (4, 7)])
def test_assignments_form_an_exact_partition(name, fn, n_tracks, n_meas):
    """Supporting: assignments + unassigned_* must partition both index sets exactly."""
    tracks, zs = _random_fixture(SEED + 21 + n_tracks * 10 + n_meas, n_tracks, n_meas)
    res = fn(tracks, zs, R_RADAR)
    _assert_partition(res, n_tracks, n_meas)


@pytest.mark.parametrize("name,fn", ALGOS)
@pytest.mark.parametrize("seed_offset", range(8))
def test_no_gated_pair_is_ever_assigned(name, fn, seed_offset):
    """Supporting: gate[n, m] must be True for every assigned (n, m), both algorithms."""
    tracks, zs = _random_fixture(SEED + 100 + seed_offset, n_tracks=6, n_meas=6)
    res = fn(tracks, zs, R_RADAR)
    _assert_partition(res, 6, 6)

    assert len(res.assignments) > 0, (
        "the fixture produced no assignments at all, so the gate check below "
        "would pass vacuously"
    )
    gate = np.asarray(res.gate)
    for ti, mi in res.assignments:
        assert bool(gate[ti, mi]), (
            f"{name} assigned track {ti} to measurement {mi} but gate is False "
            f"(d2 = {res.d2[ti, mi]:.4f} > {DEFAULT_PARAMS.gate_chi2})"
        )


# --------------------------------------------------------------------------
# COST MATRIX
# --------------------------------------------------------------------------
def test_chi2_gate_constant_is_the_two_dof_99th_percentile():
    """Supporting: the exported gate constant is pinned by the contract."""
    assert CHI2_GATE_2DOF_99 == pytest.approx(9.21034, abs=1e-9)
    assert DEFAULT_PARAMS.gate_chi2 == pytest.approx(9.21034, abs=1e-9)


@pytest.mark.parametrize("R", [R_AIS, R_RADAR])
def test_cost_matrix_matches_the_written_formula(R):
    """Supporting: cost = 0.5*d2 + log(2pi) + 0.5*logdet(S) - log(p_detect).

    S and d2 are recomputed independently -- S from CVKalman.innovation, d2 via
    scipy.linalg.solve -- so this does not just re-run the implementation.
    """
    tracks = [
        _simple_track(0, (0.0, 0.0), np.diag([900.0, 400.0]), truth_id="a"),
        _simple_track(1, (30.0, -20.0), np.diag([1600.0, 2500.0]), truth_id="b"),
    ]
    zs = np.array([[5.0, 4.0], [28.0, -25.0], [12.0, 9.0]])

    cost, gate, d2 = build_cost_matrix(tracks, zs, R)
    assert cost.shape == (2, 3) and gate.shape == (2, 3) and d2.shape == (2, 3)

    for n, tr in enumerate(tracks):
        for m in range(zs.shape[0]):
            exp_d2, exp_cost = _expected_cost(tr, zs[m], R, DEFAULT_PARAMS)
            assert d2[n, m] == pytest.approx(exp_d2, rel=1e-10, abs=1e-10), (
                f"d2[{n},{m}] mismatch"
            )
            # None of these fixtures sits on the boundary, so < and <= agree and
            # the test does not depend on which comparison the gate uses.
            assert abs(exp_d2 - DEFAULT_PARAMS.gate_chi2) > 1e-6
            assert bool(gate[n, m]) == (exp_d2 <= DEFAULT_PARAMS.gate_chi2), (
                f"gate[{n},{m}] disagrees with d2 = {exp_d2}"
            )
            if gate[n, m]:
                assert cost[n, m] == pytest.approx(exp_cost, rel=1e-10, abs=1e-10), (
                    f"cost[{n},{m}] does not match the formula"
                )
            else:
                assert cost[n, m] == big_cost(DEFAULT_PARAMS)


def test_cost_is_exactly_big_cost_where_the_gate_is_false():
    """Supporting: gated-out cells carry the finite sentinel, bit for bit."""
    tracks = [_simple_track(0, (0.0, 0.0), R_RADAR, truth_id="a")]
    zs = np.array([[0.0, 0.0], [10_000.0, 0.0]])     # one in gate, one far out

    cost, gate, _ = build_cost_matrix(tracks, zs, R_RADAR)
    assert bool(gate[0, 0]) and not bool(gate[0, 1])
    assert cost[0, 1] == big_cost(DEFAULT_PARAMS)
    assert cost[0, 0] != big_cost(DEFAULT_PARAMS)


def test_big_cost_matches_its_definition_and_is_dominated_by_miss_plus_birth():
    """Supporting: BIG = cost_miss + cost_birth + 1, finite, and never worth taking."""
    for params in (DEFAULT_PARAMS,
                   AssocParams(p_detect=0.75, beta_fa=DEFAULT_PARAMS.beta_fa),
                   AssocParams(p_detect=0.95, beta_fa=DEFAULT_PARAMS.beta_fa * 10.0)):
        cost_miss = -math.log(1.0 - params.p_detect)
        cost_birth = -math.log(params.beta_fa)
        expected = cost_miss + cost_birth + 1.0

        BIG = big_cost(params)
        assert math.isfinite(BIG)
        assert BIG == pytest.approx(expected, rel=1e-12, abs=1e-12)
        # Strictly worse than declaring a miss and a birth, by exactly 1 nat.
        assert BIG > cost_miss + cost_birth
        assert BIG - (cost_miss + cost_birth) == pytest.approx(1.0, abs=1e-12)


def test_beta_fa_is_read_from_params_not_assumed():
    """Supporting: changing beta_fa must move BIG, proving nothing hard-codes it."""
    base = DEFAULT_PARAMS.beta_fa
    assert base > 0.0
    louder = AssocParams(beta_fa=base * 100.0)
    assert big_cost(louder) < big_cost(DEFAULT_PARAMS)
    assert big_cost(louder) == pytest.approx(
        big_cost(DEFAULT_PARAMS) - math.log(100.0), rel=1e-12, abs=1e-12
    )


@pytest.mark.parametrize("factor,expect_in_gate", [(0.999, True), (1.001, False)])
def test_gate_flips_at_the_chi2_boundary(factor, expect_in_gate):
    """Supporting: d2 just inside / just outside gate_chi2 must flip the gate.

    S is made exactly isotropic (P_pos = 0, R = s*I) so d2 = |nu|^2 / s and the
    geometry inverts cleanly. The boundary is probed at 0.999x and 1.001x, never
    exactly at it -- the contract does not say whether the comparison is < or <=.
    """
    s = 400.0
    tr = _simple_track(0, (0.0, 0.0), np.zeros((2, 2)), truth_id="a")
    R = np.diag([s, s])

    target_d2 = factor * DEFAULT_PARAMS.gate_chi2
    offset = math.sqrt(target_d2 * s)
    zs = np.array([[offset, 0.0]])

    cost, gate, d2 = build_cost_matrix([tr], zs, R)

    assert d2[0, 0] == pytest.approx(target_d2, rel=1e-9)
    assert bool(gate[0, 0]) is expect_in_gate, (
        f"d2 = {d2[0, 0]:.6f} vs gate_chi2 = {DEFAULT_PARAMS.gate_chi2}: "
        f"expected in_gate={expect_in_gate}"
    )
    if expect_in_gate:
        assert cost[0, 0] != big_cost(DEFAULT_PARAMS)
    else:
        assert cost[0, 0] == big_cost(DEFAULT_PARAMS)


def test_Rs_accepted_as_single_matrix_and_as_per_measurement_stack():
    """Supporting: one (2,2) broadcast must equal an (M,2,2) stack of the same R."""
    tracks, zs = _random_fixture(SEED + 31, n_tracks=4, n_meas=5)
    R_stack = np.repeat(R_RADAR[None, :, :], zs.shape[0], axis=0)
    assert R_stack.shape == (5, 2, 2)

    c1, g1, d1 = build_cost_matrix(tracks, zs, R_RADAR)
    c2, g2, d2 = build_cost_matrix(tracks, zs, R_stack)

    np.testing.assert_allclose(c1, c2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(d1, d2, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(g1), np.asarray(g2))

    r1 = associate_global(tracks, zs, R_RADAR)
    r2 = associate_global(tracks, zs, R_stack)
    assert r1.assignments == r2.assignments


def test_Rs_per_measurement_stack_is_actually_used_per_measurement():
    """Supporting: a genuinely heterogeneous stack must not collapse to Rs[0]."""
    tr = _simple_track(0, (0.0, 0.0), np.zeros((2, 2)), truth_id="a")
    zs = np.array([[40.0, 0.0], [40.0, 0.0]])
    R_stack = np.stack([R_AIS, R_RADAR])       # same z, wildly different confidence

    _, gate, d2 = build_cost_matrix([tr], zs, R_stack)

    # 40 m at sigma = 5 m is 64 sigma^2; at sigma = 50 m it is 0.64.
    assert d2[0, 0] == pytest.approx(1600.0 / 25.0, rel=1e-9)
    assert d2[0, 1] == pytest.approx(1600.0 / 2500.0, rel=1e-9)
    assert not bool(gate[0, 0])
    assert bool(gate[0, 1])


def test_track_pos_and_pos_cov_mirror_the_filter():
    """Supporting: Track.pos / Track.pos_cov are the filter's position blocks."""
    P = np.diag([100.0, 200.0, 9.0, 16.0])
    P[0, 1] = P[1, 0] = 30.0
    tr = _make_track(7, (12.0, -34.0), (5.0, 6.0), P, truth_id="a")

    np.testing.assert_allclose(np.asarray(tr.pos), np.asarray(tr.kf.state)[:2],
                               rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(tr.pos_cov), np.asarray(tr.kf.cov)[:2, :2],
                               rtol=0, atol=0)
    assert np.asarray(tr.pos).shape == (2,)
    assert np.asarray(tr.pos_cov).shape == (2, 2)

    tr.kf.predict(2.0)
    np.testing.assert_allclose(np.asarray(tr.pos), np.asarray(tr.kf.state)[:2],
                               rtol=0, atol=0)


# --------------------------------------------------------------------------
# GLOBAL VS GREEDY
# --------------------------------------------------------------------------
def test_global_total_cost_is_strictly_below_greedy_on_a_hand_built_trap(capsys):
    """Supporting: greedy's locally-cheapest first pick forces a globally worse total.

    Hand-built 2x2 with an isotropic S = 2500 I, so cost differences are purely
    0.5 * |nu|^2 / 2500 and the geometry is transparent. Euclidean prediction-to-
    measurement distances:

                     z0      z1
        track 0      90      10     <- (0, z1) is the single cheapest cell
        track 1     145     100

    Greedy takes (0, z1) at distance 10 and is then forced onto (1, z0) at 145.
    The joint solution pays 90 and 100 instead, and wins. Both totals are read
    off the SAME cost matrix, so this compares policies, not cost models.
    """
    P_pos = np.diag([1000.0, 1000.0])
    R = np.diag([1500.0, 1500.0])         # S = P_pos + R = 2500 * I exactly

    p0 = np.array([0.0, 0.0])
    p1 = np.array([110.0, 0.0])
    z1 = np.array([10.0, 0.0])            # |z1 - p0| = 10, |z1 - p1| = 100
    d00, d10 = 90.0, 145.0
    x = (d00 ** 2 - d10 ** 2 + 110.0 ** 2) / (2.0 * 110.0)
    z0 = np.array([x, math.sqrt(d00 ** 2 - x * x)])

    tracks = [_simple_track(0, p0, P_pos, truth_id="a"),
              _simple_track(1, p1, P_pos, truth_id="b")]
    zs = np.array([z0, z1])

    cost, gate, _ = build_cost_matrix(tracks, zs, R)
    assert np.asarray(gate).all(), (
        "the trap only means something if all four pairings are inside the gate"
    )
    flat = np.asarray(cost).argmin()
    assert divmod(int(flat), 2) == (0, 1), (
        f"the cheapest single cell should be (0, 1); cost =\n{cost}"
    )

    res_g = associate_global(tracks, zs, R)
    res_r = associate_greedy(tracks, zs, R)
    _assert_partition(res_g, 2, 2)
    _assert_partition(res_r, 2, 2)

    total_g = sum(float(cost[a, b]) for a, b in res_g.assignments)
    total_r = sum(float(cost[a, b]) for a, b in res_r.assignments)

    with capsys.disabled():
        print(f"\n[OPTIMALITY] global total cost = {total_g:.4f} "
              f"{res_g.assignments}, greedy = {total_r:.4f} {res_r.assignments}")

    assert res_g.assignments == [(0, 0), (1, 1)]
    assert res_r.assignments == [(0, 1), (1, 0)]
    assert total_g < total_r, (
        f"global total {total_g:.6f} must be strictly below greedy {total_r:.6f}"
    )


def test_determinant_term_earns_its_keep(capsys):
    """Supporting: the 0.5*logdet(S) term flips the choice a bare Mahalanobis makes.

    Two tracks at the same point, equidistant from one measurement, but with
    very different confidence: diag(4, 4) versus diag(2500, 2500). The uncertain
    track has the SMALLER d2 -- a pure gating distance would hand it the plot --
    while the log-det penalty for its enormous S makes the full NLL prefer the
    confident track, which is what global picks.

    Note this term does NOT decide the symmetric crossing in AC1: there both
    tracks carry the same S and the log-det cancels exactly. This is where it
    earns its keep.

    (The confident/uncertain cost ordering only inverts at |nu| ~ 16.2 m, which
    is within a whisker of where the confident track's own gate closes at
    |nu| ~ 16.3 m, so the inversion is effectively unreachable in-gate -- the
    confident track wins across the whole usable range.)
    """
    R = R_AIS
    confident = _simple_track(0, (0.0, 0.0), np.diag([4.0, 4.0]), truth_id="confident")
    uncertain = _simple_track(1, (0.0, 0.0), np.diag([2500.0, 2500.0]),
                              truth_id="uncertain")
    zs = np.array([[5.0, 0.0]])

    cost, gate, d2 = build_cost_matrix([confident, uncertain], zs, R)
    assert bool(gate[0, 0]) and bool(gate[1, 0]), "both tracks must be in contention"

    with capsys.disabled():
        print(f"\n[LOGDET] confident: d2 = {d2[0, 0]:.4f}, cost = {cost[0, 0]:.4f} | "
              f"uncertain: d2 = {d2[1, 0]:.4f}, cost = {cost[1, 0]:.4f}")

    # A bare Mahalanobis rule would prefer the uncertain track ...
    assert d2[1, 0] < d2[0, 0], (
        "fixture is wrong: the uncertain track must have the smaller d2 for this "
        "test to say anything"
    )
    # ... but the log-det penalty overrules it.
    assert cost[0, 0] < cost[1, 0], (
        f"log-det term failed to overrule the smaller d2: confident cost "
        f"{cost[0, 0]:.4f} vs uncertain {cost[1, 0]:.4f}"
    )

    # The gap is the log-det difference minus the d2 difference; check the pieces.
    logdet_gap = 0.5 * (math.log(np.linalg.det(np.diag([2500.0, 2500.0]) + R))
                        - math.log(np.linalg.det(np.diag([4.0, 4.0]) + R)))
    d2_gap = 0.5 * (float(d2[0, 0]) - float(d2[1, 0]))
    assert float(cost[1, 0] - cost[0, 0]) == pytest.approx(logdet_gap - d2_gap,
                                                          rel=1e-9, abs=1e-9)

    res = associate_global([confident, uncertain], zs, R)
    _assert_partition(res, 2, 1)
    assert res.assignments == [(0, 0)], (
        f"global should give the plot to the track the NLL prefers, got "
        f"{res.assignments}"
    )
    assert sorted(res.unassigned_tracks) == [1]


def test_global_and_greedy_agree_when_the_problem_is_easy():
    """Supporting: with well-separated targets the two policies must not differ."""
    tracks = [
        _simple_track(i, (2000.0 * i, 0.0), R_RADAR, truth_id=f"truth-{i}")
        for i in range(5)
    ]
    zs = np.array([[2000.0 * i + 12.0, -8.0] for i in range(5)])

    res_g = associate_global(tracks, zs, R_RADAR)
    res_r = associate_greedy(tracks, zs, R_RADAR)

    assert res_g.assignments == [(i, i) for i in range(5)]
    assert res_r.assignments == res_g.assignments


# --------------------------------------------------------------------------
# TRACK LIFECYCLE
# --------------------------------------------------------------------------
# Benign hit arguments: d2 and logdet_S small enough that llr climbs for any
# sensible beta_fa, without hard-coding what beta_fa is.
HIT_D2 = 1.0
HIT_LOGDET = 0.0

# Isolate the two deletion paths from one another.
NO_LLR_FLOOR = AssocParams(llr_delete_floor=-1e12)
NO_MISS_LIMIT = AssocParams(delete_after_misses=50, llr_delete_floor=-5.0)


def _fresh_track(params=DEFAULT_PARAMS, track_id=1):
    return _simple_track(track_id, (0.0, 0.0), R_RADAR, truth_id="t", params=params)


def test_lifecycle_starts_tentative_and_alive():
    """Supporting: a brand new track is tentative, alive, with no hits or misses."""
    tr = _fresh_track()
    assert tr.status == "tentative"
    assert tr.is_alive()
    assert tr.hits == 0
    assert tr.misses == 0
    assert len(tr.hit_history) == 0
    assert tr.truth_id == "t"


def test_lifecycle_confirms_at_three_of_three():
    """Supporting: tentative -> confirmed on the third consecutive hit, not before."""
    p = NO_LLR_FLOOR
    tr = _fresh_track(p)

    tr.register_hit(HIT_D2, HIT_LOGDET, p)
    assert tr.status == "tentative"
    tr.register_hit(HIT_D2, HIT_LOGDET, p)
    assert tr.status == "tentative", "must not confirm on 2 of 2"
    tr.register_hit(HIT_D2, HIT_LOGDET, p)
    assert tr.status == "confirmed", "must confirm on the third hit"
    assert tr.hits == 3
    assert tr.misses == 0


def test_lifecycle_confirms_at_three_of_five_with_interleaved_misses():
    """Supporting: the window is 3-of-5, so hit/miss/hit/miss/hit confirms on frame 5.

    Note the misses here are never consecutive, so the coast rule is not touched.
    """
    p = NO_LLR_FLOOR
    tr = _fresh_track(p)

    tr.register_hit(HIT_D2, HIT_LOGDET, p)
    assert tr.status == "tentative"
    tr.register_miss(p)
    assert tr.status == "tentative"
    tr.register_hit(HIT_D2, HIT_LOGDET, p)
    assert tr.status == "tentative"
    tr.register_miss(p)
    assert tr.status == "tentative"
    tr.register_hit(HIT_D2, HIT_LOGDET, p)

    assert tr.status == "confirmed", (
        f"3 hits inside a 5-frame window must confirm; history = {list(tr.hit_history)}"
    )
    assert tr.misses == 0, "a hit resets the CONSECUTIVE miss counter"
    assert sum(bool(h) for h in tr.hit_history) == 3


def test_lifecycle_confirmed_goes_coasting_at_two_consecutive_misses():
    """Supporting: confirmed -> coasting exactly at coast_after_misses consecutive misses."""
    p = NO_LLR_FLOOR
    tr = _fresh_track(p)
    for _ in range(3):
        tr.register_hit(HIT_D2, HIT_LOGDET, p)
    assert tr.status == "confirmed"

    tr.register_miss(p)
    assert tr.status == "confirmed", "one miss must not coast a confirmed track"
    assert tr.misses == 1

    tr.register_miss(p)
    assert tr.status == "coasting"
    assert tr.misses == 2
    assert tr.is_alive(), "coasting is still alive"


def test_lifecycle_coasting_returns_to_confirmed_on_a_hit():
    """Supporting: coasting -> confirmed on the next hit, and the miss run resets."""
    p = NO_LLR_FLOOR
    tr = _fresh_track(p)
    for _ in range(3):
        tr.register_hit(HIT_D2, HIT_LOGDET, p)
    tr.register_miss(p)
    tr.register_miss(p)
    assert tr.status == "coasting"

    tr.register_hit(HIT_D2, HIT_LOGDET, p)
    assert tr.status == "confirmed"
    assert tr.misses == 0


def test_lifecycle_dies_at_ten_consecutive_misses():
    """Supporting: any status -> dead at delete_after_misses TOTAL consecutive misses.

    llr_delete_floor is pushed out of reach so this isolates the miss-count path.
    """
    p = NO_LLR_FLOOR
    tr = _fresh_track(p)
    for _ in range(3):
        tr.register_hit(HIT_D2, HIT_LOGDET, p)
    assert tr.status == "confirmed"

    for i in range(1, p.delete_after_misses):
        tr.register_miss(p)
        assert tr.is_alive(), f"track died at miss {i}, before delete_after_misses"
        assert tr.status != "dead"
    assert tr.misses == p.delete_after_misses - 1

    tr.register_miss(p)
    assert tr.misses == p.delete_after_misses
    assert tr.status == "dead"
    assert not tr.is_alive()


def test_lifecycle_a_hit_resets_the_consecutive_miss_run():
    """Supporting: misses counts CONSECUTIVE misses, so 9 + hit + 9 must not delete."""
    p = NO_LLR_FLOOR
    tr = _fresh_track(p)
    for _ in range(3):
        tr.register_hit(HIT_D2, HIT_LOGDET, p)

    for _ in range(p.delete_after_misses - 1):
        tr.register_miss(p)
    assert tr.is_alive()

    tr.register_hit(HIT_D2, HIT_LOGDET, p)
    assert tr.misses == 0

    for _ in range(p.delete_after_misses - 1):
        tr.register_miss(p)
    assert tr.is_alive(), (
        "a hit did not reset the consecutive-miss run -- 18 non-consecutive misses "
        "must not delete a track whose limit is 10 consecutive"
    )


def test_lifecycle_dead_is_terminal_and_idempotent():
    """Supporting: once dead, further hits and misses must not resurrect the track."""
    p = NO_LLR_FLOOR
    tr = _fresh_track(p)
    for _ in range(p.delete_after_misses):
        tr.register_miss(p)
    assert tr.status == "dead"

    for _ in range(3):
        tr.register_miss(p)
        assert tr.status == "dead"
    for _ in range(5):
        tr.register_hit(HIT_D2, HIT_LOGDET, p)
        assert tr.status == "dead", "a hit resurrected a dead track"
    assert not tr.is_alive()


def test_lifecycle_llr_floor_deletes_independently_of_the_miss_count(capsys):
    """Supporting: the LLR floor is a SECOND, independent deletion path.

    Params are chosen so the floor is crossed after a handful of misses while
    delete_after_misses sits at 50 -- if the track dies here it can only be the
    LLR path. The number of misses required is derived from the track's own
    starting llr, since the contract does not pin the initial value.
    """
    p = NO_MISS_LIMIT
    tr = _fresh_track(p)

    llr0 = float(tr.llr)
    per_miss = math.log(1.0 - p.p_detect)
    assert per_miss < 0.0
    n_needed = int(math.floor((p.llr_delete_floor - llr0) / per_miss)) + 1
    assert 1 <= n_needed < p.delete_after_misses, (
        f"fixture is wrong: needs {n_needed} misses but delete_after_misses is "
        f"{p.delete_after_misses}, so the two paths are not separated"
    )

    for i in range(1, n_needed):
        tr.register_miss(p)
        assert tr.is_alive(), (
            f"died at miss {i} with llr = {tr.llr:.4f}, floor = {p.llr_delete_floor}"
        )

    tr.register_miss(p)

    with capsys.disabled():
        print(f"\n[LLR FLOOR] llr0 = {llr0:.4f}, died after {n_needed} misses at "
              f"llr = {tr.llr:.4f} (floor {p.llr_delete_floor}, "
              f"delete_after_misses = {p.delete_after_misses})")

    assert tr.llr < p.llr_delete_floor
    assert tr.misses < p.delete_after_misses, (
        "the miss-count path fired instead -- this test proves nothing"
    )
    assert tr.status == "dead"
    assert not tr.is_alive()


def test_lifecycle_llr_increments_match_the_formula():
    """Supporting: llr moves by exactly the written amounts on hits and misses."""
    p = NO_LLR_FLOOR
    tr = _fresh_track(p)
    llr0 = float(tr.llr)

    d2, logdet_S = 2.5, 1.75
    tr.register_hit(d2, logdet_S, p)
    expected_hit = (math.log(p.p_detect / p.beta_fa) - 0.5 * d2
                    - math.log(2.0 * math.pi) - 0.5 * logdet_S)
    assert float(tr.llr) == pytest.approx(llr0 + expected_hit, rel=1e-10, abs=1e-10)

    before = float(tr.llr)
    tr.register_miss(p)
    assert float(tr.llr) == pytest.approx(before + math.log(1.0 - p.p_detect),
                                          rel=1e-10, abs=1e-10)


def test_lifecycle_hit_history_is_a_bounded_deque():
    """Supporting: hit_history is a deque capped at confirm_window frames."""
    p = NO_LLR_FLOOR
    tr = _fresh_track(p)

    assert isinstance(tr.hit_history, deque)
    assert tr.hit_history.maxlen == p.confirm_window

    for _ in range(p.confirm_window + 4):
        tr.register_hit(HIT_D2, HIT_LOGDET, p)
    assert len(tr.hit_history) == p.confirm_window

    for _ in range(p.confirm_window + 4):
        tr.register_miss(p)
    assert len(tr.hit_history) == p.confirm_window
    assert not any(tr.hit_history), "the window should have rolled over to all misses"


def test_lifecycle_default_params_match_the_contract():
    """Supporting: the documented defaults are part of the contract (beta_fa excluded)."""
    assert DEFAULT_PARAMS.p_detect == pytest.approx(0.9)
    assert DEFAULT_PARAMS.gate_chi2 == pytest.approx(9.21034, abs=1e-9)
    assert DEFAULT_PARAMS.confirm_hits == 3
    assert DEFAULT_PARAMS.confirm_window == 5
    assert DEFAULT_PARAMS.coast_after_misses == 2
    assert DEFAULT_PARAMS.delete_after_misses == 10
    assert DEFAULT_PARAMS.llr_delete_floor == pytest.approx(-20.0)
    # beta_fa is tuned elsewhere; only its sanity is asserted here.
    assert 0.0 < DEFAULT_PARAMS.beta_fa < 1.0

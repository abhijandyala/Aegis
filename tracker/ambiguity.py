"""Ambiguity detection: is one frame's association close enough to call for a fork?

What this measures
------------------
:mod:`tracker.assoc` returns the single globally optimal interpretation of a
frame. That is the right answer to give a single-hypothesis tracker, and the
wrong answer to give a multi-hypothesis one, because it throws away *how much*
the winner won by. Two ships crossing at 400 m produce a frame where swapping
the two measurements between the two tracks costs almost nothing; the solver
still returns one assignment, with total confidence, and if it picked wrong the
track IDs are swapped forever. This module answers the question the solver does
not: how much worse is the best alternative?

The margin is in nats, and it is a log-likelihood ratio
-------------------------------------------------------
Every cell of the assignment matrix is a negative log-likelihood (see the
:mod:`tracker.assoc` docstring), so the *total* cost of an assignment is
``-log`` of the joint likelihood of that whole frame interpretation --
assignments, misses and births together. The difference between two total costs

    margin = cost_second - cost_best = log( L_best / L_second )

is therefore a log-likelihood *ratio*: "the winner is exp(margin) times more
likely than the runner-up". That is the quantity a hypothesis manager needs.

Deliberately NOT a distance. The obvious alternative -- compare the two
assignments' Mahalanobis distances, or count how many metres apart the two
measurements are -- is not comparable across tracks, because a well-converged
track and a coasting one with a covariance the size of a harbour produce
distances on completely different scales, and neither scale says anything about
births or misses. A nats margin is scale-free and commensurable: 2 nats means
the same thing (about 7.4:1 odds) for a tight radar-fed track and for a dark
vessel being re-acquired.

Reading the scale
-----------------
Total frame costs run to roughly 15-25 nats for a small frame (a single birth
alone is ``-log(1e-6)`` = 13.8 nats), so absolute totals are large and their
difference is the only meaningful number. Typical margins:

    margin      odds        interpretation
    0.0         1:1         a genuine coin flip -- the two interpretations are
                            indistinguishable; forking is mandatory
    0.7         2:1         badly ambiguous
    2.0         7.4:1       the default threshold; the runner-up is still a
                            live possibility over the next few scans
    5.0        148:1        effectively decided
   10.0      22000:1        no contest

``AMBIGUITY_DELTA_NATS = 2.0`` is set where the runner-up is still plausible
enough that carrying it for the ~3 scans of an N-scan window is cheaper than
being wrong. Lower it to fork less; raise it to fork more.

Method
------
Solve the frame normally for ``cost_best``. Take the single cheapest assigned
pair -- the association the solver was most sure of, and hence the one whose
removal most sharply tests the solution -- forbid it, and re-solve for
``cost_second``. If ``cost_second - cost_best < delta`` the frame is ambiguous.

Two implementation points that are easy to get wrong:

* **Total cost, not the sum over assigned pairs.** An assignment that assigns
  fewer pairs is not automatically cheaper: each unassigned track owes
  ``cost_miss`` and each unassigned measurement owes ``cost_birth``. Comparing
  bare pair sums makes "assign nothing" look free and every frame look
  unambiguous. :func:`total_cost` is used for *both* solves so they are
  commensurable.
* **One solve path.** The best solve is run through
  :func:`solve_with_forbidden` with an empty ``forbidden`` set rather than
  through :func:`tracker.assoc.associate_global`, so any difference between the
  two totals comes from the forbidden pair and not from a padding or
  tie-breaking difference between two solvers. (The two agree -- there is a test
  asserting it -- but the margin must not *depend* on that.)

Forbidding a pair uses :func:`tracker.assoc.big_cost`, never ``np.inf``: taking
a miss for the track plus a birth for the measurement is always available in the
padded matrix and always costs strictly less than one BIG entry, so a forbidden
cell can never be selected, while ``inf`` would let ``inf - inf`` into the cost
reduction inside ``linear_sum_assignment``.

No LLM calls, no randomness: same frame in, same report out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from .assoc import (
        AssocParams,
        AssocResult,
        DEFAULT_PARAMS,
        Track,
        big_cost,
        build_cost_matrix,
        cost_birth,
        cost_miss,
    )
except ImportError:  # tracker/ is a namespace package: also support a flat sys.path
    from assoc import (  # type: ignore[no-redef]
        AssocParams,
        AssocResult,
        DEFAULT_PARAMS,
        Track,
        big_cost,
        build_cost_matrix,
        cost_birth,
        cost_miss,
    )

__all__ = [
    "AMBIGUITY_DELTA_NATS",
    "AmbiguityReport",
    "total_cost",
    "solve_with_forbidden",
    "detect_ambiguity",
]

#: Margin below which a frame is declared ambiguous, in nats. 2.0 nats is about
#: 7.4:1 odds -- see the module docstring for the full scale. Tunable.
AMBIGUITY_DELTA_NATS: float = 2.0

Pair = tuple[int, int]


@dataclass
class AmbiguityReport:
    """Verdict on one frame.

    ``margin`` is ``cost_second - cost_best`` in nats and is never negative:
    the best solve optimises over a strict superset of the second's feasible
    set. When no runner-up exists (an empty best assignment -- nothing to
    forbid, which includes every ``N == 0`` or ``M == 0`` frame),
    ``cost_second`` and ``margin`` are ``+inf``, ``second`` is empty and the
    weights are ``(1.0, 0.0)``.
    """

    ambiguous: bool
    cost_best: float
    cost_second: float
    margin: float                       # cost_second - cost_best, in nats
    best: list[Pair] = field(default_factory=list)
    second: list[Pair] = field(default_factory=list)
    weights: tuple[float, float] = (1.0, 0.0)   # (w_best, w_second), sums to 1


def total_cost(
    result: AssocResult,
    cost: np.ndarray,
    params: AssocParams = DEFAULT_PARAMS,
) -> float:
    """Total negative log-likelihood of a whole frame interpretation, in nats.

    ``sum of assigned cells + cost_miss per unassigned track + cost_birth per
    unassigned measurement``. Pass the *unforbidden* cost matrix: forbidden
    cells are never assigned, so the two matrices agree on every term that is
    actually summed, and using one matrix for both solves keeps the totals
    comparable.
    """
    total = 0.0
    for t, m in result.assignments:
        total += float(cost[t, m])
    total += cost_miss(params) * len(result.unassigned_tracks)
    total += cost_birth(params) * len(result.unassigned_meas)
    return total


def _decode(
    rows: np.ndarray,
    cols: np.ndarray,
    n: int,
    m: int,
    gate: np.ndarray,
    forbidden: set[Pair],
) -> AssocResult:
    """Turn a padded ``linear_sum_assignment`` solution into an AssocResult."""
    assignments: list[Pair] = []
    assigned_tracks = np.zeros(n, dtype=bool)
    assigned_meas = np.zeros(m, dtype=bool)
    for r, c in zip(rows, cols):
        r = int(r)
        c = int(c)
        if r < n and c < m:
            # Both checks are belt-and-braces: a gated-out or forbidden cell
            # sits at BIG, which can never beat the always-available
            # miss+birth pair. Dropping such a pair degrades to miss+birth
            # rather than emitting an association we said was impossible.
            if not gate[r, c] or (r, c) in forbidden:
                continue
            assignments.append((r, c))
            assigned_tracks[r] = True
            assigned_meas[c] = True

    return AssocResult(
        assignments=assignments,
        unassigned_tracks=[int(i) for i in np.flatnonzero(~assigned_tracks)],
        unassigned_meas=[int(j) for j in np.flatnonzero(~assigned_meas)],
        cost=np.empty((n, m), dtype=float),   # replaced by the caller
        gate=gate,
        d2=np.empty((n, m), dtype=float),     # replaced by the caller
    )


def solve_with_forbidden(
    tracks: Sequence[Track],
    zs,
    Rs,
    forbidden: Iterable[Pair] = (),
    *,
    params: AssocParams = DEFAULT_PARAMS,
) -> AssocResult:
    """Re-solve a frame with specific ``(track_idx, meas_idx)`` pairs forbidden.

    Identical to :func:`tracker.assoc.associate_global` -- same NLL cost matrix,
    same chi-squared gate, same ``(N+M) x (N+M)`` padding that makes miss and
    birth first-class choices -- except that every pair in ``forbidden`` is
    raised to :func:`tracker.assoc.big_cost` and so can never be selected. With
    an empty ``forbidden`` it *is* the plain solve, which is how
    :func:`detect_ambiguity` gets both of its costs from one code path.

    The returned ``.cost`` is the unmodified matrix (forbidden cells at their
    true cost), so it can be fed straight to :func:`total_cost` for either solve.
    Out-of-range pairs in ``forbidden`` are ignored rather than raising: callers
    forbid pairs harvested from an earlier, possibly larger frame.
    """
    n = len(tracks)
    cost, gate, d2 = build_cost_matrix(tracks, zs, Rs, params=params)
    m = cost.shape[1]

    forbidden_set: set[Pair] = {
        (int(t), int(j))
        for t, j in forbidden
        if 0 <= int(t) < n and 0 <= int(j) < m
    }

    if n + m == 0:
        return AssocResult(
            assignments=[], unassigned_tracks=[], unassigned_meas=[],
            cost=cost, gate=gate, d2=d2,
        )

    big = big_cost(params)
    size = n + m

    # Slice-only construction: with N == 0 or M == 0 the corresponding slices
    # are zero-sized and the whole thing degenerates correctly for free.
    full = np.empty((size, size), dtype=float)
    full[:n, :m] = cost
    for t, j in forbidden_set:
        full[t, j] = big

    miss_block = full[:n, m:]
    miss_block[:] = big
    np.fill_diagonal(miss_block, cost_miss(params))     # track t missed

    birth_block = full[n:, :m]
    birth_block[:] = big
    np.fill_diagonal(birth_block, cost_birth(params))   # measurement j is a birth

    full[n:, m:] = 0.0   # miss-column x birth-row padding is free

    rows, cols = linear_sum_assignment(full)

    result = _decode(rows, cols, n, m, gate, forbidden_set)
    result.cost = cost
    result.d2 = d2
    return result


def _weights(cost_best: float, cost_second: float) -> tuple[float, float]:
    """Normalized relative likelihoods ``(w_best, w_second)``.

    ``w_i`` is proportional to ``exp(-cost_i)``. Costs run to 15-25 nats, where
    ``exp(-cost)`` is 1e-7 to 1e-11 -- not yet a denormal, but the ratio of two
    such numbers loses precision fast, and a 40-track frame pushes the totals
    into the hundreds, where raw ``exp(-cost)`` underflows to exactly 0.0 and
    the normalization becomes 0/0. Subtracting the smaller cost first pins the
    larger weight at exactly 1.0 before normalization and makes the result
    depend only on the (small) margin.
    """
    if not math.isfinite(cost_second):
        return (1.0, 0.0)
    shift = min(cost_best, cost_second)
    e_best = math.exp(-(cost_best - shift))
    e_second = math.exp(-(cost_second - shift))
    denom = e_best + e_second
    if denom <= 0.0:            # unreachable: shift guarantees one term is 1.0
        return (1.0, 0.0)
    return (e_best / denom, e_second / denom)


def detect_ambiguity(
    tracks: Sequence[Track],
    zs,
    Rs,
    *,
    params: AssocParams = DEFAULT_PARAMS,
    delta: float = AMBIGUITY_DELTA_NATS,
) -> AmbiguityReport:
    """Decide whether this frame's association is too close to call.

    Returns the winning and runner-up assignments, their total costs, the nats
    margin between them, and the two normalized hypothesis weights the caller
    should hand to the Aegis hypothesis manager on a fork.

    Degenerate frames (no tracks, no measurements, or a best solution that
    assigns nothing at all) report ``ambiguous=False`` with an infinite margin:
    there is no pair to forbid, so there is no second interpretation to fork
    into.
    """
    best = solve_with_forbidden(tracks, zs, Rs, (), params=params)
    cost = best.cost
    c_best = total_cost(best, cost, params)

    if not best.assignments:
        return AmbiguityReport(
            ambiguous=False,
            cost_best=c_best,
            cost_second=math.inf,
            margin=math.inf,
            best=[],
            second=[],
            weights=(1.0, 0.0),
        )

    # The cheapest assigned pair is the one the solver was most confident in;
    # forbidding it is the sharpest available test of the solution. min() on
    # (cost, track, meas) keeps the choice deterministic under exact ties,
    # which symmetric crossing geometry produces routinely.
    pivot = min(best.assignments, key=lambda p: (float(cost[p[0], p[1]]), p[0], p[1]))

    second = solve_with_forbidden(tracks, zs, Rs, (pivot,), params=params)
    c_second = total_cost(second, cost, params)

    # Clamp at 0: the best solve optimises over a superset of the second's
    # feasible set, so a negative margin can only be float noise on two sums of
    # the same magnitude.
    margin = max(0.0, c_second - c_best)

    return AmbiguityReport(
        ambiguous=bool(margin < delta),
        cost_best=c_best,
        cost_second=c_second,
        margin=margin,
        best=list(best.assignments),
        second=list(second.assignments),
        weights=_weights(c_best, c_second),
    )

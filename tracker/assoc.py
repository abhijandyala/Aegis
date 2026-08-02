"""Measurement-to-track association for maritime multi-target tracking.

Scope
-----
Phase 2 sits on top of :mod:`tracker.kalman`. Given ``N`` predicted tracks and
``M`` position measurements for one frame, it decides which measurement (if
any) updates which track, which tracks were missed, and which measurements are
births. It performs no filter updates itself -- the caller drives
``kf.update``/``Track.register_hit`` with the returned assignment.

Cost derivation
---------------
Every entry of the assignment matrix is a *negative log-likelihood*, so that
"assign", "miss" and "birth" are commensurable and a single linear assignment
solve can trade them off. Under the standard single-target/clutter model with
Gaussian innovation ``nu ~ N(0, S)``, detection probability ``p_D`` and
false-alarm spatial density ``beta_fa`` (m^-2):

    p(assign) = p_D * N(nu; 0, S)
    cost_assign = -log p(assign)
                = 0.5*d2 + 0.5*log det(2*pi*S) - log(p_D)
                = 0.5*d2 + log(2*pi) + 0.5*logdet(S) - log(p_D)

    cost_miss  = -log(1 - p_D)      # the track existed but was not detected
    cost_birth = -log(beta_fa)      # the measurement came from clutter/new target

(the ``log det(2*pi*S)`` of a 2x2 ``S`` is ``2*log(2*pi) + logdet(S)``, hence
the ``log(2*pi) + 0.5*logdet(S)`` above.)

Why the determinant term matters
--------------------------------
Dropping ``0.5*logdet(S)`` and scoring on ``d2`` alone -- the usual shortcut --
makes an uncertain track (large ``P``) look *better* than a confident one,
because inflating ``S`` shrinks every ``d2`` it produces. A coasting track with
a covariance the size of a harbour would then hoover up measurements belonging
to well-converged neighbours. The determinant term charges a track for its own
uncertainty and restores the correct competition. It is also what makes
``cost_assign`` a true probability density, so comparing it against
``cost_birth`` (also a density, m^-2) is dimensionally meaningful: both are
"log per square metre", and the units cancel in the comparison.

Deviation from spec: BIG instead of +inf
----------------------------------------
Gated-out pairs are nominally ``+inf``. This module uses a finite

    BIG = cost_miss + cost_birth + 1.0     (see :func:`big_cost`)

instead. Taking a miss for the track *and* a birth for the measurement is
always available in the padded matrix and always costs strictly less than one
BIG entry, so no optimal solution can ever contain a BIG entry -- behaviour is
identical to ``+inf``, without letting inf/nan into the augmenting-path cost
reduction inside ``linear_sum_assignment`` (where ``inf - inf`` is a real
hazard). Decoded assignments are still re-checked against the gate before being
returned, so a gated-out pair can never reach ``AssocResult.assignments``.

Identity holdout
----------------
``Track.truth_id`` (MMSI / scenario truth) is carried for *scoring only* and is
never read anywhere in the association path. Association depends solely on
geometry and covariance. Scrambling every ``truth_id`` must leave the returned
assignments byte-identical.

Numerics
--------
Gating is fully batched: one broadcast ``np.linalg.solve`` and one
``np.linalg.slogdet`` build the whole ``(N, M)`` problem. Per-pair
``scipy.linalg.solve`` calls would cost ~50-100 us of dispatch each and blow the
20 ms budget for a 40x40 frame by an order of magnitude. ``np.linalg.solve``
broadcasts over leading dimensions and is a genuine solve, so no explicit
inverse is ever formed.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from .kalman import CVKalman
except ImportError:  # tracker/ is a namespace package: also support a flat sys.path
    from kalman import CVKalman

__all__ = [
    "CHI2_GATE_2DOF_99",
    "AssocParams",
    "DEFAULT_PARAMS",
    "Track",
    "AssocResult",
    "big_cost",
    "cost_miss",
    "cost_birth",
    "max_assignable_sigma",
    "build_cost_matrix",
    "associate_global",
    "associate_greedy",
]

# Chi-squared 99th percentile, 2 degrees of freedom: the gate admits 99% of
# correct associations, so a miss is far more likely to be a real miss than a
# gating artefact.
CHI2_GATE_2DOF_99: float = 9.21034

_LOG_2PI: float = math.log(2.0 * math.pi)


@dataclass
class AssocParams:
    """Association, gating and lifecycle parameters.

    ``beta_fa`` is the one knob that sets where assignment stops beating
    miss+birth. For a converged track (P_pos = diag(400, 400) m^2) seen by radar
    (R = diag(2500, 2500) m^2) the implicit d^2 cutoff at 1e-6 m^-2 is ~12.4,
    which brackets the 9.21 chi-squared gate: the gate binds first (so gating is
    not decoration) while marginal in-gate pairs can still lose to a birth.
    """

    p_detect: float = 0.9
    beta_fa: float = 1e-6          # false-alarm density, m^-2
    gate_chi2: float = CHI2_GATE_2DOF_99
    confirm_hits: int = 3          # 3-of-5
    confirm_window: int = 5
    coast_after_misses: int = 2    # consecutive misses -> coasting
    delete_after_misses: int = 10  # TOTAL consecutive misses -> dead
    llr_delete_floor: float = -20.0


DEFAULT_PARAMS = AssocParams()


def cost_miss(params: AssocParams = DEFAULT_PARAMS) -> float:
    """Cost of leaving a track unassigned: ``-log(1 - p_detect)``."""
    return -math.log(1.0 - params.p_detect)


def cost_birth(params: AssocParams = DEFAULT_PARAMS) -> float:
    """Cost of declaring a measurement a birth/false alarm: ``-log(beta_fa)``."""
    return -math.log(params.beta_fa)


def big_cost(params: AssocParams = DEFAULT_PARAMS) -> float:
    """Finite stand-in for ``+inf`` on gated-out pairs.

    Strictly greater than miss+birth, which is always available, so it can never
    be selected. Exposed so tests can assert that no returned assignment sits at
    this value. See the module docstring for why this is not ``+inf``.
    """
    return cost_miss(params) + cost_birth(params) + 1.0


def max_assignable_sigma(params: AssocParams = DEFAULT_PARAMS) -> float:
    """Largest isotropic innovation sigma (metres) that can still win an assignment.

    Once a track's prediction gets uncertain enough, ``cost_assign`` exceeds
    ``cost_miss + cost_birth`` *even at d2 = 0*, and the solver will always prefer
    to call the track missed and the measurement a birth. The track then can never
    be assigned again, no matter how well the measurement lines up.

    Setting ``cost_assign(d2=0) = cost_miss + cost_birth`` for an isotropic
    ``S = s*I`` gives ``log(2*pi) + log(s) - log(p_D) = cost_miss + cost_birth``,
    hence ``sqrt(s)`` below.

    THIS IS THE BINDING CONSTRAINT ON DARK-VESSEL RE-ASSOCIATION. A vessel that
    goes dark grows its covariance without bound, so ``beta_fa`` -- not the gate --
    sets how long it can stay re-acquirable:

        beta_fa    ceiling
        1e-5         378 m
        1e-6        1197 m     <- default
        1e-7        3785 m
        1e-8       11968 m

    At the default, a dark vessel is unclaimable once its position sigma passes
    ~1.2 km, which for a 12 kn vessel under q = 0.5 m^2/s^3 is a matter of a few
    minutes. Re-association over a longer gap needs a smaller ``beta_fa`` (which
    also makes spurious births more expensive), or a re-association step that does
    not put the track in competition with birth in the same assignment.
    """
    s = math.exp(cost_miss(params) + cost_birth(params)
                 - _LOG_2PI + math.log(params.p_detect))
    return math.sqrt(s)


class Track:
    """One target hypothesis: a filter plus its lifecycle bookkeeping.

    ``misses`` counts *consecutive* misses and resets on any hit. Deletion at
    ``delete_after_misses`` is a TOTAL consecutive count, not an additional
    budget granted after coasting begins: a track that starts missing at frame k
    coasts at k+2 and dies at k+10, not k+12.
    """

    # Deliberately no __slots__: Phase 3 and scoring harnesses attach their own
    # bookkeeping to tracks, and slots would reject it at runtime.

    def __init__(self, id: int, kf: CVKalman, truth_id: str | None = None,
                 params: AssocParams = DEFAULT_PARAMS) -> None:
        self.id: int = int(id)
        self.kf: CVKalman = kf
        self.status: str = "tentative"
        self.hits: int = 0
        self.misses: int = 0
        self.hit_history: collections.deque = collections.deque(
            maxlen=params.confirm_window
        )
        self.llr: float = 0.0
        # Held out for scoring only. Never read by anything in this module.
        self.truth_id: str | None = truth_id
        self.params: AssocParams = params

    @property
    def pos(self) -> np.ndarray:
        """Predicted position ``[x, y]``, shape (2,)."""
        return self.kf.state[:2]

    @property
    def pos_cov(self) -> np.ndarray:
        """Predicted position covariance, shape (2, 2)."""
        return self.kf.cov[:2, :2]

    def register_hit(self, d2: float, logdet_S: float,
                     params: AssocParams | None = None) -> None:
        """Record an association with squared Mahalanobis ``d2`` and ``logdet_S``.

        Falls back to the params the track was built with. Defaulting to
        DEFAULT_PARAMS instead would silently apply stock thresholds to a track
        constructed with tuned ones -- which matters once Phase 3 spawns tracks
        in bulk from a hypothesis branch.
        """
        params = self.params if params is None else params
        if self.status == "dead":
            return  # dead is absorbing

        self.hits += 1
        self.misses = 0
        self.hit_history.append(True)

        # llr -= (cost_assign - cost_birth): the evidence that this measurement
        # came from the track rather than from clutter. A tight, likely
        # association raises the score; a marginal one can still lower it.
        self.llr += (math.log(params.p_detect / params.beta_fa)
                     - 0.5 * float(d2) - _LOG_2PI - 0.5 * float(logdet_S))

        if self.status == "coasting":
            # A coasting track is already confirmed history; one hit reinstates it.
            self.status = "confirmed"
        elif self.status == "tentative":
            # M-of-N over the sliding window, so confirmation can happen at
            # frame 3 on a clean 3/3.
            if sum(self.hit_history) >= params.confirm_hits:
                self.status = "confirmed"

        self._apply_llr_floor(params)

    def register_miss(self, params: AssocParams | None = None) -> None:
        """Record that no measurement was associated to this track this frame.

        Falls back to the track's own params; see :meth:`register_hit`.
        """
        params = self.params if params is None else params
        if self.status == "dead":
            return

        self.misses += 1
        self.hit_history.append(False)
        self.llr += math.log(1.0 - params.p_detect)

        if self.misses >= params.delete_after_misses:
            self.status = "dead"
        elif self.status == "confirmed" and self.misses >= params.coast_after_misses:
            self.status = "coasting"

        self._apply_llr_floor(params)

    def _apply_llr_floor(self, params: AssocParams) -> None:
        """Score-based deletion, independent of the miss counter.

        Either path can kill a track on its own; whichever fires first wins.
        This catches the track that keeps getting fed marginal, barely-in-gate
        measurements -- its miss counter never climbs, but its score bleeds out.
        """
        if self.llr < params.llr_delete_floor:
            self.status = "dead"

    def is_alive(self) -> bool:
        """True while the track has not been deleted."""
        return self.status != "dead"


@dataclass
class AssocResult:
    """Outcome of one association frame. Indices are positional into ``tracks``/``zs``."""

    assignments: list[tuple[int, int]]   # (track_index, meas_index), sorted by track
    unassigned_tracks: list[int]         # -> register_miss
    unassigned_meas: list[int]           # -> track births
    cost: np.ndarray                     # (N, M), BIG where gated out
    gate: np.ndarray                     # (N, M) bool, True = inside gate
    d2: np.ndarray                       # (N, M) squared Mahalanobis distance


def _coerce_inputs(zs, Rs) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(zs, Rs)`` as ``(M, 2)`` and ``(M, 2, 2)`` float arrays."""
    z = np.asarray(zs, dtype=float)
    # reshape(-1, 2) rather than a shape check so [] and np.empty((0,)) both
    # land on the legitimate M=0 case.
    z = z.reshape(-1, 2)
    m = z.shape[0]

    R = np.asarray(Rs, dtype=float)
    if R.ndim == 2:
        # Single R shared by every measurement. broadcast_to gives a read-only
        # view; nothing here mutates R, and avoiding the copy keeps 40x40 cheap.
        R = np.broadcast_to(R.reshape(1, 2, 2), (m, 2, 2))
    elif R.ndim == 3:
        R = R.reshape(-1, 2, 2)
        if R.shape[0] != m:
            raise ValueError(
                f"Rs has {R.shape[0]} entries but zs has {m} measurements"
            )
    else:
        raise ValueError(f"Rs must be (2,2) or (M,2,2), got shape {R.shape}")

    return z, R


def build_cost_matrix(
    tracks: Sequence[Track],
    zs,
    Rs,
    *,
    params: AssocParams = DEFAULT_PARAMS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the ``(N, M)`` NLL cost matrix, gate mask and squared distances.

    Fully batched: two LAPACK calls total, regardless of N and M.
    """
    z, R = _coerce_inputs(zs, Rs)
    n = len(tracks)
    m = z.shape[0]

    if n == 0 or m == 0:
        empty = np.empty((n, m), dtype=float)
        return empty, np.zeros((n, m), dtype=bool), np.empty((n, m), dtype=float)

    x_pos = np.stack([t.pos for t in tracks])          # (N,2)
    P_pos = np.stack([t.pos_cov for t in tracks])      # (N,2,2)

    S = P_pos[:, None, :, :] + R[None, :, :, :]        # (N,M,2,2)
    nu = z[None, :, :] - x_pos[:, None, :]             # (N,M,2)

    # solve() broadcasts over the (N,M) leading dims: one dispatch, not N*M.
    # Never np.linalg.inv -- an explicit inverse is both slower and less stable.
    sol = np.linalg.solve(S, nu[..., None])[..., 0]    # (N,M,2)
    d2 = np.einsum("nmi,nmi->nm", nu, sol)

    sign, logdet = np.linalg.slogdet(S)
    if np.any(sign <= 0.0):
        raise np.linalg.LinAlgError(
            "innovation covariance S is not positive definite; "
            "check that every R and every track position covariance is PD"
        )

    gate = d2 <= params.gate_chi2

    cost = (0.5 * d2 + _LOG_2PI + 0.5 * logdet - math.log(params.p_detect))
    # Gated-out pairs are unreachable rather than merely expensive.
    np.copyto(cost, big_cost(params), where=~gate)

    return cost, gate, d2


def _empty_result(n: int, m: int) -> AssocResult:
    """Result for a frame with nothing to solve (N == 0 or M == 0)."""
    return AssocResult(
        assignments=[],
        unassigned_tracks=list(range(n)),
        unassigned_meas=list(range(m)),
        cost=np.empty((n, m), dtype=float),
        gate=np.zeros((n, m), dtype=bool),
        d2=np.empty((n, m), dtype=float),
    )


def associate_global(
    tracks: Sequence[Track],
    zs,
    Rs,
    *,
    params: AssocParams = DEFAULT_PARAMS,
) -> AssocResult:
    """Globally optimal association by padded linear assignment.

    The ``(N+M) x (N+M)`` padded matrix makes "miss" and "birth" first-class
    choices with their own costs, so the solver minimises the total NLL of the
    whole frame interpretation rather than just of the assigned subset.
    """
    z, R = _coerce_inputs(zs, Rs)
    n = len(tracks)
    m = z.shape[0]
    if n + m == 0:
        return _empty_result(0, 0)

    cost, gate, d2 = build_cost_matrix(tracks, z, R, params=params)

    big = big_cost(params)
    size = n + m
    # Fill by slices only: with N=0 or M=0 the corresponding slices are
    # zero-sized and the whole construction degenerates correctly for free.
    full = np.empty((size, size), dtype=float)
    full[:n, :m] = cost

    miss_block = full[:n, m:]
    miss_block[:] = big
    np.fill_diagonal(miss_block, cost_miss(params))     # track n misses <-> col m+n

    birth_block = full[n:, :m]
    birth_block[:] = big
    np.fill_diagonal(birth_block, cost_birth(params))   # meas m is a birth

    # Padding: miss-column x birth-row pairings carry no extra cost.
    full[n:, m:] = 0.0

    rows, cols = linear_sum_assignment(full)

    assignments: list[tuple[int, int]] = []
    assigned_tracks = np.zeros(n, dtype=bool)
    assigned_meas = np.zeros(m, dtype=bool)
    for r, c in zip(rows, cols):
        if r < n and c < m:
            # Defensive: BIG can never win against miss+birth, so this should be
            # unreachable. Dropping the pair degrades to miss+birth rather than
            # emitting an association the gate rejected.
            if not gate[r, c]:
                continue
            assignments.append((int(r), int(c)))
            assigned_tracks[r] = True
            assigned_meas[c] = True

    # linear_sum_assignment returns rows sorted, so assignments already are.
    return AssocResult(
        assignments=assignments,
        unassigned_tracks=[int(i) for i in np.flatnonzero(~assigned_tracks)],
        unassigned_meas=[int(j) for j in np.flatnonzero(~assigned_meas)],
        cost=cost,
        gate=gate,
        d2=d2,
    )


def associate_greedy(
    tracks: Sequence[Track],
    zs,
    Rs,
    *,
    params: AssocParams = DEFAULT_PARAMS,
) -> AssocResult:
    """Greedy nearest-in-NLL baseline, drop-in swappable with :func:`associate_global`.

    Deliberately shares the *same* gate and the *same* NLL cost matrix as the
    global solver; the only difference is the strategy (commit best-first,
    locally, without backtracking). An honest comparison isolates the assignment
    algorithm instead of also swapping the distance metric underneath it.
    """
    z, R = _coerce_inputs(zs, Rs)
    n = len(tracks)
    m = z.shape[0]
    if n == 0 or m == 0:
        return _empty_result(n, m)

    cost, gate, d2 = build_cost_matrix(tracks, z, R, params=params)

    ti, mi = np.nonzero(gate)
    c = cost[ti, mi]
    # lexsort's last key is primary: cost, then track then measurement index.
    # Explicit index tie-break keeps the output reproducible when two pairs
    # score identically (common with a shared R and symmetric geometry).
    order = np.lexsort((mi, ti, c))

    assigned_tracks = np.zeros(n, dtype=bool)
    assigned_meas = np.zeros(m, dtype=bool)
    assignments: list[tuple[int, int]] = []
    for k in order:
        t = int(ti[k])
        j = int(mi[k])
        if assigned_tracks[t] or assigned_meas[j]:
            continue
        assigned_tracks[t] = True
        assigned_meas[j] = True
        assignments.append((t, j))

    assignments.sort()

    return AssocResult(
        assignments=assignments,
        unassigned_tracks=[int(i) for i in np.flatnonzero(~assigned_tracks)],
        unassigned_meas=[int(j) for j in np.flatnonzero(~assigned_meas)],
        cost=cost,
        gate=gate,
        d2=d2,
    )

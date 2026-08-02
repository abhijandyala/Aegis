"""Dark-vessel coasting: uncertainty growth, display ellipses and re-attribution.

Scope
-----
Phase 4 answers one question: *a vessel switched its AIS transponder off some
minutes ago; an unattributed radar detection has just appeared -- is it the same
vessel, and how sure are we?* This module owns the geometry of the growing
uncertainty region and the probability arithmetic of re-attribution. It holds no
state, reads no files, and knows nothing about zones, severities or the
hypothesis graph -- all of that lives in the Aegis orchestration layer.

Coasting semantics
------------------
While a track is dark the filter only ever ``predict``s. With no measurement to
pull it back, ``P`` grows **without bound** -- under the DWNA model of
:mod:`tracker.kalman` the position variance accumulates roughly ``q*dt*T^3/3``
over a dark interval ``T`` stepped at cadence ``dt``. That growth is not a bug
to be capped; it *is* the product. The ellipse widening on the display is the
honest statement "the vessel is somewhere in here", and the fact that it widens
super-linearly in time is why a re-attribution after ten minutes is a weaker
claim than one after two, and must score as such.

Two chi-squared constants, and why they differ
----------------------------------------------
``CHI2_95_2DOF = 5.991`` is the **display** constant: the 95% confidence region
that gets drawn. ``GATE_CHI2 = 9.21034`` (:data:`tracker.assoc.CHI2_GATE_2DOF_99`)
is the **association** constant: the 99% gate a candidate must pass to be
considered at all. They are deliberately not the same number. A drawn region
should be the one an operator reads as "almost certainly in there" without it
swallowing half the chart, while a gate should reject only what is genuinely
implausible -- being stingy at the gate throws away correct re-attributions,
which is the expensive error here. So a detection can legitimately sit *outside*
the drawn ellipse and still be accepted: between 5.991 and 9.21 lies the 95-99%
annulus. That is intended, and worth saying out loud in the brief rather than
letting it look like an inconsistency.

The false-alarm term
--------------------
The posterior for candidate ``i`` is

    p_i = L_i / (sum_j L_j + BETA_FA_REACQ)

where each ``L_i`` is a Gaussian *density* in m^-2 and ``BETA_FA_REACQ`` is the
spatial density of detections that belong to no coasted track at all -- new
contacts, clutter, a vessel that was never tracked. It is the same physical
quantity as ``AssocParams.beta_fa`` in :mod:`tracker.assoc`, which is what makes
adding it to a sum of ``L_i`` dimensionally honest: everything in that
denominator is "probability mass per square metre".

Structurally, that term is the whole point. Without it a single surviving
candidate always scores ``p = 1.0`` no matter how absurd the geometry, because
it is the only explanation on offer. With it, "none of these" is always an
explanation on offer, so ``p < 1`` strictly, for free, with no clamping. A
strong match on a tight prediction still reaches ~0.96; a marginal one at the
edge of the gate lands near 0.18. Both are true statements.

Calibration -- and its tension with assoc.max_assignable_sigma
--------------------------------------------------------------
With a single candidate, ``p = L / (L + beta)``, so ``p > 0.90`` needs
``L > 9*beta``. A perfect detection (``d2 = 0``) on an isotropic ``S ~ SIGMA^2 I``
gives ``L = 1/(2*pi*SIGMA^2)``, hence a hard ceiling on how uncertain a track may
be and still be re-attributed with confidence::

    beta_fa     SIGMA ceiling at p = 0.90
    1e-6            133 m
    1e-7            421 m
    1e-8           1330 m
    1e-9           4205 m      <- BETA_FA_REACQ
    1e-10         13298 m

Measured against the actual filter (``q = 0.01 m^2/s^3``, coasted at a 10 s
radar cadence, seeded from a two-point AIS init) the position sigma of a dark
track reaches ~970 m at 5 minutes and ~2720 m at 10 minutes. ``1e-9`` clears the
whole 5-10 minute demo window with margin; ``1e-8`` would lose the vessel some
time after 6 minutes. The cadence matters as much as the duration -- sigma
scales as ``sqrt(dt)`` under DWNA -- so "dark for 10 minutes" does not pin a
sigma on its own, and this ceiling is quoted for a 10 s coast step.

``1e-9 m^-2`` is about one unattributed contact per 1000 km^2. That is three
decades below ``AssocParams.beta_fa = 1e-6``, and legitimately so: re-association
is only ever asked about a detection that has *already* failed normal
association, so the population it competes against is the residue, not the raw
plot rate.

The two acceptance criteria pull in opposite directions, and it is worth being
explicit about the window where both hold. ``p(d2=0) > 0.90`` needs
``sigma < 4205 m``; ``p(d2=6.25) < 0.94`` -- a 2.5-sigma offset must *not* score
like a match -- needs ``sigma > 668 m``. So at ``beta = 1e-9`` both hold for::

    sigma in (668 m, 4205 m)   ~  4 to 13 minutes dark at a 10 s coast cadence

Below that band the track is still so tight that even a 2.5-sigma miss scores
0.99, which is correct behaviour (the prediction really is that good) but is not
the demo. Above it, confidence in a perfect hit falls under 0.90 and the vessel
is effectively lost. Measured across the band: at 240 s, ``p(0) = 0.997`` and
``p(2.5 sigma) = 0.934``; at 600 s, ``0.956`` and ``0.486``; at 800 s, ``0.902``
and ``0.287``. The 5-10 minute demo interval sits comfortably inside.

This is not "make beta tiny so p comes out high". The far end is carried by the
gate, not by beta: a detection 10 sigma out has ``d2 = 100 >> 9.21`` and is
rejected outright regardless of beta. And within the gate the ratio structure is
scale-free -- at the 10-minute demo sigma, ``d2 = 0`` gives 0.96 while a 2.5-sigma
offset gives 0.49 and the gate edge gives 0.18. Shrinking beta further would
flatten those apart-cases upward, which is exactly the failure mode to avoid.

Note the mirror: :func:`tracker.assoc.max_assignable_sigma` documents a ceiling
of the same shape from a different criterion (assignment must beat miss+birth in
the linear-assignment solve, 1197 m at beta = 1e-6, versus 133 m here for
posterior > 0.90). Same tension, same single knob, two thresholds. They are not
the same number and neither is a typo -- one asks "can this pair win an
assignment", the other asks "is the resulting claim strong enough to brief".

Numerics
--------
``det(2*pi*S)`` for 2x2 ``S`` is ``(2*pi)^2 * det(S)``. A track dark for ten
minutes has ``P_pos ~ 7e6 m^2``, so ``det(S) ~ 5e13`` and ``L ~ 2e-8``: computing
the density directly is a race between a vanishing exponential and an enormous
determinant, and the naive expression silently returns zeros. Everything here is
therefore done in log space via ``np.linalg.slogdet``, with the maximum log-term
-- *including* ``log(beta)`` -- subtracted before any exponentiation, so the
largest term is exactly 1.0 and nothing underflows. The log-likelihood

    log L = -0.5*d2 - log(2*pi) - 0.5*logdet(S)

is precisely ``assoc.py``'s ``cost_assign`` negated and without its ``p_D`` term,
which is the consistency argument: the two modules score a pair identically.
``d2`` is obtained with ``np.linalg.solve``; no explicit inverse is ever formed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

try:
    from .assoc import CHI2_GATE_2DOF_99
except ImportError:  # tracker/ is a namespace package: also support a flat sys.path
    from assoc import CHI2_GATE_2DOF_99

__all__ = [
    "CHI2_95_2DOF",
    "GATE_CHI2",
    "BETA_FA_REACQ",
    "MPS_TO_KN",
    "ellipse_axes",
    "ellipse_polygon",
    "clip_to_water",
    "ReassocCandidate",
    "DarkTrack",
    "reassociate",
    "sigma_ceiling_for_p90",
]

# 95th percentile of chi-squared with 2 dof: the DISPLAY ellipse. See the module
# docstring for why this is not the gate constant.
CHI2_95_2DOF: float = 5.991

# 99th percentile, 2 dof: the association gate. Re-exported from assoc rather
# than re-typed, so the two phases can never drift apart.
GATE_CHI2: float = CHI2_GATE_2DOF_99

# False-alarm spatial density for re-acquisition, m^-2. ~1 unattributed contact
# per 1000 km^2. Chosen so a vessel dark for the 5-10 minute demo window still
# clears p = 0.90 on a perfect detection; see the calibration section above.
BETA_FA_REACQ: float = 1e-9

MPS_TO_KN: float = 1.943844492440605  # 1 m/s in knots (1 kn = 1852 m/h exactly)

_LOG_2PI: float = math.log(2.0 * math.pi)


def sigma_ceiling_for_p90(beta_fa: float = BETA_FA_REACQ) -> float:
    """Largest isotropic sigma (m) at which a *perfect* detection still scores p > 0.90.

    Inverts ``p = L/(L + beta) > 0.9`` with ``L = 1/(2*pi*sigma^2)`` at ``d2 = 0``.
    Exposed so the demo brief can state "this track is re-acquirable for another
    N metres of drift" instead of hard-coding a number that would rot the moment
    ``BETA_FA_REACQ`` moves. Mirrors :func:`tracker.assoc.max_assignable_sigma`.
    """
    return math.sqrt(1.0 / (18.0 * math.pi * float(beta_fa)))


def ellipse_axes(P_pos: ArrayLike,
                 conf_chi2: float = CHI2_95_2DOF) -> tuple[float, float, float]:
    """Return ``(semi_major_m, semi_minor_m, theta_degrees)`` of the confidence ellipse.

    ``theta`` is the bearing of the major axis measured counter-clockwise from
    the ENU +x (east) axis, in degrees, in (-180, 180]. An ellipse axis is
    undirected, so ``theta`` and ``theta +/- 180`` describe the same ellipse and
    which one comes out depends on the sign LAPACK happens to give the
    eigenvector -- a 30-degree major axis may legitimately report as -150. It is
    left unnormalised because that is what the plotting rotation consumes and
    normalising would only hide the ambiguity, not remove it. For a circular
    covariance the axes are degenerate and ``theta`` is arbitrary.

    ``eigh`` is used rather than ``eig`` because ``P_pos`` is symmetric: it
    returns real eigenvalues in *ascending* order with orthonormal eigenvectors,
    so the MAJOR axis is the LAST column, not the first. Getting that backwards
    draws every ellipse rotated 90 degrees, which looks plausible on a
    near-circular covariance and is wrong exactly when the covariance is
    elongated -- i.e. exactly on a long-dark track, the case that matters.
    """
    P = np.asarray(P_pos, dtype=float).reshape(2, 2)
    # Symmetrise: a covariance propagated through millions of cycles can pick up
    # femto-scale asymmetry, and eigh reads only one triangle -- so which
    # triangle it reads would otherwise be observable in the output.
    P = 0.5 * (P + P.T)

    vals, vecs = np.linalg.eigh(P)
    # A PSD-but-near-singular block can yield a tiny negative eigenvalue from
    # round-off; sqrt of that is NaN and poisons the whole ellipse downstream.
    vals = np.maximum(vals, 0.0)

    a, b = np.sqrt(float(conf_chi2) * vals[::-1])
    theta = math.degrees(math.atan2(vecs[1, -1], vecs[0, -1]))

    assert a >= b, f"major axis must dominate: got a={a!r}, b={b!r}"
    return float(a), float(b), float(theta)


def ellipse_polygon(center: ArrayLike, P_pos: ArrayLike,
                    conf_chi2: float = CHI2_95_2DOF, n: int = 64) -> Polygon:
    """``n``-point polygon approximation of the confidence ellipse, in ENU metres.

    64 points keeps the maximum radial error under 0.1% of the semi-major axis --
    below a pixel at any sane display scale, and cheap enough to rebuild every
    frame for every coasting track. The ring is built by rotating the axis-aligned
    ellipse rather than by sampling a parametric bearing, so it stays exact for
    highly eccentric covariances where the two differ.
    """
    c = np.asarray(center, dtype=float).reshape(2)
    a, b, theta_deg = ellipse_axes(P_pos, conf_chi2)

    # Open ring: shapely closes the polygon itself, and an explicit duplicate
    # last vertex would show up in every coordinate count downstream.
    t = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    th = math.radians(theta_deg)
    cos_t, sin_t = math.cos(th), math.sin(th)

    ex = a * np.cos(t)
    ey = b * np.sin(t)
    xs = c[0] + cos_t * ex - sin_t * ey
    ys = c[1] + sin_t * ex + cos_t * ey

    return Polygon(np.column_stack((xs, ys)))


def clip_to_water(poly: BaseGeometry, land: BaseGeometry | None) -> BaseGeometry:
    """Subtract ``land`` from ``poly`` -- the uncertainty region minus dry ground.

    A vessel cannot be ashore, so a search region that covers a headland is
    overstating the area to search and understating the confidence. Removing the
    land is the detail a maritime analyst notices immediately.

    Tolerates ``land=None`` (returns ``poly`` unchanged), a result that is empty
    (the whole region was land) and a result that is a MultiPolygon (an island
    split it in two). The geometry type is returned as-is rather than coerced,
    because "the search region is two disjoint lobes" is real information.

    ``buffer(0)`` is applied to a self-intersecting input **only when it is
    actually invalid**. Applying it unconditionally is the common shortcut and a
    bad one: it re-nodes valid geometry, can collapse thin slivers we were handed
    deliberately, and hides upstream bugs that produced a broken polygon.
    """
    if land is None:
        return poly
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly.difference(land)


@dataclass(frozen=True)
class ReassocCandidate:
    """One coasting track scored as an explanation for one unattributed detection.

    Rejected candidates are returned too, populated with their real ``d2`` and
    ``likelihood`` but ``p = 0.0`` and a ``reason``. An intelligence brief that
    says "considered and rejected 3 other tracks, nearest at 14 sigma" is far
    more convincing than one that silently shows a single answer.

    ``implied_speed_*`` is the speed the vessel would have had to sustain to get
    from its last *measured* fix to this detection. It is not used in the
    probability at all -- it is a plausibility cross-check for the operator,
    because the filter will happily accept a geometrically fine hypothesis that
    requires 60 knots from a bulk carrier.
    """

    track_id: int
    p: float                  # posterior probability this detection is this track
    d2: float                 # squared Mahalanobis distance
    sigma_distance: float     # sqrt(d2) -- "how many sigma away"
    likelihood: float         # L_i, the Gaussian density value (m^-2)
    time_dark_s: float
    range_m: float            # metres from the track's last MEASURED fix to z
    implied_speed_mps: float  # range_m / time_dark_s (0.0 if time_dark_s <= 0)
    implied_speed_kn: float
    gated_out: bool
    reason: str               # "" if accepted, else why it was rejected


@dataclass
class DarkTrack:
    """Minimal standalone view of a coasting track.

    Deliberately *not* a reference to :class:`tracker.assoc.Track`: this module
    scores a frozen snapshot, and holding a live track would mean the numbers in
    a brief silently change if anything calls ``predict`` between scoring and
    rendering. :meth:`from_track` copies everything for that reason --
    ``Track.pos`` returns a view into ``kf.state``, which mutates in place.

    ``last_fix_xy`` / ``last_fix_t`` describe the last position actually
    MEASURED, not the current prediction. They exist only for the range and
    implied-speed cross-check; the probability uses ``pos``/``pos_cov``.
    """

    id: int
    pos: np.ndarray            # (2,) current PREDICTED position
    pos_cov: np.ndarray        # (2,2) current predicted position covariance
    last_fix_xy: np.ndarray    # (2,) last position actually MEASURED
    last_fix_t: float          # timestamp of that fix
    label: str = ""

    def __post_init__(self) -> None:
        # Coerce here rather than trusting callers: a (1,2) or a list slips
        # through every arithmetic path below and only fails at the einsum.
        self.id = int(self.id)
        self.pos = np.array(self.pos, dtype=float).reshape(2)
        self.pos_cov = np.array(self.pos_cov, dtype=float).reshape(2, 2)
        self.last_fix_xy = np.array(self.last_fix_xy, dtype=float).reshape(2)
        self.last_fix_t = float(self.last_fix_t)

    @classmethod
    def from_track(cls, track: object, last_fix_xy: ArrayLike | None = None,
                   last_fix_t: float | None = None,
                   label: str = "") -> "DarkTrack":
        """Snapshot an :class:`tracker.assoc.Track` (or anything with the same shape).

        Only ``.id``, ``.pos`` and ``.pos_cov`` are read, so this works for any
        duck-typed stand-in and keeps Phase 4 free of assoc's internals.

        ``last_fix_xy``/``last_fix_t`` default to the *current* prediction and
        filter time. That default is exactly right at the moment the track goes
        dark (prediction == last fix) and increasingly wrong afterwards, so any
        caller that has kept the real last fix should pass it -- otherwise the
        implied-speed cross-check silently measures drift since the last call
        instead of since the last transponder report.
        """
        pos = np.array(track.pos, dtype=float).reshape(2)
        cov = np.array(track.pos_cov, dtype=float).reshape(2, 2)
        fix = pos.copy() if last_fix_xy is None else last_fix_xy
        if last_fix_t is None:
            last_fix_t = float(getattr(track, "kf").t) if hasattr(track, "kf") else 0.0
        return cls(id=int(track.id), pos=pos, pos_cov=cov,
                   last_fix_xy=fix, last_fix_t=float(last_fix_t), label=label)


def reassociate(
    z: ArrayLike,
    R: ArrayLike,
    dark_tracks: Sequence[DarkTrack],
    t_now: float,
    *,
    beta_fa: float = BETA_FA_REACQ,
    gate_chi2: float = GATE_CHI2,
) -> list[ReassocCandidate]:
    """Rank coasting tracks as explanations for one unattributed detection ``z``.

    Returns **all** candidates sorted by ``p`` descending (ties broken by
    ascending ``d2``, so the rejected tail reads nearest-first), including gated-out
    ones with ``p = 0.0`` and a ``reason``. Gated-out candidates are excluded from
    the denominator: a hypothesis the gate has already ruled out must not dilute
    the probability of the survivors.

    An empty ``dark_tracks`` returns ``[]`` -- there is no track to blame, which
    is a perfectly ordinary frame, not an error.
    """
    z = np.asarray(z, dtype=float).reshape(2)
    R = np.asarray(R, dtype=float).reshape(2, 2)
    t_now = float(t_now)
    gate_chi2 = float(gate_chi2)
    beta_fa = float(beta_fa)

    n = len(dark_tracks)
    if n == 0:
        return []

    pos = np.stack([t.pos for t in dark_tracks])          # (N,2)
    P = np.stack([t.pos_cov for t in dark_tracks])        # (N,2,2)

    S = P + R[None, :, :]                                  # (N,2,2)
    nu = z[None, :] - pos                                  # (N,2)

    # solve() broadcasts over the leading track axis: one LAPACK dispatch for
    # the whole candidate set, and a genuine solve rather than an inverse.
    sol = np.linalg.solve(S, nu[..., None])[..., 0]        # (N,2)
    d2 = np.einsum("ni,ni->n", nu, sol)

    sign, logdet = np.linalg.slogdet(S)
    if np.any(sign <= 0.0):
        raise np.linalg.LinAlgError(
            "innovation covariance S is not positive definite; check that R and "
            "every dark track's pos_cov are PD"
        )

    # log L = -0.5*d2 - log(2*pi) - 0.5*logdet(S). This is assoc.py's
    # cost_assign negated, minus its p_D term -- the two modules agree on the
    # score of a pair by construction, not by coincidence.
    log_L = -0.5 * d2 - _LOG_2PI - 0.5 * logdet

    accepted = d2 <= gate_chi2

    # --- posterior, entirely in log space -------------------------------------
    # The false-alarm density joins the log-sum-exp as just another term, which
    # is what makes p < 1 structural rather than clamped: the "none of these"
    # hypothesis is always in the denominator and never in a numerator.
    p = np.zeros(n, dtype=float)
    if np.any(accepted):
        log_beta = math.log(beta_fa)
        terms = np.concatenate([log_L[accepted], [log_beta]])
        # Subtract the max over ALL terms including beta before exponentiating:
        # at a ten-minute dark sigma every log_L is around -18, and exponentiating
        # them raw is where the naive implementation quietly returns zeros.
        m = float(np.max(terms))
        denom = float(np.sum(np.exp(terms - m)))
        p[accepted] = np.exp(log_L[accepted] - m) / denom

    # Likelihoods are reported for rejected candidates too (they are honest
    # numbers, just not competing), and may legitimately underflow to 0.0 for a
    # detection tens of sigma out. That is a display value, never a divisor.
    L = np.exp(log_L)

    out: list[ReassocCandidate] = []
    for i, trk in enumerate(dark_tracks):
        d2_i = float(d2[i])
        sig = math.sqrt(max(d2_i, 0.0))
        time_dark = t_now - trk.last_fix_t
        rng = float(np.hypot(*(z - trk.last_fix_xy)))
        # A non-positive dark interval means the caller handed us a fix at or
        # after t_now -- speed is undefined, not infinite, so it is reported as
        # 0.0 rather than dividing.
        v = rng / time_dark if time_dark > 0.0 else 0.0

        if accepted[i]:
            reason = ""
        else:
            reason = (f"gated out: d2={d2_i:.2f} > {gate_chi2:.2f} "
                      f"({sig:.1f} sigma)")

        out.append(ReassocCandidate(
            track_id=trk.id,
            p=float(p[i]),
            d2=d2_i,
            sigma_distance=sig,
            likelihood=float(L[i]),
            time_dark_s=float(time_dark),
            range_m=rng,
            implied_speed_mps=float(v),
            implied_speed_kn=float(v * MPS_TO_KN),
            gated_out=not bool(accepted[i]),
            reason=reason,
        ))

    # Every rejected candidate ties at p = 0.0, so the d2 tie-break is what makes
    # the rejected tail read "nearest miss first" and the order reproducible.
    out.sort(key=lambda c: (-c.p, c.d2, c.track_id))
    return out

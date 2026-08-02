"""Constant-velocity Kalman filter kernel for maritime multi-target tracking.

Model
-----
State is ``[x, y, vx, vy]`` in metres and metres/second (both positions first,
then both velocities). Motion is the discrete white-noise-acceleration (DWNA)
constant-velocity model: velocity is a random walk driven by an acceleration
noise of intensity ``q`` (m^2/s^3), and position integrates that velocity.
Only position is observed, so ``H`` is a 2x4 selection matrix.

Measurement noise asymmetry
---------------------------
``R`` is a *per-call* argument of :meth:`CVKalman.update`, never a constructor
field, because a single track is fused from sensors with wildly different
accuracy: AIS reports carry ``R = diag(25, 25) m^2`` (5 m 1-sigma) while radar
plots carry ``R = diag(2500, 2500) m^2`` (50 m 1-sigma) -- a 100x variance
ratio. Caching ``R`` anywhere would silently apply one sensor's confidence to
the other's measurement, so nothing here stores it.

Numerics / performance
----------------------
The tracker runs at 60x replay speed, so ``update`` must stay well under
100 microseconds. Three things make that hold without sacrificing stability:
Joseph-form covariance update (stays symmetric and positive semi-definite
across the millions of cycles a long replay accumulates, where the shorthand
``(I - KH)P`` form drifts negative), a closed-form 2x2 inverse of the
innovation covariance instead of ``np.linalg.inv``, and slicing rather than
matrix-multiplying by the constant selection matrix ``H``.

This module knows nothing about vessels, sensors, MMSI or scenarios -- it takes
raw ``(z, R, dt)`` only.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["CVKalman", "F_matrix", "Q_matrix", "H"]

# Observation model: position-only. Kept as a module constant for tests and for
# documentation value; the hot path slices instead of multiplying by it.
H: np.ndarray = np.array([[1.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0]], dtype=float)

# Preallocated once: rebuilding a 4x4 identity per update is pure overhead.
_I4: np.ndarray = np.eye(4, dtype=float)

# Replay feeds long runs of repeated dt values, so memoising F and Q on
# (dt, q) removes most of the per-cycle matrix construction cost.
_F_CACHE: dict[float, np.ndarray] = {}
_Q_CACHE: dict[Tuple[float, float], np.ndarray] = {}
_CACHE_LIMIT = 4096  # irregular float dt would otherwise grow these forever


def _f_cached(dt: float) -> np.ndarray:
    """Cached (shared, do-not-mutate) state transition matrix."""
    F = _F_CACHE.get(dt)
    if F is None:
        if len(_F_CACHE) > _CACHE_LIMIT:
            _F_CACHE.clear()
        F = np.array([[1.0, 0.0, dt, 0.0],
                      [0.0, 1.0, 0.0, dt],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]], dtype=float)
        _F_CACHE[dt] = F
    return F


def _q_cached(dt: float, q: float) -> np.ndarray:
    """Cached (shared, do-not-mutate) process noise matrix."""
    key = (dt, q)
    Q = _Q_CACHE.get(key)
    if Q is None:
        if len(_Q_CACHE) > _CACHE_LIMIT:
            _Q_CACHE.clear()
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        a = q * dt4 / 4.0
        b = q * dt3 / 2.0
        c = q * dt2
        Q = np.array([[a, 0.0, b, 0.0],
                      [0.0, a, 0.0, b],
                      [b, 0.0, c, 0.0],
                      [0.0, b, 0.0, c]], dtype=float)
        _Q_CACHE[key] = Q
    return Q


def F_matrix(dt: float) -> np.ndarray:
    """Constant-velocity state transition matrix for a step of ``dt`` seconds."""
    # Copy: callers own the result, and a mutation of a cached array would
    # silently corrupt every subsequent predict().
    return _f_cached(float(dt)).copy()


def Q_matrix(dt: float, q: float) -> np.ndarray:
    """DWNA process noise covariance for step ``dt`` and intensity ``q``."""
    return _q_cached(float(dt), float(q)).copy()


class CVKalman:
    """Constant-velocity Kalman filter over ``[x, y, vx, vy]``.

    Parameters
    ----------
    q:
        Acceleration noise intensity in m^2/s^3. The default of 0.5 suits
        merchant-vessel manoeuvring at typical AIS/radar revisit rates.
    """

    __slots__ = ("_x", "_P", "q", "t")

    def __init__(self, q: float = 0.5) -> None:
        self.q: float = float(q)
        self._x: np.ndarray = np.zeros(4, dtype=float)
        self._P: np.ndarray = np.eye(4, dtype=float)
        # `t` is advanced by predict(); update() takes no timestamp, so the
        # caller drives time exclusively through predict(dt).
        self.t: float = 0.0

    @staticmethod
    def init_two_point(z1, t1: float, z2, t2: float, R, q: float = 0.5) -> "CVKalman":
        """Initialise from two position fixes by two-point differencing.

        The resulting position and velocity estimates share the noise of ``z2``,
        so the initial covariance is built with the correct cross-terms rather
        than a diagonal guess: an over-optimistic off-diagonal-free P makes the
        first few updates over-trust the seed velocity and diverge under gating.
        """
        z1 = np.asarray(z1, dtype=float).reshape(2)
        z2 = np.asarray(z2, dtype=float).reshape(2)
        R = np.asarray(R, dtype=float).reshape(2, 2)

        dt = float(t2) - float(t1)
        if dt <= 0.0:
            raise ValueError(
                f"init_two_point requires t2 > t1, got t1={t1!r}, t2={t2!r} (dt={dt!r})"
            )

        kf = CVKalman(q)
        kf._x[:2] = z2
        kf._x[2:] = (z2 - z1) / dt

        # State orders both positions then both velocities, so the position
        # block is [0:2] and the velocity block is [2:4]:
        #   cov(pos) = R,  cov(vel) = 2R/dt^2,  cov(pos, vel) = R/dt
        P = np.empty((4, 4), dtype=float)
        P[0:2, 0:2] = R
        P[0:2, 2:4] = R / dt
        P[2:4, 0:2] = R / dt
        P[2:4, 2:4] = 2.0 * R / (dt * dt)
        kf._P = P

        kf.t = float(t2)
        return kf

    def predict(self, dt: float) -> None:
        """Propagate state and covariance forward by ``dt`` seconds.

        ``dt == 0`` is legitimate (simultaneous contacts at replay speed) and
        reduces to F = I, Q = 0.
        """
        dt = float(dt)
        F = _f_cached(dt)
        Q = _q_cached(dt, self.q)
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q
        self.t += dt

    def innovation(self, z, R) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(nu, S)`` -- the innovation and its full 2x2 covariance.

        ``S = H P H^T + R`` including the sensor's measurement noise, which is
        what Mahalanobis gating needs: ``d^2 = nu^T S^-1 nu``. Omitting R would
        shrink S by the sensor variance -- a factor of several against radar's
        diag(2500, 2500) once P has converged -- inflating every d^2 and
        rejecting correct associations.
        """
        z = np.asarray(z, dtype=float).reshape(2)
        R = np.asarray(R, dtype=float).reshape(2, 2)
        # H is a selection matrix: H@x == x[:2] and H@P@H.T == P[:2,:2].
        nu = z - self._x[:2]
        S = self._P[:2, :2] + R
        return nu, S

    def update(self, z, R) -> None:
        """Fuse one position measurement ``z`` with per-sensor noise ``R``."""
        z = np.asarray(z, dtype=float).reshape(2)
        R = np.asarray(R, dtype=float).reshape(2, 2)

        x = self._x
        P = self._P

        # H is a selection matrix, so every product with it is a slice:
        # H@x -> x[:2], H@P@H.T -> P[:2,:2], P@H.T -> P[:,:2].
        nu = z - x[:2]
        S = P[:2, :2] + R
        PHt = P[:, :2]

        # Closed-form 2x2 inverse; np.linalg.inv would dominate the cycle cost.
        s00 = S[0, 0]
        s01 = S[0, 1]
        s10 = S[1, 0]
        s11 = S[1, 1]
        det = s00 * s11 - s01 * s10
        # Relative test: det has units m^4 and its scale differs ~10^4 between
        # AIS and radar R, so an absolute epsilon would be wrong for one of them.
        scale = abs(s00 * s11) + abs(s01 * s10)
        if not np.isfinite(det) or abs(det) <= 1e-12 * max(scale, 1.0):
            raise np.linalg.LinAlgError(
                f"innovation covariance S is singular or non-finite (det={det!r}); "
                "check that R is positive definite"
            )

        inv_det = 1.0 / det
        Sinv = np.array([[s11 * inv_det, -s01 * inv_det],
                         [-s10 * inv_det, s00 * inv_det]], dtype=float)

        K = PHt @ Sinv                      # (4,2)
        self._x = x + K @ nu

        # Joseph form: algebraically equal to (I - KH)P but numerically keeps P
        # symmetric and PSD over the millions of cycles a 60x replay accumulates.
        # (I - K@H) is the identity with K written into its first two columns.
        A = _I4.copy()
        A[:, :2] -= K
        self._P = A @ P @ A.T + K @ R @ K.T

    @property
    def state(self) -> np.ndarray:
        """Current state estimate ``[x, y, vx, vy]``, shape (4,)."""
        return self._x

    @property
    def cov(self) -> np.ndarray:
        """Current state covariance, shape (4, 4)."""
        return self._P

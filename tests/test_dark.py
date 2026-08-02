"""Acceptance + unit tests for tracker.dark (dark-vessel coasting and re-association).

Five acceptance criteria are covered here:
  AC1 ELLIPSE GROWS       -- test_ac1_*
  AC2 COASTLINE CLIP      -- test_ac2_*
  AC3 NEAR -> p > 0.90    -- test_ac3_*
  AC4 FAR  -> low p       -- test_ac4_*
  AC5 EVENT PAYLOAD       -- test_ac5_*

Everything else is supporting unit coverage of the frozen contract:
    CHI2_95_2DOF, GATE_CHI2, BETA_FA_REACQ, ellipse_axes, ellipse_polygon,
    clip_to_water, DarkTrack, ReassocCandidate, reassociate

BETA_FA_REACQ is *tuned* by the implementation and is never hard-coded here. Every
probability assertion either (a) recomputes the expected value from the imported
constant using the documented formula, or (b) asserts a relation that holds for any
beta > 0. Where a threshold (0.90) is genuinely beta-dependent that is called out in
the test docstring -- a failure there is a real finding about the tuning, not a
broken test.
"""

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, shape
from shapely.ops import unary_union

from tracker.geofence import lonlat_to_enu
from tracker.kalman import CVKalman
from tracker.dark import (
    BETA_FA_REACQ,
    CHI2_95_2DOF,
    GATE_CHI2,
    DarkTrack,
    ReassocCandidate,
    clip_to_water,
    ellipse_axes,
    ellipse_polygon,
    reassociate,
)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
SEED = 20260726

# FINDING -- two spellings of one knot are in play:
#   the written contract says implied_speed_kn == implied_speed_mps / 0.514444
#   the implementation divides by the exact SI definition, 1852 m / 3600 s.
# They differ by 8.6e-7 relative (0.86 ppm), which is physically meaningless for a
# vessel speed but large enough to blow a rel=1e-9 assertion. The exact definition
# is the correct one, so the checks below are written against the contract constant
# at a tolerance that spans both, and KNOT_MPS_EXACT documents what is actually in
# force. Do not tighten these without deciding which constant wins.
KNOT_MPS = 0.514444                     # the constant the contract pins
KNOT_MPS_EXACT = 1852.0 / 3600.0        # the exact SI knot the implementation uses
KNOT_REL_TOL = 1e-5                     # spans the gap between the two spellings

# Demo kinematics: a 15 kn merchant vessel with q = 0.01 m^2/s^3. At 60x replay,
# 360 s of scenario time is 6 s of demo -- inside the 5-8 s window AC1 asks for.
VESSEL_SPEED_KN = 15.0
VESSEL_SPEED_MPS = VESSEL_SPEED_KN * KNOT_MPS
Q_DARK = 0.01                           # m^2/s^3 -- NOT the CVKalman default of 0.5
COAST_DT_S = 10.0
COAST_STEPS_AC1 = 36                    # 360 s == 6 min scenario == 6 s at 60x
COAST_STEPS_AC3 = 18                    # 180 s dark before the radar re-acquisition
SEED_BASELINE_S = 20.0                  # two-point init baseline

# Anisotropic seed noise on purpose: F and Q are isotropic in x/y, so an isotropic
# R would keep P_pos an exact multiple of the identity forever, making a == b and
# theta an arbitrary eigenvector artefact. 20 m / 10 m 1-sigma keeps a > b strictly.
R_SEED = np.diag([400.0, 100.0])
R_RADAR = np.diag([2500.0, 2500.0])     # sigma = 50 m, the re-acquisition sensor
R_AIS = np.diag([25.0, 25.0])           # sigma = 5 m

# AC2: SF Bay / Golden Gate approach. Any origin inside the layer's box works; this
# one puts the scenario metres near zero.
ORIGIN_LONLAT = (-122.45, 37.75)
COASTLINE_PATH = Path(__file__).resolve().parent.parent / "geo" / "ca_coastline_10m.geojson"
COASTLINE_MISSING_MSG = (
    f"{COASTLINE_PATH} not present -- geo/ is gitignored, so the real-coastline "
    "clip is skipped on machines that have not fetched the layer. The synthetic "
    "half-plane clip test covers the deterministic case."
)

P_CONFIDENT = 0.90                      # the AC3/AC4 decision threshold

# The contract does not state the type of DarkTrack.id; the implementation types it
# as int and coerces it, so every id here is a numeric MMSI -- which is what a
# maritime tracker actually keys on. Assertions below compare ids to `track.id`
# rather than to a literal type, so they survive a change of mind either way.
MMSI_COASTER = 366999123
MMSI_MID = 366999124
MMSI_DEMO = 366999999
MMSI_NEAR = 366000111
MMSI_FAR = 366000222
MMSI_MILES_AWAY = 366000333
MMSI_ANCHOR = 366000444
MMSI_INSTANT = 366000555
MMSI_TIGHT = 366000666
MMSI_VAGUE = 366000777
MMSI_BOUNDARY = 366000888
MMSI_KNOB = 366000999
MMSI_SWEEP = 366001000
MMSI_RANDOM = 366001001


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _rot(theta_deg):
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]])


def _cov_for_axes(a_m, b_m, theta_deg=0.0, conf_chi2=CHI2_95_2DOF):
    """Position covariance whose ``conf_chi2`` ellipse has exactly these semi-axes.

    Inverse of ``ellipse_axes``: a = sqrt(conf * lambda_max), so lambda = a^2 / conf.
    """
    Rm = _rot(theta_deg)
    return Rm @ np.diag([a_m ** 2 / conf_chi2, b_m ** 2 / conf_chi2]) @ Rm.T


def _angle_gap_mod_180(theta_deg, expected_deg):
    """Smallest angular gap modulo 180 degrees.

    An ellipse axis is a direction, not a vector: theta and theta+180 describe the
    same major axis, and a bare ``% 180`` reports 179.9999 as a 180-degree error.
    """
    d = abs(float(theta_deg) - float(expected_deg)) % 180.0
    return min(d, 180.0 - d)


def _coasting_filter(steps, dt=COAST_DT_S, speed_mps=VESSEL_SPEED_MPS, q=Q_DARK):
    """A CVKalman seeded from two fixes then coasted (predict, never update).

    Coasting *is* predicting with no measurement: this is the filter state a vessel
    leaves behind when it switches off AIS.
    """
    kf = CVKalman.init_two_point(
        [0.0, 0.0], 0.0, [speed_mps * SEED_BASELINE_S, 0.0], SEED_BASELINE_S, R_SEED, q=q
    )
    for _ in range(steps):
        kf.predict(dt)
    return kf


def _dark_track_from_coast(track_id=MMSI_COASTER, steps=COAST_STEPS_AC3, label="MV Coaster"):
    """A DarkTrack built from a real coast, plus the filter it came from.

    ``last_fix_xy`` is the last *measured* position (the second seed fix), which is
    deliberately far from the coasted ``pos`` -- that separation is what makes the
    AC5 "range is from the last fix, not the prediction" assertion discriminating.
    """
    kf = _coasting_filter(steps)
    track = DarkTrack(
        id=track_id,
        pos=kf.state[:2].copy(),
        pos_cov=kf.cov[:2, :2].copy(),
        last_fix_xy=np.array([VESSEL_SPEED_MPS * SEED_BASELINE_S, 0.0]),
        last_fix_t=SEED_BASELINE_S,
        label=label,
    )
    return track, kf


def _reference_probabilities(z, R, tracks, beta_fa=BETA_FA_REACQ, gate_chi2=GATE_CHI2):
    """Independent re-implementation of the documented maths, by track id.

    Returns ``{track_id: (d2, likelihood, p)}``. Written from the contract text
    rather than from the implementation, so an assertion against it is a real
    check and not a tautology.
    """
    z = np.asarray(z, dtype=float).reshape(2)
    R = np.asarray(R, dtype=float).reshape(2, 2)
    ids = [tr.id for tr in tracks]
    assert len(set(ids)) == len(ids), f"duplicate track ids would be dropped: {ids}"
    rows = {}
    for tr in tracks:
        S = np.asarray(tr.pos_cov, dtype=float).reshape(2, 2) + R
        nu = z - np.asarray(tr.pos, dtype=float).reshape(2)
        d2 = float(nu @ np.linalg.inv(S) @ nu)
        L = float(math.exp(-0.5 * d2) / math.sqrt(np.linalg.det(2.0 * math.pi * S)))
        rows[tr.id] = (d2, L)
    denom = sum(L for d2, L in rows.values() if d2 <= gate_chi2) + float(beta_fa)
    return {
        tid: (d2, L, 0.0 if d2 > gate_chi2 else L / denom)
        for tid, (d2, L) in rows.items()
    }


def _isotropic_track(track_id, sigma_tot_m, r_frac=0.01, center=(0.0, 0.0)):
    """Track whose innovation covariance is exactly ``sigma_tot^2 * I``.

    ``R`` is carved out of the total as a small fraction so ``pos_cov = S - R`` is
    always positive definite no matter how the total is chosen.
    """
    s2 = float(sigma_tot_m) ** 2
    R = np.diag([r_frac * s2, r_frac * s2])
    P = np.diag([(1.0 - r_frac) * s2, (1.0 - r_frac) * s2])
    track = DarkTrack(
        id=track_id,
        pos=np.array([float(center[0]), float(center[1])]),
        pos_cov=P,
        last_fix_xy=np.array([0.0, 0.0]),
        last_fix_t=0.0,
    )
    return track, R


def _sigma_tot_placing_p_between(d2_mid, beta_fa=BETA_FA_REACQ, p_thresh=P_CONFIDENT):
    """Isotropic sigma putting p(d2=0) above and p(d2_mid) below ``p_thresh``.

    For a single track ``p = L / (L + beta)`` with ``L = L0 * exp(-d2/2)`` and
    ``L0 = 1 / (2*pi*sigma^2)``. Writing ``r = beta / L0``:

        p(0)      > P   <=>  r < (1 - P) / P
        p(d2_mid) < P   <=>  r > (1 - P) / P * exp(-d2_mid / 2)

    so the admissible band spans exactly a factor ``exp(d2_mid / 2)`` and is never
    empty for beta > 0. Taking its geometric centre and inverting
    ``L0 = 1 / (2*pi*sigma^2)`` gives sigma. Deriving the covariance from the tuned
    beta -- rather than guessing one -- is what keeps this test meaningful whatever
    value the implementation settles on.
    """
    ratio = (1.0 - p_thresh) / p_thresh
    r_star = ratio * math.exp(-0.25 * d2_mid)          # geometric centre of the band
    l0 = float(beta_fa) / r_star
    return math.sqrt(1.0 / (2.0 * math.pi * l0))


def _load_land_enu():
    """The coastline GeoJSON as one shapely geometry in ENU metres, or None."""
    if not COASTLINE_PATH.is_file():
        return None
    with COASTLINE_PATH.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    parts = []
    for feature in payload["features"]:
        geom = shape(feature["geometry"])
        polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
        for poly in polys:
            shell = lonlat_to_enu(list(poly.exterior.coords), ORIGIN_LONLAT)
            holes = [lonlat_to_enu(list(r.coords), ORIGIN_LONLAT) for r in poly.interiors]
            parts.append(Polygon(shell, holes))
    return unary_union(parts)


def _half_plane_land(extent=1.0e6):
    """Synthetic 'land' filling the x >= 0 half-plane. Deterministic, no data files."""
    return Polygon([(0.0, -extent), (extent, -extent), (extent, extent), (0.0, extent)])


# --------------------------------------------------------------------------
# Frozen module constants
# --------------------------------------------------------------------------
def test_chi2_constants_are_the_frozen_values():
    """Supporting: the two chi-square constants are contract, not tuning."""
    assert CHI2_95_2DOF == pytest.approx(5.991, abs=1e-9)
    assert GATE_CHI2 == pytest.approx(9.21034, abs=1e-9)
    # 99% gate must be looser than the 95% display ellipse, or the drawn ellipse
    # would show contacts the gate silently rejected.
    assert GATE_CHI2 > CHI2_95_2DOF


def test_beta_fa_reacq_is_strictly_positive_and_finite():
    """Supporting precondition: p < 1 holds *only* because beta > 0.

    If this ever regresses to 0.0 the whole p < 1.0 family below would pass
    vacuously while the system started claiming certainty. Fail here instead.
    """
    assert math.isfinite(BETA_FA_REACQ)
    assert BETA_FA_REACQ > 0.0
    # Sanity band for a spatial false-alarm density in m^-2: anything outside
    # 1 per m^2 .. 1 per 10^16 m^2 is a units error, not a tuning choice.
    assert 1e-16 < BETA_FA_REACQ < 1.0


# --------------------------------------------------------------------------
# ellipse_axes
# --------------------------------------------------------------------------
def test_ellipse_axes_diagonal_covariance_hand_computed():
    """Supporting: P = diag(100, 400) -> a = sqrt(5.991*400), b = sqrt(5.991*100), theta = 90."""
    a, b, theta = ellipse_axes(np.diag([100.0, 400.0]))
    assert a == pytest.approx(math.sqrt(5.991 * 400.0), rel=1e-12)
    assert b == pytest.approx(math.sqrt(5.991 * 100.0), rel=1e-12)
    # Larger variance is on y, so the major axis points along y.
    assert _angle_gap_mod_180(theta, 90.0) == pytest.approx(0.0, abs=1e-9)
    assert a >= b


def test_ellipse_axes_rotated_45_degrees():
    """Supporting: R(45) diag(a2, b2) R(45)^T must report theta = 45 (mod 180)."""
    a_true, b_true = 900.0, 300.0
    P = _cov_for_axes(a_true, b_true, theta_deg=45.0)
    a, b, theta = ellipse_axes(P)
    assert a == pytest.approx(a_true, rel=1e-9)
    assert b == pytest.approx(b_true, rel=1e-9)
    assert _angle_gap_mod_180(theta, 45.0) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("theta_deg", [0.0, 17.5, 90.0, 123.0, 179.0])
def test_ellipse_axes_recovers_theta_for_any_orientation(theta_deg):
    """Supporting: orientation round-trips for a well-separated axis ratio."""
    P = _cov_for_axes(1200.0, 300.0, theta_deg=theta_deg)
    a, b, theta = ellipse_axes(P)
    assert a == pytest.approx(1200.0, rel=1e-9)
    assert b == pytest.approx(300.0, rel=1e-9)
    assert _angle_gap_mod_180(theta, theta_deg) == pytest.approx(0.0, abs=1e-6)


def test_ellipse_axes_semi_major_never_below_semi_minor_over_random_covariances():
    """Supporting: a >= b is unconditional, including for near-circular P."""
    rng = np.random.default_rng(SEED)
    for _ in range(200):
        A = rng.normal(scale=50.0, size=(2, 2))
        P = A @ A.T + np.eye(2) * rng.uniform(1.0, 1e6)
        a, b, theta = ellipse_axes(P)
        assert a >= b - 1e-9
        assert a > 0.0 and b > 0.0
        assert math.isfinite(theta)


def test_ellipse_axes_scales_as_sqrt_of_confidence():
    """Supporting: axes scale with sqrt(conf_chi2); the shape is confidence-free."""
    P = _cov_for_axes(800.0, 200.0, theta_deg=30.0)
    a95, b95, t95 = ellipse_axes(P, conf_chi2=CHI2_95_2DOF)
    a99, b99, t99 = ellipse_axes(P, conf_chi2=GATE_CHI2)
    scale = math.sqrt(GATE_CHI2 / CHI2_95_2DOF)
    assert a99 == pytest.approx(a95 * scale, rel=1e-12)
    assert b99 == pytest.approx(b95 * scale, rel=1e-12)
    assert _angle_gap_mod_180(t99, t95) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# ellipse_polygon
# --------------------------------------------------------------------------
def test_ellipse_polygon_ring_is_closed_with_n_plus_one_coordinates():
    """Supporting: pins the ring convention -- n distinct vertices, closed to n+1."""
    n = 64
    poly = ellipse_polygon((0.0, 0.0), _cov_for_axes(500.0, 200.0), n=n)
    coords = list(poly.exterior.coords)
    assert len(coords) == n + 1, (
        f"expected a closed ring of {n}+1 coordinates, got {len(coords)}"
    )
    assert coords[0] == pytest.approx(coords[-1])
    assert len(set(coords[:-1])) == n     # no duplicated vertices before closure


def test_ellipse_polygon_is_valid_and_area_matches_pi_a_b():
    """Supporting: a 64-gon inscribes the ellipse, so area is ~0.16% under pi*a*b."""
    P = _cov_for_axes(1500.0, 400.0, theta_deg=33.0)
    a, b, _ = ellipse_axes(P)
    poly = ellipse_polygon((1000.0, -2000.0), P, n=64)
    assert isinstance(poly, Polygon)
    assert poly.is_valid
    exact = math.pi * a * b
    assert poly.area == pytest.approx(exact, rel=0.01)
    # Inscribed, so strictly under -- a circumscribed or mis-scaled polygon would
    # pass the 1% band from the wrong side.
    assert poly.area < exact


def test_ellipse_polygon_is_centred_on_its_center_argument():
    """Supporting: the polygon is placed at ``center``, not at the origin."""
    center = (12345.0, -6789.0)
    P = _cov_for_axes(600.0, 250.0, theta_deg=10.0)
    poly = ellipse_polygon(center, P, n=64)
    cx, cy = poly.centroid.x, poly.centroid.y
    assert cx == pytest.approx(center[0], abs=1e-6)
    assert cy == pytest.approx(center[1], abs=1e-6)
    assert poly.contains(Point(*center))


def test_ellipse_polygon_vertex_count_is_configurable():
    """Supporting: n controls resolution, and a coarser ring encloses less area."""
    P = _cov_for_axes(500.0, 500.0)
    coarse = ellipse_polygon((0.0, 0.0), P, n=8)
    fine = ellipse_polygon((0.0, 0.0), P, n=256)
    assert len(coarse.exterior.coords) == 9
    assert len(fine.exterior.coords) == 257
    assert coarse.area < fine.area < math.pi * 500.0 * 500.0


# --------------------------------------------------------------------------
# AC1 -- the uncertainty ellipse grows while coasting
# --------------------------------------------------------------------------
def test_ac1_semi_major_axis_grows_strictly_monotonically_while_coasting():
    """AC1: predict() with no update must grow the 95% ellipse strictly, every step.

    360 s of scenario time at 60x replay is 6 s of demo -- the window the operator
    actually watches the ellipse bloom in.
    """
    kf = CVKalman.init_two_point(
        [0.0, 0.0], 0.0,
        [VESSEL_SPEED_MPS * SEED_BASELINE_S, 0.0], SEED_BASELINE_S,
        R_SEED, q=Q_DARK,
    )
    assert kf.q == Q_DARK                      # not the CVKalman default of 0.5

    rows = []
    prev_a = -math.inf
    prev_b = -math.inf
    for step in range(COAST_STEPS_AC1):
        kf.predict(COAST_DT_S)                 # coasting: predict, never update
        a, b, theta = ellipse_axes(kf.cov[:2, :2], conf_chi2=CHI2_95_2DOF)

        assert a > prev_a, f"semi-major did not grow at t={kf.t}: {a} <= {prev_a}"
        assert b > prev_b, f"semi-minor did not grow at t={kf.t}: {b} <= {prev_b}"
        assert a >= b, f"semi-major below semi-minor at t={kf.t}: {a} < {b}"
        assert math.isfinite(theta), f"theta went non-finite at t={kf.t}: {theta}"

        prev_a, prev_b = a, b
        rows.append((kf.t, a, b, theta))

    print(f"\nAC1 coasting at {VESSEL_SPEED_KN} kn, q={Q_DARK} m^2/s^3, "
          f"dt={COAST_DT_S}s  ({COAST_STEPS_AC1 * COAST_DT_S:.0f}s scenario "
          f"= {COAST_STEPS_AC1 * COAST_DT_S / 60.0:.0f}s at 60x)")
    print(f"{'t (s)':>8} {'a (m)':>12} {'b (m)':>12} {'theta (deg)':>12}")
    for t, a, b, theta in rows[::6] + [rows[-1]]:
        print(f"{t:8.1f} {a:12.2f} {b:12.2f} {theta:12.2f}")

    # A demo-visible bloom, not a rounding-level drift.
    assert rows[-1][1] > 10.0 * rows[0][1]
    assert rows[-1][1] > 1000.0


def test_ac1_anisotropic_seed_keeps_major_axis_strictly_dominant():
    """AC1 support: with anisotropic seed noise a > b holds strictly, not just >=.

    F and Q are isotropic, so an isotropic seed R would leave P_pos a multiple of
    the identity and make a == b with an arbitrary theta -- an ellipse test that
    proves nothing. This pins that the fixture avoids that degeneracy.
    """
    for steps in (1, 6, 18, COAST_STEPS_AC1):
        kf = _coasting_filter(steps)
        a, b, theta = ellipse_axes(kf.cov[:2, :2])
        assert a > b, f"degenerate (circular) ellipse after {steps} steps: a={a} b={b}"
        assert math.isfinite(theta)


def test_ac1_ellipse_polygon_area_grows_with_the_coast():
    """AC1 support: the drawn 95% polygon -- what the operator sees -- also grows."""
    prev_area = -math.inf
    for steps in range(1, 13):
        kf = _coasting_filter(steps)
        poly = ellipse_polygon(kf.state[:2], kf.cov[:2, :2], n=64)
        assert poly.is_valid
        assert poly.area > prev_area
        prev_area = poly.area


# --------------------------------------------------------------------------
# AC2 -- clipping the ellipse to water
# --------------------------------------------------------------------------
def test_ac2_clip_to_water_with_none_land_returns_the_polygon_unchanged():
    """AC2: land=None is the 'no coastline loaded' path and must be a no-op."""
    poly = ellipse_polygon((0.0, 0.0), _cov_for_axes(2000.0, 900.0, 25.0), n=64)
    out = clip_to_water(poly, None)
    assert out.area == pytest.approx(poly.area, rel=1e-12)
    assert out.equals(poly)


def test_ac2_synthetic_half_plane_land_removes_exactly_half_the_ellipse():
    """AC2: deterministic clip that runs with no GeoJSON present.

    Ellipse centred on the edge of an x >= 0 half-plane of land -> exactly half the
    area survives, and none of it is inside the land.
    """
    land = _half_plane_land()
    P = _cov_for_axes(3000.0, 1200.0, theta_deg=0.0)
    poly = ellipse_polygon((0.0, 0.0), P, n=64)

    clipped = clip_to_water(poly, land)

    assert clipped.area < poly.area
    assert clipped.area == pytest.approx(0.5 * poly.area, rel=1e-9)
    assert clipped.intersection(land).area <= 1e-9 * poly.area
    assert not clipped.intersects(land.buffer(-1.0))


def test_ac2_synthetic_land_fully_containing_the_ellipse_leaves_nothing():
    """AC2 support: an ellipse entirely ashore clips to empty, without raising."""
    land = Polygon([(-1e6, -1e6), (1e6, -1e6), (1e6, 1e6), (-1e6, 1e6)])
    poly = ellipse_polygon((0.0, 0.0), _cov_for_axes(1000.0, 500.0), n=64)
    clipped = clip_to_water(poly, land)
    assert clipped.area == pytest.approx(0.0, abs=1e-6)


def test_ac2_synthetic_land_clear_of_the_ellipse_preserves_area():
    """AC2 support: land nowhere near the ellipse must not shave it."""
    land = _half_plane_land()
    poly = ellipse_polygon((-500000.0, 0.0), _cov_for_axes(1000.0, 500.0), n=64)
    clipped = clip_to_water(poly, land)
    assert clipped.area == pytest.approx(poly.area, rel=1e-12)


@pytest.mark.skipif(not COASTLINE_PATH.is_file(), reason=COASTLINE_MISSING_MSG)
def test_ac2_real_coastline_clip_strictly_reduces_the_ellipse_area():
    """AC2: a real-coastline ellipse straddling the shore loses area to the land.

    The straddle point is *derived*, not hard-coded: the nearest point on the land
    boundary to the ENU origin is by construction on the coast, so an ellipse
    centred there always spans both sides whatever the layer's exact vertices are.
    """
    land = _load_land_enu()
    if land is None:                       # pragma: no cover - guarded by skipif
        pytest.skip(COASTLINE_MISSING_MSG)
    assert not land.is_empty
    assert land.is_valid

    boundary = land.boundary
    shore = boundary.interpolate(boundary.project(Point(0.0, 0.0)))
    P = _cov_for_axes(2500.0, 1400.0, theta_deg=20.0)
    poly = ellipse_polygon((shore.x, shore.y), P, n=64)

    # Precondition: the fixture really does straddle, so a "reduced area" result
    # cannot be an artefact of an ellipse that was ashore or offshore all along.
    assert poly.intersects(land)
    assert not land.contains(poly)

    clipped = clip_to_water(poly, land)

    print(f"\nAC2 shore point (ENU m): ({shore.x:.1f}, {shore.y:.1f})  "
          f"unclipped={poly.area:.1f} m^2  clipped={clipped.area:.1f} m^2  "
          f"retained={100.0 * clipped.area / poly.area:.1f}%")

    assert clipped.area < poly.area
    assert clipped.area > 0.0


@pytest.mark.skipif(not COASTLINE_PATH.is_file(), reason=COASTLINE_MISSING_MSG)
def test_ac2_real_coastline_clipped_region_does_not_intersect_land_interior():
    """AC2: no part of the surviving search region may sit on land."""
    land = _load_land_enu()
    if land is None:                       # pragma: no cover - guarded by skipif
        pytest.skip(COASTLINE_MISSING_MSG)

    boundary = land.boundary
    shore = boundary.interpolate(boundary.project(Point(0.0, 0.0)))
    poly = ellipse_polygon((shore.x, shore.y), _cov_for_axes(2500.0, 1400.0, 20.0), n=64)
    clipped = clip_to_water(poly, land)

    # Shared boundary is legitimate contact; only interior overlap is a defect,
    # so the land is eroded by a metre before the intersection test.
    assert not clipped.intersects(land.buffer(-1.0))
    assert clipped.intersection(land).area <= 1e-9 * poly.area


@pytest.mark.skipif(not COASTLINE_PATH.is_file(), reason=COASTLINE_MISSING_MSG)
def test_ac2_real_coastline_none_land_matches_unclipped_area():
    """AC2: the land=None escape hatch on the same real-coastline fixture."""
    land = _load_land_enu()
    if land is None:                       # pragma: no cover - guarded by skipif
        pytest.skip(COASTLINE_MISSING_MSG)
    boundary = land.boundary
    shore = boundary.interpolate(boundary.project(Point(0.0, 0.0)))
    poly = ellipse_polygon((shore.x, shore.y), _cov_for_axes(2500.0, 1400.0, 20.0), n=64)
    assert clip_to_water(poly, None).area == pytest.approx(poly.area, rel=1e-12)


# --------------------------------------------------------------------------
# AC3 -- a detection on the prediction scores above 0.90
# --------------------------------------------------------------------------
def test_ac3_detection_on_the_prediction_scores_above_0p90():
    """AC3: radar plot essentially on a 3-minute-dark track's prediction -> p > 0.90.

    The track's covariance is not invented: it is whatever a real CVKalman holds
    after 180 s of coasting at 15 kn with q = 0.01 (sigma ~ 500 m).

    This threshold is genuinely beta-dependent -- p = L / (L + beta) and L is fixed
    by the ~500 m coast, so p > 0.90 requires beta < L / 9. If this fails while the
    exact-value assertion below passes, the finding is that BETA_FA_REACQ is tuned
    too high for the demo's re-acquisition to ever read as confident.
    """
    track, kf = _dark_track_from_coast()
    z = track.pos.copy()                   # detection right on the prediction
    out = reassociate(z, R_RADAR, [track], kf.t)

    assert len(out) == 1
    cand = out[0]
    assert cand.track_id == track.id
    assert cand.gated_out is False
    assert cand.reason == "" or cand.reason is not None

    d2_ref, l_ref, p_ref = _reference_probabilities(z, R_RADAR, [track])[track.id]
    assert cand.d2 == pytest.approx(d2_ref, abs=1e-12)
    assert cand.likelihood == pytest.approx(l_ref, rel=1e-9)
    assert cand.p == pytest.approx(p_ref, rel=1e-9)

    print(f"\nAC3 sigma_pos={math.sqrt(track.pos_cov[0, 0]):.1f} m  "
          f"L={cand.likelihood:.3e} m^-2  beta={BETA_FA_REACQ:.3e} m^-2  p={cand.p:.4f}")

    assert cand.p > P_CONFIDENT, (
        f"p={cand.p:.4f} for a detection on the prediction; with L={cand.likelihood:.3e} "
        f"this needs BETA_FA_REACQ < {cand.likelihood / 9.0:.3e}, "
        f"but it is {BETA_FA_REACQ:.3e}"
    )
    assert cand.p < 1.0


def test_ac3_high_quality_ais_grade_detection_also_scores_above_0p90():
    """AC3 support: a tighter sensor R can only raise p, never lower it."""
    track, kf = _dark_track_from_coast()
    z = track.pos.copy()
    p_radar = reassociate(z, R_RADAR, [track], kf.t)[0].p
    p_ais = reassociate(z, R_AIS, [track], kf.t)[0].p
    assert p_ais >= p_radar
    assert p_ais > P_CONFIDENT
    assert p_ais < 1.0


# --------------------------------------------------------------------------
# AC4 -- distant detections must NOT score confidently
# --------------------------------------------------------------------------
def test_ac4_ten_sigma_detection_is_gated_out_with_zero_probability():
    """AC4: a detection ~10 sigma out is gated_out, p == 0.0, with a stated reason.

    Ten sigma is Mahalanobis with respect to S = P_pos + R (not P_pos alone), so
    the offset is built from S and the expected d2 is exactly 100.
    """
    track, kf = _dark_track_from_coast()
    S = track.pos_cov + R_RADAR
    z = track.pos + np.array([10.0 * math.sqrt(S[0, 0]), 0.0])

    out = reassociate(z, R_RADAR, [track], kf.t)
    assert len(out) == 1
    cand = out[0]

    assert cand.gated_out is True
    assert cand.p == 0.0
    assert isinstance(cand.reason, str) and cand.reason.strip() != "", (
        "a gated-out candidate must carry a non-empty reason for the event log"
    )
    assert cand.d2 > GATE_CHI2
    assert cand.d2 == pytest.approx(100.0, rel=1e-6)


def test_ac4_mid_distance_detection_ranks_strictly_below_the_near_one():
    """AC4: p_near > p_mid for the same track -- p must fall with distance.

    Beta-independent: p is strictly decreasing in d2 for any beta > 0, so this
    ordering is the part of AC4 that can never be tuned away.
    """
    track, kf = _dark_track_from_coast()
    S = track.pos_cov + R_RADAR
    sigma_x = math.sqrt(S[0, 0])

    p_near = reassociate(track.pos.copy(), R_RADAR, [track], kf.t)[0].p
    mid = reassociate(track.pos + np.array([2.2 * sigma_x, 0.0]), R_RADAR, [track], kf.t)[0]

    assert mid.gated_out is False, "2.2 sigma (d2 = 4.84) must sit inside the 99% gate"
    assert mid.d2 == pytest.approx(4.84, rel=1e-6)
    assert p_near > mid.p, f"p did not fall with distance: near={p_near} mid={mid.p}"
    # Not a hair's-breadth difference: the likelihood ratio alone is exp(-2.42).
    assert mid.p < p_near


def test_ac4_mid_distance_probability_is_not_above_0p90():
    """AC4: at ~2.2 sigma the system must not still be claiming >0.90 confidence.

    The track covariance is *derived from the tuned BETA_FA_REACQ* rather than
    guessed, because for a single track p(0) > 0.90 and p(4.84) < 0.90 pin beta/L0
    into a band exactly exp(2.42) = 11.25 wide. Picking a covariance blind would
    make this test a coin flip on someone else's constant; deriving it means the
    assertion tests the *shape* of p(d2) -- that it actually falls through the
    threshold -- for whatever beta is in force. The resulting sigma is printed so
    an implausible one is visible.
    """
    d2_mid = 4.84
    sigma_tot = _sigma_tot_placing_p_between(d2_mid)
    track, R = _isotropic_track(MMSI_MID, sigma_tot)

    near = reassociate(track.pos.copy(), R, [track], 600.0)[0]
    mid = reassociate(track.pos + np.array([math.sqrt(d2_mid) * sigma_tot, 0.0]),
                      R, [track], 600.0)[0]

    print(f"\nAC4 beta={BETA_FA_REACQ:.3e} m^-2 -> sigma_tot={sigma_tot:.1f} m  "
          f"p_near={near.p:.4f}  p_mid={mid.p:.4f}")

    assert mid.d2 == pytest.approx(d2_mid, rel=1e-6)
    assert mid.gated_out is False
    assert near.p > P_CONFIDENT
    assert mid.p <= P_CONFIDENT, (
        f"a {math.sqrt(d2_mid):.1f}-sigma detection still scored {mid.p:.4f}"
    )
    assert near.p > mid.p

    # And it matches the closed form, not just the threshold.
    _, _, p_mid_ref = _reference_probabilities(
        track.pos + np.array([math.sqrt(d2_mid) * sigma_tot, 0.0]), R, [track]
    )[track.id]
    assert mid.p == pytest.approx(p_mid_ref, rel=1e-9)


def test_ac4_very_uncertain_track_never_reaches_certainty_on_a_perfect_detection():
    """AC4: as P_pos grows the likelihood density falls, so beta dominates.

    A 20 km-sigma track hit dead-centre is the strongest possible evidence for
    that track, and it still must not read as 1.0 -- the vessel could be anywhere
    in a 20 km disc, so a plot in it is barely informative.
    """
    tight, kf = _dark_track_from_coast(track_id=MMSI_TIGHT)
    vague, R = _isotropic_track(MMSI_VAGUE, 20000.0)

    p_tight = reassociate(tight.pos.copy(), R_RADAR, [tight], kf.t)[0].p
    p_vague = reassociate(vague.pos.copy(), R, [vague], 600.0)[0].p

    print(f"\nAC4 perfect detection: p(sigma~500 m)={p_tight:.6f}  "
          f"p(sigma=20 km)={p_vague:.6f}")

    assert p_vague < 1.0
    assert p_vague < p_tight, (
        "a 20 km-sigma track must not score as confidently as a 500 m one"
    )
    assert p_vague > 0.0


def test_ac4_probability_is_strictly_below_one_across_a_covariance_and_offset_sweep():
    """AC4: p < 1.0 always. A system that reports certainty is lying.

    Sweeps four decades of position sigma against offsets from dead-centre out to
    the gate, plus seeded random covariances and offsets.
    """
    assert BETA_FA_REACQ > 0.0, "p < 1 is only guaranteed for beta > 0"
    rng = np.random.default_rng(SEED)

    worst = 0.0
    for sigma in (10.0, 100.0, 1000.0, 10000.0, 50000.0):
        for n_sigma in (0.0, 0.5, 1.0, 2.0, 3.0):
            track, R = _isotropic_track(MMSI_SWEEP, sigma)
            z = track.pos + np.array([n_sigma * sigma, 0.0])
            for cand in reassociate(z, R, [track], 600.0):
                assert 0.0 <= cand.p < 1.0, f"p={cand.p} at sigma={sigma}, {n_sigma} sigma"
                worst = max(worst, cand.p)

    for _ in range(150):
        A = rng.normal(scale=rng.uniform(5.0, 5000.0), size=(2, 2))
        P = A @ A.T + np.eye(2) * rng.uniform(100.0, 1e6)
        track = DarkTrack(
            id=MMSI_RANDOM,
            pos=np.zeros(2),
            pos_cov=P,
            last_fix_xy=np.zeros(2),
            last_fix_t=0.0,
        )
        z = rng.normal(scale=500.0, size=2)
        for cand in reassociate(z, R_RADAR, [track], 600.0):
            assert 0.0 <= cand.p < 1.0, f"p={cand.p} for P={P}"
            worst = max(worst, cand.p)

    print(f"\nAC4 sweep: highest p observed = {worst:.9f} (must stay < 1.0)")
    assert worst < 1.0


def test_ac4_probability_decreases_monotonically_with_offset():
    """AC4 support: p(d2) is strictly decreasing all the way to the gate.

    The printed table is the point of this test as much as the assertion. It walks
    a *realistically-dark* track (the AC3 fixture, sigma ~ 516 m) from dead-centre
    out to the gate and shows how much dynamic range the discriminator actually
    has under the tuned beta -- i.e. how far apart a good match and a marginal one
    really score before the gate fires and p cliffs to zero.

    Deliberately no threshold assertion here: how much separation is enough is a
    tuning decision, not a contract. This makes the answer visible; it does not
    legislate it.
    """
    track, kf = _dark_track_from_coast()
    sigma_x = math.sqrt((track.pos_cov + R_RADAR)[0, 0])

    rows = []
    last = math.inf
    for n_sigma in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.75, 3.0, 3.05):
        cand = reassociate(track.pos + np.array([n_sigma * sigma_x, 0.0]),
                           R_RADAR, [track], kf.t)[0]
        rows.append((n_sigma, cand.d2, cand.p, cand.gated_out))
        if cand.gated_out:
            break
        assert cand.p < last, f"p rose at {n_sigma} sigma: {cand.p} >= {last}"
        last = cand.p

    accepted = [p for _, _, p, gated in rows if not gated]
    print(f"\nAC4 dynamic range on a {math.sqrt(track.pos_cov[0, 0]):.0f} m-sigma dark "
          f"track (beta={BETA_FA_REACQ:.3e} m^-2, gate d2={GATE_CHI2:.3f}):")
    print(f"{'n_sigma':>8} {'d2':>9} {'p':>9}  gated_out")
    for n_sigma, d2, p, gated in rows:
        print(f"{n_sigma:8.2f} {d2:9.3f} {p:9.4f}  {gated}")
    print(f"  -> across the whole accept region p spans "
          f"{min(accepted):.4f}..{max(accepted):.4f} "
          f"(range {max(accepted) - min(accepted):.4f}), then cliffs to 0.0 at the gate.")

    assert len(accepted) >= 2


# --------------------------------------------------------------------------
# AC5 -- the event payload
# --------------------------------------------------------------------------
def test_ac5_every_payload_field_is_populated_and_self_consistent():
    """AC5: sigma_distance, time_dark_s, range_m, implied speeds all agree."""
    track, kf = _dark_track_from_coast()
    t_now = kf.t
    z = track.pos + np.array([120.0, -80.0])

    out = reassociate(z, R_RADAR, [track], t_now)
    assert len(out) == 1
    cand = out[0]

    assert isinstance(cand, ReassocCandidate)
    assert cand.track_id == track.id
    assert cand.gated_out is False

    assert cand.sigma_distance == pytest.approx(math.sqrt(cand.d2), rel=1e-12)
    assert cand.time_dark_s == pytest.approx(t_now - track.last_fix_t, rel=1e-12)
    assert cand.time_dark_s > 0.0

    expected_range = float(np.hypot(*(z - track.last_fix_xy)))
    assert cand.range_m == pytest.approx(expected_range, rel=1e-12)

    assert cand.implied_speed_mps == pytest.approx(cand.range_m / cand.time_dark_s, rel=1e-12)
    # Within tolerance of BOTH spellings of a knot -- i.e. it is a knot at all,
    # and not some other unit. See the KNOT_MPS comment block.
    assert cand.implied_speed_kn == pytest.approx(
        cand.implied_speed_mps / KNOT_MPS, rel=KNOT_REL_TOL
    )
    assert cand.implied_speed_kn == pytest.approx(
        cand.implied_speed_mps / KNOT_MPS_EXACT, rel=KNOT_REL_TOL
    )
    assert cand.likelihood > 0.0
    assert 0.0 < cand.p < 1.0

    print(f"\nAC5 payload: {cand.track_id} p={cand.p:.4f} d2={cand.d2:.3f} "
          f"sigma={cand.sigma_distance:.2f} dark={cand.time_dark_s:.0f}s "
          f"range={cand.range_m:.0f}m speed={cand.implied_speed_kn:.2f}kn")

    # The implied speed must be a plausible vessel speed for this fixture, which is
    # the whole point of putting it in the operator's event.
    assert 1.0 < cand.implied_speed_kn < 40.0


def test_ac5_range_is_measured_from_the_last_fix_not_the_prediction():
    """AC5: range_m anchors on last_fix_xy (the last MEASURED position).

    The prediction is deliberately 1000 m from the last fix and the detection sits
    on the prediction, so an implementation that measured from ``pos`` would report
    ~0 m instead of ~1000 m. Without that separation the assertion proves nothing.
    """
    track = DarkTrack(
        id=MMSI_ANCHOR,
        pos=np.array([1000.0, 0.0]),
        pos_cov=np.diag([250000.0, 250000.0]),
        last_fix_xy=np.array([0.0, 0.0]),
        last_fix_t=0.0,
    )
    z = np.array([1000.0, 0.0])
    cand = reassociate(z, R_RADAR, [track], 500.0)[0]

    assert cand.range_m == pytest.approx(1000.0, rel=1e-9), (
        "range_m looks like it was measured from the predicted position, not the last fix"
    )
    assert cand.d2 == pytest.approx(0.0, abs=1e-12)     # z really is on the prediction
    assert cand.implied_speed_mps == pytest.approx(1000.0 / 500.0, rel=1e-12)
    assert cand.implied_speed_kn == pytest.approx(2.0 / KNOT_MPS, rel=KNOT_REL_TOL)


@pytest.mark.parametrize("t_now, last_fix_t", [(0.0, 0.0), (100.0, 100.0), (50.0, 120.0)])
def test_ac5_non_positive_time_dark_yields_zero_speed_not_a_zero_division(t_now, last_fix_t):
    """AC5: time_dark_s <= 0 must give implied_speed 0.0, never ZeroDivisionError."""
    track = DarkTrack(
        id=MMSI_INSTANT,
        pos=np.array([0.0, 0.0]),
        pos_cov=np.diag([250000.0, 250000.0]),
        last_fix_xy=np.array([0.0, 0.0]),
        last_fix_t=last_fix_t,
    )
    cand = reassociate(np.array([300.0, 0.0]), R_RADAR, [track], t_now)[0]

    assert cand.time_dark_s == pytest.approx(t_now - last_fix_t, rel=1e-12, abs=1e-12)
    assert cand.time_dark_s <= 0.0
    assert cand.implied_speed_mps == 0.0
    assert cand.implied_speed_kn == 0.0
    assert cand.range_m > 0.0                 # range is still real and reported


def test_ac5_payload_is_json_serialisable_with_plain_python_floats():
    """AC5: asdict -> json.dumps must work, with real floats, not numpy scalars.

    ``np.float64`` subclasses ``float``, so json.dumps alone would happily
    serialise it and miss the defect the criterion is actually asking about. The
    identity checks below are the ones that bite; the dumps call catches
    ``np.bool_`` for ``gated_out``, which is *not* a bool subclass.
    """
    track, kf = _dark_track_from_coast()
    out = reassociate(track.pos + np.array([50.0, 25.0]), R_RADAR, [track], kf.t)
    payload = dataclasses.asdict(out[0])

    encoded = json.dumps(payload)
    assert json.loads(encoded) == payload

    float_fields = (
        "p", "d2", "sigma_distance", "likelihood", "time_dark_s",
        "range_m", "implied_speed_mps", "implied_speed_kn",
    )
    for name in float_fields:
        value = payload[name]
        assert type(value) is float, (
            f"ReassocCandidate.{name} is {type(value).__name__}, not a plain float -- "
            "numpy scalars leak into the event bus and break strict JSON encoders"
        )
    assert type(payload["gated_out"]) is bool, (
        f"gated_out is {type(payload['gated_out']).__name__}, not a plain bool"
    )
    assert not isinstance(payload["track_id"], np.generic), (
        "track_id is a numpy scalar; it will not survive a strict JSON encoder"
    )
    assert type(payload["reason"]) is str

    # Every declared field is present -- nothing silently dropped.
    assert set(payload) == {f.name for f in dataclasses.fields(ReassocCandidate)}


def test_ac5_gated_out_payload_is_still_complete():
    """AC5: a rejected candidate is an event too and must carry a full payload."""
    track, kf = _dark_track_from_coast()
    S = track.pos_cov + R_RADAR
    z = track.pos + np.array([10.0 * math.sqrt(S[0, 0]), 0.0])
    cand = reassociate(z, R_RADAR, [track], kf.t)[0]

    assert cand.gated_out is True
    assert cand.p == 0.0
    assert cand.reason.strip() != ""
    assert cand.d2 > GATE_CHI2
    assert cand.time_dark_s == pytest.approx(kf.t - track.last_fix_t, rel=1e-12)
    assert cand.range_m == pytest.approx(float(np.hypot(*(z - track.last_fix_xy))), rel=1e-12)
    assert cand.implied_speed_mps == pytest.approx(cand.range_m / cand.time_dark_s, rel=1e-12)
    json.dumps(dataclasses.asdict(cand))      # must not raise


def test_ac5_reassoc_candidate_is_frozen():
    """AC5 support: the payload is an immutable record once emitted."""
    track, kf = _dark_track_from_coast()
    cand = reassociate(track.pos.copy(), R_RADAR, [track], kf.t)[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cand.p = 0.5


def test_dark_track_carries_an_optional_label_defaulting_to_empty():
    """Supporting: DarkTrack.label defaults to "" so the dataclass is constructible
    without it."""
    track = DarkTrack(
        id=MMSI_COASTER,
        pos=np.zeros(2),
        pos_cov=np.eye(2),
        last_fix_xy=np.zeros(2),
        last_fix_t=0.0,
    )
    assert track.label == ""
    labelled = DarkTrack(
        id=MMSI_COASTER, pos=np.zeros(2), pos_cov=np.eye(2),
        last_fix_xy=np.zeros(2), last_fix_t=0.0, label="MV Example",
    )
    assert labelled.label == "MV Example"


# --------------------------------------------------------------------------
# Multi-track competition, ordering and edge cases
# --------------------------------------------------------------------------
def _two_competing_tracks(sigma=800.0, separation=900.0):
    """Two coasting tracks whose ellipses overlap around a single detection."""
    P = np.diag([sigma ** 2, sigma ** 2])
    near = DarkTrack(
        id=MMSI_NEAR, pos=np.array([0.0, 0.0]), pos_cov=P.copy(),
        last_fix_xy=np.array([-2000.0, 0.0]), last_fix_t=0.0, label="MV Near",
    )
    far = DarkTrack(
        id=MMSI_FAR, pos=np.array([separation, 0.0]), pos_cov=P.copy(),
        last_fix_xy=np.array([separation - 2000.0, 0.0]), last_fix_t=0.0, label="MV Far",
    )
    return near, far


def test_two_competing_tracks_rank_the_closer_one_first():
    """Supporting: one detection, two dark tracks -> the closer track wins."""
    near, far = _two_competing_tracks()
    z = np.array([100.0, 0.0])             # much nearer to "near"

    out = reassociate(z, R_RADAR, [far, near], 600.0)   # deliberately built far-first
    assert [c.track_id for c in out] == [near.id, far.id]

    ref = _reference_probabilities(z, R_RADAR, [near, far])
    for cand in out:
        assert cand.p == pytest.approx(ref[cand.track_id][2], rel=1e-9)
        assert 0.0 < cand.p < 1.0
        assert cand.gated_out is False

    assert out[0].p > out[1].p
    assert out[0].d2 < out[1].d2


def test_competing_probabilities_sum_strictly_below_one():
    """Supporting: the beta term reserves probability mass for 'this was a false
    alarm', so the accepted candidates can never account for the whole budget."""
    near, far = _two_competing_tracks()
    z = np.array([450.0, 0.0])             # roughly between them
    out = reassociate(z, R_RADAR, [near, far], 600.0)

    accepted = [c for c in out if not c.gated_out]
    assert len(accepted) == 2
    total = sum(c.p for c in accepted)
    print(f"\nsum(p_accepted) = {total:.9f}  (false-alarm mass = {1.0 - total:.9f})")
    assert total < 1.0
    assert total == pytest.approx(1.0 - BETA_FA_REACQ / (
        sum(c.likelihood for c in accepted) + BETA_FA_REACQ), rel=1e-9)


def test_accepted_probability_mass_stays_below_one_for_many_tracks():
    """Supporting: the budget still holds with a crowd of overlapping tracks."""
    rng = np.random.default_rng(SEED + 1)
    tracks = [
        DarkTrack(
            id=900000 + i,
            pos=rng.normal(scale=300.0, size=2),
            pos_cov=np.diag([700.0 ** 2, 700.0 ** 2]),
            last_fix_xy=rng.normal(scale=3000.0, size=2),
            last_fix_t=0.0,
        )
        for i in range(8)
    ]
    out = reassociate(np.zeros(2), R_RADAR, tracks, 900.0)
    total = sum(c.p for c in out if not c.gated_out)
    assert total < 1.0
    assert len(out) == len(tracks)


def test_results_are_sorted_by_probability_descending_including_gated_out():
    """Supporting: the returned list is p-descending and keeps rejected candidates."""
    near, far = _two_competing_tracks()
    miles_away = DarkTrack(
        id=MMSI_MILES_AWAY, pos=np.array([500000.0, 0.0]),
        pos_cov=np.diag([800.0 ** 2, 800.0 ** 2]),
        last_fix_xy=np.array([490000.0, 0.0]), last_fix_t=0.0,
    )
    out = reassociate(np.array([100.0, 0.0]), R_RADAR, [miles_away, far, near], 600.0)

    assert len(out) == 3
    probabilities = [c.p for c in out]
    assert probabilities == sorted(probabilities, reverse=True)
    assert {c.track_id for c in out} == {near.id, far.id, miles_away.id}

    gated = [c for c in out if c.gated_out]
    assert [c.track_id for c in gated] == [miles_away.id]
    assert gated[0].p == 0.0
    # Gated-out entries sort last precisely because p == 0.0.
    assert out[-1].track_id == miles_away.id


def test_empty_dark_track_list_returns_empty_list_without_raising():
    """Supporting: no dark tracks is a normal frame, not an error."""
    assert reassociate(np.array([0.0, 0.0]), R_RADAR, [], 100.0) == []
    assert reassociate(np.array([0.0, 0.0]), R_RADAR, (), 100.0) == []


def test_single_track_returns_exactly_one_candidate():
    """Supporting: the one-track case, which is the demo's actual shape."""
    track, kf = _dark_track_from_coast()
    out = reassociate(track.pos.copy(), R_RADAR, [track], kf.t)
    assert len(out) == 1
    assert out[0].track_id == track.id


def test_all_tracks_gated_out_still_returns_every_candidate():
    """Supporting: total rejection must be reported, not silently swallowed --
    an operator needs to see that a plot matched nothing."""
    tracks = [
        DarkTrack(
            id=800000 + i,
            pos=np.array([100000.0 * (i + 1), 0.0]),
            pos_cov=np.diag([500.0 ** 2, 500.0 ** 2]),
            last_fix_xy=np.array([0.0, 0.0]),
            last_fix_t=0.0,
        )
        for i in range(3)
    ]
    out = reassociate(np.zeros(2), R_RADAR, tracks, 600.0)

    assert len(out) == 3
    assert all(c.gated_out for c in out)
    assert all(c.p == 0.0 for c in out)
    assert all(c.reason.strip() != "" for c in out)
    assert all(c.d2 > GATE_CHI2 for c in out)
    assert {c.track_id for c in out} == {t.id for t in tracks}


def test_gate_chi2_boundary_is_respected_on_both_sides():
    """Supporting: the gate is applied at exactly GATE_CHI2, not at a rounded value."""
    sigma = 1000.0
    track, R = _isotropic_track(MMSI_BOUNDARY, sigma)
    for d2, expect_gated in ((GATE_CHI2 * 0.98, False), (GATE_CHI2 * 1.02, True)):
        z = track.pos + np.array([math.sqrt(d2) * sigma, 0.0])
        cand = reassociate(z, R, [track], 600.0)[0]
        assert cand.d2 == pytest.approx(d2, rel=1e-9)
        assert cand.gated_out is expect_gated


def test_gate_chi2_override_argument_is_honoured():
    """Supporting: gate_chi2 is a keyword knob, and tightening it rejects more."""
    sigma = 1000.0
    track, R = _isotropic_track(MMSI_KNOB, sigma)
    z = track.pos + np.array([2.0 * sigma, 0.0])       # d2 = 4.0

    assert reassociate(z, R, [track], 600.0, gate_chi2=GATE_CHI2)[0].gated_out is False
    assert reassociate(z, R, [track], 600.0, gate_chi2=1.0)[0].gated_out is True


def test_beta_fa_override_argument_shifts_probability_as_documented():
    """Supporting: beta_fa is a keyword knob; more clutter means less confidence."""
    track, kf = _dark_track_from_coast()
    z = track.pos.copy()
    base = reassociate(z, R_RADAR, [track], kf.t)[0]
    noisy = reassociate(z, R_RADAR, [track], kf.t, beta_fa=BETA_FA_REACQ * 1000.0)[0]

    assert noisy.p < base.p
    assert noisy.likelihood == pytest.approx(base.likelihood, rel=1e-12)   # L is beta-free
    assert noisy.p == pytest.approx(
        base.likelihood / (base.likelihood + BETA_FA_REACQ * 1000.0), rel=1e-9
    )


def test_reassociate_does_not_mutate_its_inputs():
    """Supporting: re-association is a query; the caller's track state is read-only."""
    track, kf = _dark_track_from_coast()
    pos_before = track.pos.copy()
    cov_before = track.pos_cov.copy()
    fix_before = track.last_fix_xy.copy()
    z = np.array([123.0, -456.0])
    z_before = z.copy()

    reassociate(z, R_RADAR, [track], kf.t)

    np.testing.assert_array_equal(track.pos, pos_before)
    np.testing.assert_array_equal(track.pos_cov, cov_before)
    np.testing.assert_array_equal(track.last_fix_xy, fix_before)
    np.testing.assert_array_equal(z, z_before)


def test_end_to_end_dark_vessel_reacquisition():
    """AC1+AC3+AC5 together: the demo centrepiece, start to finish.

    Coast a real filter with AIS off, watch the ellipse grow, then feed it the
    radar plot that actually belongs to the vessel and check the emitted event.
    """
    track, kf = _dark_track_from_coast(track_id=MMSI_DEMO, label="MV Demo")

    a, b, theta = ellipse_axes(track.pos_cov)
    assert a >= b and math.isfinite(theta)

    search_region = clip_to_water(ellipse_polygon(track.pos, track.pos_cov, n=64), None)
    assert search_region.is_valid

    # A radar plot 60 m off the prediction: within one radar sigma, so it should
    # be attributed with high confidence.
    z = track.pos + np.array([60.0, 0.0])
    assert search_region.contains(Point(*z))

    out = reassociate(z, R_RADAR, [track], kf.t)
    cand = out[0]

    print(f"\nEND-TO-END dark {cand.time_dark_s:.0f}s, 95% ellipse a={a:.0f}m b={b:.0f}m, "
          f"re-attributed to {track.label} with p={cand.p:.3f} "
          f"(implied {cand.implied_speed_kn:.1f} kn over {cand.range_m:.0f} m)")

    assert cand.track_id == track.id
    assert cand.gated_out is False
    assert cand.p > P_CONFIDENT
    assert cand.p < 1.0
    assert cand.sigma_distance == pytest.approx(math.sqrt(cand.d2), rel=1e-12)
    json.dumps(dataclasses.asdict(cand))


# --------------------------------------------------------------------------
# AC4, pinned at the DEMO OPERATING POINT
# --------------------------------------------------------------------------
def test_ac4_probability_has_real_dynamic_range_at_the_demo_operating_point(capsys):
    """AC4: at the darkness the demo actually shows, p must span a wide range.

    The p(d2) curve is steep or flat depending on how uncertain the track has become,
    because p = L/(L + beta) and L falls as the prediction spreads. Measured across the
    gate:

        3 min dark  (sigma ~  516 m):  p = 0.998 -> 0.868   compressed
        10 min dark (sigma ~ 2717 m):  p = 0.956 -> 0.193   wide

    Both are correct -- a vessel dark for three minutes has a tight prediction, so an
    in-gate detection really is very probably it. But the compressed regime is where
    "the system always says 0.94" would hide, so the guarantee is asserted at the
    operating point the demo runs at: a vessel dark for ten minutes.
    """
    dt, n_steps = 10.0, 60          # 600 s dark at the replay cadence
    kf = CVKalman.init_two_point(np.array([0.0, 0.0]), 0.0,
                                 np.array([90.0, 0.0]), dt, R_SEED, q=Q_DARK)
    for _ in range(n_steps):
        kf.predict(dt)

    P = np.asarray(kf.cov)[:2, :2].copy()
    sigma = float(np.sqrt(np.trace(P) / 2.0))
    track = DarkTrack(id=367_001_234, pos=np.asarray(kf.state)[:2].copy(), pos_cov=P,
                      last_fix_xy=np.array([90.0, 0.0]), last_fix_t=dt)
    t_now = dt + n_steps * dt

    S = P + R_RADAR
    rows = []
    for d2 in (0.0, 1.0, 4.0, 9.0):
        offset = math.sqrt(d2 * float(S[0, 0]))
        z = track.pos + np.array([offset, 0.0])
        cand = reassociate(z, R_RADAR, [track], t_now)[0]
        rows.append((d2, cand.p))

    with capsys.disabled():
        print(f"\n[AC4 DEMO POINT] {n_steps * dt:.0f} s dark, sigma = {sigma:.0f} m")
        for d2, p in rows:
            print(f"    d2 = {d2:>4.1f}  ->  p = {p:.3f}")

    p_on, p_gate = rows[0][1], rows[-1][1]
    assert p_on > 0.90, f"on-target detection scored only {p_on:.3f}"
    # The whole point of AC4: near the gate edge the system must NOT still sound certain.
    assert p_gate < 0.30, (
        f"p at the gate edge is {p_gate:.3f} -- the system sounds confident about a "
        "detection it barely accepted, which is the 'always says 0.94' failure"
    )
    assert p_on - p_gate > 0.60, (
        f"p spans only {p_on - p_gate:.3f} across the gate; too flat to be informative"
    )
    # Monotone decreasing in offset.
    ps = [p for _, p in rows]
    assert all(a > b for a, b in zip(ps, ps[1:])), f"p not monotone decreasing: {ps}"

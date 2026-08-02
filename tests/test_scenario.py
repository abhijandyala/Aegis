"""Phase 1 acceptance tests for the scenario loader (data/scenario.py).

The synthetic pack (s02) tests always run. The real-AIS pack (s01) tests need
its cached ais_window.csv, which is committed with the pack; they skip with a
loud reason if it is missing rather than failing someone who hasn't pulled it.
"""

import os
import time

import pytest

from data.contracts import AIS_SIGMA_M, Measurement
from data.scenario import SCENARIOS_DIR, find_crossings, lla_to_enu, load_scenario

S01 = "s01_dark_in_sanctuary"
S02 = "s02_synthetic_demo"
_S01_CSV = os.path.join(SCENARIOS_DIR, S01, "ais_window.csv")

needs_ais = pytest.mark.skipif(
    not os.path.isfile(_S01_CSV),
    reason=f"{_S01_CSV} missing -- run scripts/extract_window.py",
)


@pytest.fixture(scope="module")
def s02():
    return load_scenario(S02)


@pytest.fixture(scope="module")
def s01():
    if not os.path.isfile(_S01_CSV):
        pytest.skip("s01 AIS window not extracted")
    return load_scenario(S01)


# ------------------------------------------------------------------ geometry

def test_enu_roundtrip_and_scale():
    # One degree of longitude at 37.825N is ~88 km; sanity-check the projection.
    x, y = lla_to_enu(37.825, -122.4 + 1.0, 37.825, -122.4)
    assert 87_000 < x < 89_000 and abs(y) < 1.0
    x, y = lla_to_enu(37.825 + 0.01, -122.4, 37.825, -122.4)
    assert 1_050 < y < 1_170 and abs(x) < 1e-6


# ------------------------------------------------------- loader, synthetic pack

def test_load_returns_frames_and_geofences(s02):
    assert s02.n_frames == 121                      # 60 min at 30 s + fencepost
    assert s02.frames[1].t - s02.frames[0].t == 30.0
    assert [g.fence_id for g in s02.geofences] == ["central_bay_sanctuary"]
    # Geofence is in ENU metres. It's DELIBERATELY tight around Alcatraz waters
    # (see the geojson's own "note" property) so routine ferry traffic doesn't
    # intrude -- a few hundred metres to ~2 km, not the several-km span an
    # earlier version of this assertion assumed.
    ring_x = [p[0] for p in s02.geofences[0].ring]
    assert 500 < max(ring_x) - min(ring_x) < 3_000


def test_geofence_contains(s02):
    fence = s02.geofences[0]
    # Centroid of the actual ring, not an assumed "mid-sanctuary" point -- the
    # real polygon (-2810..-1054 x, -556..1001 y) doesn't contain (0, 1500),
    # which was sized for an earlier, larger placeholder geometry.
    assert fence.contains(-1932.3, 222.4)
    assert not fence.contains(50_000.0, 50_000.0)


def test_ground_truth_covers_every_measurement(s02):
    for frame in s02.frames:
        for m in frame.measurements:
            assert m.meas_id in s02.ground_truth
            assert s02.ground_truth[m.meas_id] in s02.true_tracks


def test_no_measurement_carries_identity(s02):
    """The headline rule: identity never rides on a Measurement.

    The dataclass has no identity field, and its actual field set is pinned so
    adding one later fails here, not in a demo.
    """
    assert set(Measurement.__dataclass_fields__) == {
        "meas_id", "t", "x", "y", "source", "sigma",
    }
    for frame in s02.frames:
        for m in frame.measurements:
            assert not hasattr(m, "mmsi")
            if m.source == "radar":
                # Radar detections are anonymous by construction; the only
                # link to identity is the ground-truth side table.
                assert not any(str(v).startswith("mmsi:") for v in
                               (m.meas_id, m.source))


def test_ais_off_suppresses_and_radar_contact_injects(s02):
    """The LOADER's promise, not the walker's: "the loader always emits the
    full unsuppressed picture" (see README's Data layer section) -- ais_off
    suppression and radar_contact injection are state changes applied later by
    aegis.driver.ScenarioDriver. This test checks the input the driver
    receives: raw AIS runs the FULL duration unsuppressed, no radar-source
    measurement exists yet, and every scripted event for this actor is
    present, in order, with the pack's actual current timings/params.
    """
    ghost_meas = [
        m for f in s02.frames for m in f.measurements
        if s02.ground_truth[m.meas_id] == "ghost_1"
    ]
    ais = [m for m in ghost_meas if m.source == "ais"]
    radar = [m for m in ghost_meas if m.source == "radar"]
    assert max(m.t for m in ais) >= 3600.0 - 30.0   # unsuppressed to the end
    assert radar == []                              # not the loader's job yet

    ghost_events = [ev for ev in s02.events if ev.actor == "ghost_1"]
    assert [(ev.kind, ev.t) for ev in ghost_events] == [
        ("ais_off", 180.0),
        ("radar_contact", 420.0), ("radar_contact", 720.0),
        ("radar_contact", 1020.0), ("radar_contact", 1500.0),
        ("radar_contact", 2100.0),
    ]
    assert all(ev.params.get("sigma") == 50.0
               for ev in ghost_events if ev.kind == "radar_contact")
    # The vessel still exists while dark: truth continues past the event.
    assert s02.true_tracks["ghost_1"][-1][0] >= 3600.0 - 30.0


def test_radar_contact_lands_near_truth(s02):
    """true_position_at() (public -- used by ScenarioDriver to place injected
    radar contacts on the actor's real, possibly-dark position; not the
    private `_true_position_at` an earlier version of this test imported,
    which never existed in this file) interpolates correctly along the
    resampled true track at every real event timestamp for this actor.
    """
    from data.scenario import true_position_at
    track = s02.true_tracks["ghost_1"]
    for ev in s02.events:
        if ev.actor != "ghost_1":
            continue
        pos = true_position_at(track, ev.t)
        assert pos is not None
        # Within one resample tick's worth of travel of a directly-interpolated
        # frame straddling ev.t -- confirms it's reading the real track, not a
        # placeholder.
        before = [p for p in track if p[0] <= ev.t]
        after = [p for p in track if p[0] >= ev.t]
        assert before and after
        lo, hi = before[-1], after[0]
        assert min(lo[1], hi[1]) - 1.0 <= pos[0] <= max(lo[1], hi[1]) + 1.0
        assert min(lo[2], hi[2]) - 1.0 <= pos[1] <= max(lo[2], hi[2]) + 1.0


def test_synthetic_crossing_found(s02):
    crossings = find_crossings(s02.true_tracks, s02.frame_interval_s)
    pairs = {c["pair"] for c in crossings}
    assert ("eastbound_1", "westbound_1") in pairs
    assert all(c["min_dist_m"] <= 500.0 for c in crossings)


def test_loads_are_deterministic():
    a = load_scenario(S02)
    b = load_scenario(S02)
    assert [m for f in a.frames for m in f.measurements] == \
           [m for f in b.frames for m in f.measurements]


def test_identity_change_swaps_display_only(s02_swap=None):
    s01_pack = load_scenario(S02)                   # no identity_change here
    assert s01_pack.display_id("ghost_1", 9_999.0) == "ghost_1"


# ----------------------------------------------------------- real AIS pack

# NOTE (Omar, 2026-07-26): s01_dark_in_sanctuary's content changed from a real-AIS
# SF Bay scene (30+ vessels, ghost_1/spoofer_1) to a synthetic Monterey Bay NMS
# demo per the pitch brief -- single target, no real AIS, no identity_change.
# The three tests below were written against the old content and are replaced
# with equivalents against the new one. Real-AIS ingestion (_ingest_ais) has no
# dedicated test now that s01 doesn't exercise it -- flagging for Abhi since he
# owns the loader; a small fixture pack with a tiny real ais_window.csv would
# close that gap without needing a full scenario.

def test_s01_load_under_five_seconds():
    start = time.perf_counter()
    load_scenario(S01)
    assert time.perf_counter() - start < 5.0


def test_s01_narrative_timeline(s01):
    """The demo's beat sheet: enter -> dark -> radar contact, on the synthetic
    target, now running alongside real ambient SF Bay AIS traffic (pack.json's
    ais_csv was wired to the real, already-committed ais_window.csv, and
    time_window was aligned to that CSV's actual coverage
    2024-06-15T17:55:00 - 19:55:00 UTC -- the pack originally declared
    06:00-08:00 UTC, which never overlapped the real data at all, so real
    traffic silently never loaded; every *_at_s figure in `expected` is
    frame-relative and is untouched by that wall-clock shift).
    """
    assert s01.n_vessels > 200                   # real ambient traffic + target_1
    assert "target_1" in s01.true_tracks

    ghost_meas = [m for f in s01.frames for m in f.measurements
                  if s01.ground_truth[m.meas_id] == "target_1"]
    ais = [m for m in ghost_meas if m.source == "ais"]
    radar = [m for m in ghost_meas if m.source == "radar"]

    # Loader promise (see test_ais_off_suppresses_and_radar_contact_injects):
    # unsuppressed to the end, no radar-source measurement injected yet.
    assert max(m.t for m in ais) >= s01.duration_s - 30.0
    assert radar == []

    dark_ev = [ev for ev in s01.events if ev.kind == "ais_off" and ev.actor == "target_1"]
    radar_ev = [ev for ev in s01.events if ev.kind == "radar_contact" and ev.actor == "target_1"]
    assert [ev.t for ev in dark_ev] == [s01.expected["dark_at_s"]]
    assert [ev.t for ev in radar_ev] == [s01.expected["radar_contact_at_s"]]

    fence = s01.geofences[0]
    assert fence.fence_id == "monterey_bay_nms"
    from data.scenario import true_position_at
    pos_at_dark = true_position_at(s01.true_tracks["target_1"], s01.expected["dark_at_s"])
    assert fence.contains(*pos_at_dark)          # still inside the sanctuary when it goes dark

    for key in ("id_switches_max", "reassoc_p_min", "geofence_alert"):
        assert key in s01.expected


def test_s01_no_identity_change(s01):
    # No identity_change events in the new s01 -- display_id is a no-op passthrough.
    assert s01.display_id("target_1", 9_999.0) == "target_1"


def test_pack_swap_needs_no_code_changes():
    """Same call, different pack id -- the whole acceptance criterion."""
    for pack in (S02, S01) if os.path.isfile(_S01_CSV) else (S02,):
        s = load_scenario(pack)
        assert s.frames and s.geofences


# ------------------------------------------------------------- s02_mmsi_spoof

def test_s02_mmsi_spoof_same_display_id():
    """Two vessels, one spoofed MMSI -- ground truth keeps them distinct."""
    s = load_scenario("s02_mmsi_spoof")
    assert set(s.true_tracks) == {"vessel_a", "vessel_b"}
    assert s.display_id("vessel_a", 0.0) == s.display_id("vessel_b", 0.0) == "mmsi:412345678"

    pos_a = s.true_tracks["vessel_a"][0]
    pos_b = s.true_tracks["vessel_b"][0]
    sep_m = ((pos_a[1] - pos_b[1]) ** 2 + (pos_a[2] - pos_b[2]) ** 2) ** 0.5
    assert 39_000 < sep_m < 41_000                  # ~40 km apart, per the brief

    for frame in s.frames:
        for m in frame.measurements:
            assert not hasattr(m, "mmsi")           # spoof lives in display_id, not on the measurement


def test_s02_mmsi_spoof_runs_on_same_engine():
    """Zero code changes claim: same load_scenario call as every other pack."""
    s = load_scenario("s02_mmsi_spoof")
    assert s.frames and s.n_vessels == 2


# ------------------------------------------------------------- s03_ghost_fleet

def test_s03_ghost_fleet_relink_and_ofac_hit():
    s = load_scenario("s03_ghost_fleet")
    assert list(s.true_tracks) == ["actor_1"]        # one continuous kinematic track
    assert s.display_id("actor_1", 0.0) == "mmsi:367111222"
    assert s.display_id("actor_1", 1800.0) == "mmsi:572469210"
    assert s.expected["ofac_hit_mmsi"] == "mmsi:572469210"

    import csv
    ofac_path = os.path.join(SCENARIOS_DIR, "s03_ghost_fleet", "ofac_sdn_vessels_subset.csv")
    with open(ofac_path, newline="", encoding="utf-8") as f:
        names = {row["SDN_Name"] for row in csv.DictReader(f)}
    assert s.expected["ofac_hit_name"] in names       # real OFAC lookup, not a synthetic label

"""Phase 3.5 acceptance tests for stateful geofence analysis.

All coordinates are local ENU metres, never longitude/latitude.
"""

from __future__ import annotations

import pytest

import aegis.geofence as geofence
from tracker.geofence import GeofenceIndex, corridor_fence, polygon_fence


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gf():
    return geofence


@pytest.fixture(scope="module")
def fences():
    """Four fences in ENU metres about an origin near 36.80 N, 121.90 W.

    ``mbnms`` and ``elkhorn`` overlap on 15000..20000 x 5000..15000, which is
    what the overlapping-fence test sails through.
    """
    return GeofenceIndex([
        polygon_fence("mbnms", "mpa", "Monterey Bay NMS",
                      [[0.0, 0.0], [20000.0, 0.0],
                       [20000.0, 20000.0], [0.0, 20000.0]]),
        polygon_fence("elkhorn", "mpa", "Elkhorn Slough Reserve",
                      [[15000.0, 5000.0], [25000.0, 5000.0],
                       [25000.0, 15000.0], [15000.0, 15000.0]]),
        corridor_fence("trans_pac", "Trans-Pacific Cable",
                       [[40000.0, 0.0], [40000.0, 30000.0]], 1500.0),
        polygon_fence("moss_port", "port", "Moss Landing Harbour",
                      [[-8000.0, -6000.0], [-4000.0, -6000.0],
                       [-4000.0, -2000.0], [-8000.0, -2000.0]]),
    ])


@pytest.fixture
def mon(gf, fences):
    """A monitor with the fences installed and a clean event log."""
    gf.set_index(fences)
    gf.reset()
    return gf


# Convenient points.
OPEN_WATER = [-5000.0, 10000.0]      # outside every fence
IN_MBNMS = [5000.0, 10000.0]         # mbnms only
IN_MBNMS_2 = [6000.0, 10000.0]       # mbnms only, a little further along
IN_OVERLAP = [17000.0, 10000.0]      # mbnms AND elkhorn
IN_PORT = [-6000.0, -4000.0]         # moss_port only
IN_CABLE = [40000.0, 15000.0]        # trans_pac corridor only
FAR_AWAY = [-30000.0, -30000.0]      # outside every fence, nowhere near one


def types_for(mon, actor: str) -> list[str]:
    return [e["type"] for e in mon.events_for(actor)]


# ---------------------------------------------------------------------------
# Sanity: the geometry fixture really does say what the transition tests assume
# ---------------------------------------------------------------------------

def test_fixture_geometry_is_what_the_tests_assume(fences):
    assert fences.query(IN_MBNMS) == ["mbnms"]
    assert fences.query(IN_OVERLAP) == ["elkhorn", "mbnms"]
    assert fences.query(OPEN_WATER) == []
    assert fences.query(IN_PORT) == ["moss_port"]
    assert fences.query(IN_CABLE) == ["trans_pac"]


# ===========================================================================
# ACCEPTANCE 1: entering a sanctuary fires EXACTLY ONCE
# ===========================================================================

def test_enter_fires_exactly_once_across_fifty_frames_inside(mon):
    """The single most important behaviour in the phase.

    Fifty successful point-in-polygon tests are ONE event. If this regresses,
    the operator gets fifty identical alerts for one intrusion and stops reading
    them, which is worse than having no geofence at all.
    """
    mon.ingest_frame(["V-12"], [OPEN_WATER])
    for i in range(50):
        mon.ingest_frame(["V-12"], [[1000.0 + i * 100.0, 10000.0]])

    assert mon.count_of_type("geofence_enter") == 1
    # And nothing else fired either: no exits, no spurious AIS events.
    assert mon.event_types() == ["geofence_enter"]

    ev = mon.events_of_type("geofence_enter")[0]
    assert ev["actor"] == "V-12"
    assert ev["geofence_id"] == "mbnms"
    # Fired on the frame the vessel crossed the line, not on frame 1.
    assert ev["frame"] == 2


def test_frames_still_inside_emit_nothing(mon):
    """The 'still inside' arm of the state machine emits literally nothing."""
    mon.ingest_frame(["V-77"], [OPEN_WATER])
    first = mon.ingest_frame(["V-77"], [IN_MBNMS])
    assert [e["type"] for e in first] == ["geofence_enter"]

    for _ in range(5):
        assert mon.ingest_frame(["V-77"], [IN_MBNMS_2]) == []


def test_a_track_that_never_enters_anything_emits_nothing(mon):
    for _ in range(10):
        mon.ingest_frame(["V-99"], [OPEN_WATER])
    assert mon.events() == []
    # It is still under watch, though -- the graph node exists.
    assert mon.watch_count() == 1


# ===========================================================================
# ACCEPTANCE 2: dark inside a sanctuary outranks a plain enter
# ===========================================================================

def test_dark_inside_is_the_compound_event_with_strictly_higher_rank(mon):
    mon.ingest_frame(["V-77"], [OPEN_WATER])
    mon.ingest_frame(["V-77"], [IN_MBNMS])
    dark_frame = mon.ingest_frame(["V-77"], [IN_MBNMS_2], ["V-77"])

    assert len(dark_frame) == 1
    compound = dark_frame[0]
    assert compound["type"] == "dark_inside_protected_area"
    assert compound["actor"] == "V-77"
    assert compound["geofence_id"] == "mbnms"

    enter = mon.events_of_type("geofence_enter")[0]
    # STRICT numeric ordering, not string equality -- the rank is on the event
    # precisely so this assertion can be one line.
    assert compound["severity_rank"] > enter["severity_rank"]
    assert compound["severity_rank"] == mon.severity_rank(compound["severity"])
    assert enter["severity_rank"] == mon.severity_rank(enter["severity"])


def test_going_dark_inside_does_not_also_emit_a_plain_ais_dark(mon):
    """Darkness inside a fence is reported ONCE, as the compound event."""
    mon.ingest_frame(["V-77"], [IN_MBNMS])
    mon.ingest_frame(["V-77"], [IN_MBNMS_2], ["V-77"])
    assert mon.count_of_type("ais_dark") == 0
    assert mon.count_of_type("dark_inside_protected_area") == 1


def test_compound_fires_once_no_matter_how_long_the_vessel_stays_dark(mon):
    mon.ingest_frame(["V-77"], [IN_MBNMS])
    for i in range(20):
        mon.ingest_frame(["V-77"], [[6000.0 + i * 50.0, 10000.0]], ["V-77"])
    assert mon.count_of_type("dark_inside_protected_area") == 1


def test_severity_table_orders_dark_inside_above_enter_for_every_kind(mon):
    """The invariant has to hold for every fence kind, not just the one the
    demo happens to sail through."""
    for kind in ("mpa", "cable", "port", "danger"):
        enter = mon.enter_severity(kind)
        compound = mon.dark_inside_severity(kind)
        assert mon.severity_rank(compound) > mon.severity_rank(enter), kind


def test_severity_scale_is_strictly_ordered(mon):
    ranks = [mon.severity_rank(s)
             for s in ("info", "warning", "critical", "emergency")]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == 4


def test_fence_kind_moves_the_severity(mon):
    """An mpa is a bigger deal than a port, and the events say so."""
    mon.ingest_frame(["V-A", "V-B"], [OPEN_WATER, OPEN_WATER])
    mon.ingest_frame(["V-A", "V-B"], [IN_MBNMS, IN_PORT])

    mpa_enter = [e for e in mon.events_of_type("geofence_enter")
                 if e["geofence_id"] == "mbnms"][0]
    port_enter = [e for e in mon.events_of_type("geofence_enter")
                  if e["geofence_id"] == "moss_port"][0]
    assert mpa_enter["severity_rank"] > port_enter["severity_rank"]

    mon.ingest_frame(["V-A", "V-B"], [IN_MBNMS_2, IN_PORT], ["V-A", "V-B"])
    compound = {e["geofence_id"]: e
                for e in mon.events_of_type("dark_inside_protected_area")}
    assert set(compound) == {"mbnms", "moss_port"}
    assert compound["mbnms"]["severity_rank"] > compound["moss_port"]["severity_rank"]


# ===========================================================================
# ACCEPTANCE 3: exit fires once on leaving; re-entry fires again
# ===========================================================================

def test_exit_fires_once_on_leaving(mon):
    mon.ingest_frame(["V-77"], [OPEN_WATER])
    mon.ingest_frame(["V-77"], [IN_MBNMS])
    for _ in range(3):
        mon.ingest_frame(["V-77"], [IN_MBNMS_2])

    leaving = mon.ingest_frame(["V-77"], [OPEN_WATER])
    assert [e["type"] for e in leaving] == ["geofence_exit"]
    assert leaving[0]["geofence_id"] == "mbnms"

    # Staying outside says nothing more.
    for _ in range(3):
        assert mon.ingest_frame(["V-77"], [OPEN_WATER]) == []
    assert mon.count_of_type("geofence_exit") == 1


def test_leave_and_re_enter_gives_two_enters_and_one_exit_between(mon):
    for point in (OPEN_WATER, IN_MBNMS, OPEN_WATER, IN_MBNMS):
        mon.ingest_frame(["V-77"], [point])

    assert types_for(mon, "V-77") == [
        "geofence_enter", "geofence_exit", "geofence_enter",
    ]


def test_full_enter_stay_dark_exit_sequence(mon):
    """The headline narrative, end to end, in the order the operator sees it."""
    mon.ingest_frame(["V-77"], [OPEN_WATER])
    mon.ingest_frame(["V-77"], [IN_MBNMS])
    mon.ingest_frame(["V-77"], [IN_MBNMS_2])
    mon.ingest_frame(["V-77"], [[7000.0, 10000.0]])
    mon.ingest_frame(["V-77"], [[8000.0, 10000.0]], ["V-77"])
    mon.ingest_frame(["V-77"], [[9000.0, 10000.0]], ["V-77"])
    mon.ingest_frame(["V-77"], [OPEN_WATER], ["V-77"])

    assert types_for(mon, "V-77") == [
        "geofence_enter", "dark_inside_protected_area", "geofence_exit",
    ]
    # Seven frames, three events.
    assert [e["frame"] for e in mon.events()] == [2, 5, 7]


# ===========================================================================
# ACCEPTANCE 4: overlapping fences produce one event each
# ===========================================================================

def test_overlapping_fences_produce_one_event_each_not_one_merged(mon):
    mon.ingest_frame(["V-31"], [OPEN_WATER])
    both = mon.ingest_frame(["V-31"], [IN_OVERLAP])

    assert len(both) == 2
    assert {e["type"] for e in both} == {"geofence_enter"}
    assert sorted(e["geofence_id"] for e in both) == ["elkhorn", "mbnms"]
    assert mon.inside_of("V-31") == ["elkhorn", "mbnms"]


def test_partial_exit_from_an_overlap_reports_only_the_fence_left(mon):
    mon.ingest_frame(["V-31"], [IN_OVERLAP])
    partial = mon.ingest_frame(["V-31"], [IN_MBNMS])

    assert [(e["type"], e["geofence_id"]) for e in partial] == [
        ("geofence_exit", "elkhorn"),
    ]
    assert mon.inside_of("V-31") == ["mbnms"]


def test_dark_inside_an_overlap_fires_once_per_fence(mon):
    mon.ingest_frame(["V-31"], [IN_OVERLAP])
    compound = mon.ingest_frame(["V-31"], [IN_OVERLAP], ["V-31"])

    assert len(compound) == 2
    assert {e["type"] for e in compound} == {"dark_inside_protected_area"}
    assert sorted(e["geofence_id"] for e in compound) == ["elkhorn", "mbnms"]


def test_swapping_one_fence_for_another_in_a_single_frame(mon):
    """Exit is reported before enter, so the log reads as a departure then an
    arrival rather than the vessel being in two places at once."""
    mon.ingest_frame(["V-31"], [IN_MBNMS])
    swap = mon.ingest_frame(["V-31"], [IN_CABLE])

    assert [(e["type"], e["geofence_id"]) for e in swap] == [
        ("geofence_exit", "mbnms"),
        ("geofence_enter", "trans_pac"),
    ]


# ===========================================================================
# ACCEPTANCE 5: dark while OUTSIDE any fence is not the compound event
# ===========================================================================

def test_dark_in_open_water_is_an_ordinary_event(mon):
    mon.ingest_frame(["V-05"], [FAR_AWAY])
    dark = mon.ingest_frame(["V-05"], [FAR_AWAY], ["V-05"])

    assert len(dark) == 1
    assert dark[0]["type"] == "ais_dark"
    assert dark[0]["type"] != "dark_inside_protected_area"
    assert dark[0]["geofence_id"] == ""
    assert mon.count_of_type("dark_inside_protected_area") == 0


def test_dark_in_open_water_outranked_by_dark_inside_a_sanctuary(mon):
    mon.ingest_frame(["V-05", "V-77"], [FAR_AWAY, IN_MBNMS])
    mon.ingest_frame(["V-05", "V-77"], [FAR_AWAY, IN_MBNMS_2], ["V-05", "V-77"])

    plain = mon.events_of_type("ais_dark")[0]
    compound = mon.events_of_type("dark_inside_protected_area")[0]
    assert compound["severity_rank"] > plain["severity_rank"]


def test_dark_then_resume_then_dark_again_gives_two_dark_events(mon):
    for point, dark in ((FAR_AWAY, []), (FAR_AWAY, ["V-05"]),
                        (FAR_AWAY, []), (FAR_AWAY, ["V-05"])):
        mon.ingest_frame(["V-05"], [point], dark)

    assert types_for(mon, "V-05") == ["ais_dark", "ais_resume", "ais_dark"]


def test_dark_resume_dark_inside_a_sanctuary_gives_two_compound_events(mon):
    """A vessel that surfaces briefly and goes dark again has committed two
    separate acts, and the fired-set has to re-arm for the second one."""
    mon.ingest_frame(["V-77"], [IN_MBNMS])
    mon.ingest_frame(["V-77"], [IN_MBNMS], ["V-77"])
    mon.ingest_frame(["V-77"], [IN_MBNMS])          # resurfaces, still inside
    mon.ingest_frame(["V-77"], [IN_MBNMS], ["V-77"])

    assert types_for(mon, "V-77") == [
        "geofence_enter",
        "dark_inside_protected_area",
        "ais_resume",
        "dark_inside_protected_area",
    ]


def test_entering_a_sanctuary_while_already_dark_still_alerts(mon):
    """The compound rule is a condition on (inside AND dark), not a reaction to
    the dark transition, so arriving dark is caught just like going dark."""
    mon.ingest_frame(["V-77"], [OPEN_WATER])
    mon.ingest_frame(["V-77"], [OPEN_WATER], ["V-77"])       # ais_dark, outside
    arriving = mon.ingest_frame(["V-77"], [IN_MBNMS], ["V-77"])

    assert [e["type"] for e in arriving] == [
        "geofence_enter", "dark_inside_protected_area",
    ]
    # And the enter was NOT suppressed or swallowed by the upgrade.
    assert mon.count_of_type("geofence_enter") == 1


def test_leaving_and_re_entering_while_still_dark_alerts_twice(mon):
    mon.ingest_frame(["V-77"], [IN_MBNMS])
    mon.ingest_frame(["V-77"], [IN_MBNMS], ["V-77"])
    mon.ingest_frame(["V-77"], [OPEN_WATER], ["V-77"])       # exit, still dark
    mon.ingest_frame(["V-77"], [IN_MBNMS], ["V-77"])         # back in, still dark

    assert mon.count_of_type("dark_inside_protected_area") == 2
    assert mon.count_of_type("geofence_enter") == 2
    assert mon.count_of_type("geofence_exit") == 1
    # It never resurfaced, so there is no resume line.
    assert mon.count_of_type("ais_resume") == 0


# ===========================================================================
# Event shape, multi-track traversal, and graph state
# ===========================================================================

def test_event_dict_shape(mon):
    mon.ingest_frame(["V-77"], [IN_MBNMS])
    ev = mon.events()[0]
    assert set(ev) == {
        "type", "actor", "geofence_id", "severity", "severity_rank",
        "detail", "frame",
    }
    assert isinstance(ev["type"], str)
    assert isinstance(ev["actor"], str)
    assert isinstance(ev["geofence_id"], str)
    assert isinstance(ev["severity"], str)
    assert isinstance(ev["severity_rank"], int)
    assert isinstance(ev["detail"], str)
    assert "Monterey Bay NMS" in ev["detail"]


def test_event_log_is_ordered_and_matches_the_per_frame_returns(mon):
    a = mon.ingest_frame(["V-77"], [IN_MBNMS])
    b = mon.ingest_frame(["V-77"], [IN_OVERLAP])
    c = mon.ingest_frame(["V-77"], [OPEN_WATER])
    assert mon.events() == a + b + c
    assert mon.event_count() == len(a) + len(b) + len(c)
    assert [e["frame"] for e in mon.events()] == sorted(
        e["frame"] for e in mon.events()
    )


def test_three_vessels_are_all_updated_by_one_traversal(mon):
    mon.ingest_frame(["V-77", "V-31", "V-05"],
                     [OPEN_WATER, OPEN_WATER, FAR_AWAY])
    fired = mon.ingest_frame(["V-77", "V-31", "V-05"],
                             [IN_MBNMS, IN_OVERLAP, IN_CABLE],
                             ["V-31"])

    assert mon.watch_count() == 3
    by_actor = {}
    for e in fired:
        by_actor.setdefault(e["actor"], []).append(e["type"])
    assert by_actor["V-77"] == ["geofence_enter"]
    assert by_actor["V-05"] == ["geofence_enter"]
    assert by_actor["V-31"] == [
        "geofence_enter", "geofence_enter",
        "dark_inside_protected_area", "dark_inside_protected_area",
    ]

    assert mon.inside_of("V-77") == ["mbnms"]
    assert mon.inside_of("V-31") == ["elkhorn", "mbnms"]
    assert mon.inside_of("V-05") == ["trans_pac"]
    assert mon.is_dark("V-31") is True
    assert mon.is_dark("V-77") is False


def test_an_unobserved_track_keeps_its_state(mon):
    """A track missing from a frame was not seen; that is not evidence it left."""
    mon.ingest_frame(["V-77", "V-31"], [IN_MBNMS, OPEN_WATER])
    quiet = mon.ingest_frame(["V-31"], [OPEN_WATER])

    assert quiet == []
    assert mon.inside_of("V-77") == ["mbnms"]
    assert mon.count_of_type("geofence_exit") == 0


def test_reset_clears_events_and_the_watch_graph(mon):
    mon.ingest_frame(["V-77"], [IN_MBNMS])
    assert mon.event_count() == 1
    assert mon.watch_count() == 1

    mon.reset()
    assert mon.events() == []
    assert mon.event_count() == 0
    assert mon.watch_count() == 0
    assert mon.inside_of("V-77") == []

    # And a fresh run re-alerts on a fence the previous run was already inside.
    assert len(mon.ingest_frame(["V-77"], [IN_MBNMS])) == 1


def test_event_lines_render_ascii_only(mon):
    """The demo has to print on a cp1252 Windows console."""
    mon.ingest_frame(["V-77"], [IN_MBNMS])
    mon.ingest_frame(["V-77"], [IN_MBNMS], ["V-77"])
    lines = mon.event_lines()
    assert len(lines) == 2
    for line in lines:
        line.encode("ascii")           # raises if anything non-ASCII crept in
        assert line.startswith("[geofence]")

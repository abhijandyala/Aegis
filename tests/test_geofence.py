"""Tests for tracker.geofence -- pure ENU-metre containment geometry.

All coordinates here are ENU metres unless the test is explicitly about
lonlat_to_enu. Transition and event-emission behavior is covered separately.
"""

import json
import os
import time

import numpy as np
import pytest

from tracker import contracts
from tracker.geofence import (
    CANONICAL_KINDS,
    EARTH_RADIUS_M,
    KIND_ALIASES,
    Geofence,
    GeofenceContractError,
    GeofenceIndex,
    build_index,
    corridor_fence,
    geofence_from_mapping,
    geofence_from_ring,
    load_geojson_fences,
    lonlat_to_enu,
    normalize_kind,
    polygon_fence,
)


def _square(cx, cy, half):
    """Axis-aligned square ring centred on (cx, cy)."""
    return [
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
    ]


# --------------------------------------------------------------------------
# Containment correctness, including the STRtree argument-order trap
# --------------------------------------------------------------------------

def _nested_index():
    """A small fence wholly inside a big one -- the argument-order canary."""
    big = polygon_fence("big", "mpa", "Big MPA", _square(0.0, 0.0, 5000.0))
    small = polygon_fence("small", "port", "Small Port", _square(0.0, 0.0, 500.0))
    return GeofenceIndex([big, small])


def test_nested_fences_point_in_both():
    # Dead centre: inside the small fence AND the big one that encloses it.
    assert _nested_index().query([0.0, 0.0]) == ["big", "small"]


def test_nested_fences_point_in_big_only():
    # 3 km out: still in the big fence, well clear of the small one. Together
    # with the test above this fails if the predicate is reversed -- a flipped
    # "fence contains point" -> "point contains fence" yields [] for both, and
    # any symmetric predicate (e.g. intersects on the wrong operand) cannot
    # produce ["big", "small"] here and ["big"] there.
    assert _nested_index().query([3000.0, 0.0]) == ["big"]


def test_point_outside_everything():
    assert _nested_index().query([99999.0, 99999.0]) == []


def test_query_many_matches_query_pointwise():
    idx = _nested_index()
    pts = np.array([[0.0, 0.0], [3000.0, 0.0], [99999.0, 0.0]])
    assert idx.query_many(pts) == [["big", "small"], ["big"], []]
    assert idx.query_many(pts) == [idx.query(p) for p in pts]


def test_ids_are_sorted_not_build_order():
    # Build order puts "zulu" first; the contract is sorted id order.
    a = polygon_fence("zulu", "mpa", "Z", _square(0.0, 0.0, 1000.0))
    b = polygon_fence("alpha", "mpa", "A", _square(0.0, 0.0, 1000.0))
    assert GeofenceIndex([a, b]).query([0.0, 0.0]) == ["alpha", "zulu"]


def test_len_and_fences_property():
    idx = _nested_index()
    assert len(idx) == 2
    assert [f.id for f in idx.fences] == ["big", "small"]
    assert isinstance(idx.fences, tuple)
    assert all(isinstance(f, Geofence) for f in idx.fences)


# --------------------------------------------------------------------------
# Boundary semantics -- measured, not guessed
# --------------------------------------------------------------------------

def test_point_exactly_on_boundary_is_outside():
    # PINNED BEHAVIOUR (shapely 2.1.2, OGC "contains" semantics): a point lying
    # exactly on the polygon edge is NOT contained -- contains/within exclude
    # the boundary, since the point's only intersection with the fence is with
    # the fence's boundary, not its interior. Verified, not assumed. We keep
    # this rather than switching to "intersects" because the frozen API says
    # containment; a vessel exactly on a sanctuary line is not yet inside it.
    idx = GeofenceIndex([polygon_fence("f", "mpa", "F", _square(0.0, 0.0, 1000.0))])
    assert idx.query([1000.0, 0.0]) == []       # on the edge -> outside
    assert idx.query([1000.0, 1000.0]) == []    # on a vertex -> outside
    assert idx.query([999.9, 0.0]) == ["f"]     # just inside -> inside


# --------------------------------------------------------------------------
# Buffering: corridors and dilated polygons
# --------------------------------------------------------------------------

def test_corridor_fence_buffer_width():
    coords = [[0.0, 0.0], [10000.0, 0.0]]  # cable running due east
    off_axis = [5000.0, 100.0]             # 100 m off the route

    wide = GeofenceIndex([corridor_fence("c", "Cable", coords, buffer_m=500.0)])
    narrow = GeofenceIndex([corridor_fence("c", "Cable", coords, buffer_m=50.0)])

    assert wide.query(off_axis) == ["c"]
    assert narrow.query(off_axis) == []
    # The route itself is inside either way.
    assert narrow.query([5000.0, 0.0]) == ["c"]


def test_corridor_fence_defaults_to_cable_kind():
    f = corridor_fence("c", "Cable", [[0.0, 0.0], [1000.0, 0.0]], buffer_m=200.0)
    assert f.kind == "cable"
    assert f.buffer_m == 200.0
    # Buffering happened at construction: the stored geometry is already areal.
    assert f.geom.area > 0.0


def test_polygon_fence_buffer_dilates():
    coords = _square(0.0, 0.0, 1000.0)
    just_outside = [1200.0, 0.0]

    plain = GeofenceIndex([polygon_fence("p", "mpa", "P", coords)])
    dilated = GeofenceIndex([polygon_fence("p", "mpa", "P", coords, buffer_m=500.0)])

    assert plain.query(just_outside) == []
    assert dilated.query(just_outside) == ["p"]
    assert dilated.get("p").buffer_m == 500.0


def test_polygon_fence_zero_buffer_preserves_area():
    coords = _square(0.0, 0.0, 1000.0)
    f = polygon_fence("p", "mpa", "P", coords)
    assert f.geom.area == pytest.approx(2000.0 * 2000.0)
    assert f.buffer_m == 0.0


# --------------------------------------------------------------------------
# Degenerate cases
# --------------------------------------------------------------------------

def test_empty_index():
    idx = GeofenceIndex([])
    assert len(idx) == 0
    assert idx.fences == ()
    assert idx.query([0.0, 0.0]) == []
    assert idx.query_many(np.array([[0.0, 0.0], [1.0, 1.0]])) == [[], []]


def test_empty_point_array():
    assert _nested_index().query_many(np.empty((0, 2))) == []
    assert GeofenceIndex([]).query_many(np.empty((0, 2))) == []


def test_query_many_length_always_matches_input():
    idx = _nested_index()
    for n in (0, 1, 3, 17):
        pts = np.zeros((n, 2))
        assert len(idx.query_many(pts)) == n


def test_get_missing_id_raises():
    with pytest.raises(KeyError):
        _nested_index().get("no-such-fence")


def test_get_returns_the_fence():
    idx = _nested_index()
    assert idx.get("small").label == "Small Port"
    assert idx.get("small").kind == "port"


# --------------------------------------------------------------------------
# lonlat_to_enu
# --------------------------------------------------------------------------

def test_lonlat_origin_maps_to_zero():
    origin = [-9.5, 38.7]  # off Lisbon
    out = lonlat_to_enu([origin], origin)
    assert out.shape == (1, 2)
    np.testing.assert_allclose(out[0], [0.0, 0.0], atol=1e-9)


def test_earth_radius_is_wgs84_semi_major():
    # The exported constant is load-bearing for the hand-computed metres below.
    assert EARTH_RADIUS_M == 6378137.0


def test_lonlat_known_offset_in_metres():
    origin = [-9.5, 38.7]
    # 0.01 deg north, and 0.01 deg east (which shrinks by cos(lat0)).
    out = lonlat_to_enu([[-9.5, 38.71], [-9.49, 38.7]], origin)

    # Independent expected values, NOT recomputed from the module's own formula
    # (that would pass even if R or the cos factor were wrong in both places):
    #   0.01 deg of latitude  = 6378137 * 0.01 * pi/180            = 1113.2 m
    #   0.01 deg of longitude at 38.7 N = 1113.2 * cos(38.7 deg)   =  868.8 m
    assert out[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert out[0, 1] == pytest.approx(1113.2, abs=2.0)   # north
    assert out[1, 0] == pytest.approx(868.8, abs=2.0)    # east, cos-shrunk
    assert out[1, 1] == pytest.approx(0.0, abs=1e-9)
    # Sanity on the cos(lat0) shrink: east is shorter than north at 38.7 N.
    assert out[1, 0] < out[0, 1]


def test_lonlat_shape_preserved():
    origin = [0.0, 0.0]
    pts = np.array([[0.0, 0.0], [0.1, 0.1], [-0.2, 0.3], [1.0, -1.0]])
    assert lonlat_to_enu(pts, origin).shape == (4, 2)


def test_lonlat_feeds_fences_end_to_end():
    # The realistic path: GeoJSON degrees -> ENU -> fence -> containment.
    origin = [-9.5, 38.7]
    ring = lonlat_to_enu(
        [[-9.55, 38.65], [-9.45, 38.65], [-9.45, 38.75], [-9.55, 38.75]], origin
    )
    idx = GeofenceIndex([polygon_fence("mpa1", "mpa", "Sanctuary", ring.tolist())])
    inside = lonlat_to_enu([[-9.50, 38.70]], origin)[0]
    outside = lonlat_to_enu([[-9.20, 38.70]], origin)[0]
    assert idx.query(inside) == ["mpa1"]
    assert idx.query(outside) == []


# --------------------------------------------------------------------------
# ACCEPTANCE: 40 tracks resolved in well under 1 ms
# --------------------------------------------------------------------------

def _realistic_index():
    """Seven fences of mixed kind, roughly a 100 km scenario box."""
    return GeofenceIndex([
        polygon_fence("mpa_north", "mpa", "North Sanctuary", _square(-20000, 20000, 8000)),
        polygon_fence("mpa_south", "mpa", "South Sanctuary", _square(15000, -25000, 12000)),
        polygon_fence("port_a", "port", "Port A", _square(0, 0, 3000), buffer_m=1000.0),
        polygon_fence("port_b", "port", "Port B", _square(30000, 30000, 2500)),
        polygon_fence("danger_1", "danger", "Firing Range", _square(-30000, -30000, 6000)),
        corridor_fence("cable_e", "East Cable", [[-40000, 0], [40000, 5000]], 750.0),
        corridor_fence("cable_n", "North Cable", [[0, -40000], [5000, 40000]], 500.0),
    ])


def test_query_many_40_points_under_1ms():
    idx = _realistic_index()
    rng = np.random.default_rng(20260726)

    # Half the points are seeded inside real fences so the tree actually has to
    # do containment work and return hits, rather than rejecting on bbox alone.
    seeded = np.array([
        [-20000.0, 20000.0], [-18000.0, 22000.0],   # mpa_north
        [15000.0, -25000.0], [18000.0, -22000.0],   # mpa_south
        [0.0, 0.0], [3500.0, 0.0],                  # port_a (+1 km buffer)
        [30000.0, 30000.0],                         # port_b
        [-30000.0, -30000.0],                       # danger_1
        [0.0, 2500.0], [20000.0, 3750.0],           # cable_e corridor
        [2500.0, 0.0], [1250.0, -20000.0],          # cable_n corridor
        [0.0, 1.0], [1000.0, 1000.0],               # port_a and cable_e
    ])
    scatter = rng.uniform(-50000.0, 50000.0, size=(40 - len(seeded), 2))
    pts = np.vstack([seeded, scatter])
    assert pts.shape == (40, 2)

    # Correctness first: a fast wrong answer is not the acceptance criterion.
    hits = idx.query_many(pts)
    assert len(hits) == 40
    n_hits = sum(len(h) for h in hits)
    assert n_hits >= 12, f"expected the seeded points to hit fences, got {n_hits}"

    # Warm up: first call pays numpy/shapely import-time and branch-predictor
    # costs that are not representative of the steady-state replay loop.
    for _ in range(5):
        idx.query_many(pts)

    # perf_counter ONLY: time.time has ~15 ms resolution on Windows, which is
    # 15x the entire budget -- it would measure nothing but zero.
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        idx.query_many(pts)
        best = min(best, time.perf_counter() - t0)

    assert best < 1e-3, (
        f"query_many for 40 points took {best * 1e6:.1f} us "
        f"(budget 1000.0 us) over 7 fences with {n_hits} hits"
    )


# ==========================================================================
# INTEROP: the three-way contract divergence
# ==========================================================================

# --------------------------------------------------------------------------
# normalize_kind
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("sanctuary", "mpa"),
        ("marine_sanctuary", "mpa"),
        ("mpa", "mpa"),
        ("protected", "mpa"),
        ("reserve", "mpa"),
        ("nms", "mpa"),
        ("restricted", "danger"),
        ("danger", "danger"),
        ("military", "danger"),
        ("exclusion", "danger"),
        ("anchorage", "port"),
        ("port", "port"),
        ("harbor", "port"),
        ("harbour", "port"),
        ("terminal", "port"),
        ("cable", "cable"),
        ("corridor", "cable"),
        ("submarine_cable", "cable"),
        ("pipeline", "cable"),
    ],
)
def test_normalize_kind_every_required_alias(raw, expected):
    assert normalize_kind(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["Marine Sanctuary", "marine-sanctuary", "MARINE_SANCTUARY", "  Marine  Sanctuary  ",
     "marine.sanctuary"],
)
def test_normalize_kind_case_and_separator_tolerant(raw):
    assert normalize_kind(raw) == "mpa"


def test_normalize_kind_canonical_kinds_are_fixed_points():
    # Idempotence: normalizing an already-canonical kind must not move it.
    for k in CANONICAL_KINDS:
        assert normalize_kind(k) == k
        assert normalize_kind(normalize_kind(k)) == k


def test_kind_alias_table_only_targets_canonical_kinds():
    # Guards the table itself: a typo'd value here would put a non-canonical
    # kind into Geofence.kind, which downstream severity rules switch on.
    assert set(KIND_ALIASES.values()) <= set(CANONICAL_KINDS)


def test_normalize_kind_unknown_raises_and_does_not_become_mpa():
    # THE DOCUMENTED DECISION: unknown input raises. It must never silently
    # become "mpa" (that invents a protected area and manufactures the CRITICAL
    # dark-vessel-in-sanctuary alert) nor silently become anything else.
    with pytest.raises(GeofenceContractError, match="unknown geofence kind"):
        normalize_kind("sancturay")          # plausible typo
    with pytest.raises(GeofenceContractError, match="unknown geofence kind"):
        normalize_kind("fishing_zone")       # plausible unmapped vocabulary
    with pytest.raises(GeofenceContractError):
        normalize_kind("")


def test_normalize_kind_explicit_default_escape_hatch():
    # A caller may opt in to the conservative fallback, per call, out loud.
    assert normalize_kind("fishing_zone", default="danger") == "danger"
    assert normalize_kind("fishing_zone", default=None) is None
    # ...and the escape hatch must not affect known values.
    assert normalize_kind("anchorage", default="danger") == "port"


# --------------------------------------------------------------------------
# geofence_from_ring -- Abhi's `ring` field
# --------------------------------------------------------------------------

def test_geofence_from_ring_unclosed_and_closed_agree():
    ring = _square(0.0, 0.0, 1000.0)
    closed = ring + [ring[0]]

    a = geofence_from_ring("f", "sanctuary", "F", ring)
    b = geofence_from_ring("f", "sanctuary", "F", closed)

    assert a.geom.area == pytest.approx(b.geom.area)
    assert a.geom.area == pytest.approx(2000.0 * 2000.0)
    assert a.kind == b.kind == "mpa"        # Abhi's vocabulary, normalized


def test_geofence_from_ring_accepts_tuples_and_ndarray():
    ring_lists = _square(0.0, 0.0, 1000.0)
    ring_tuples = tuple(tuple(p) for p in ring_lists)
    ring_array = np.asarray(ring_lists, dtype=float)

    areas = {
        geofence_from_ring("f", "anchorage", "F", r).geom.area
        for r in (ring_lists, ring_tuples, ring_array)
    }
    assert len(areas) == 1
    assert areas.pop() == pytest.approx(4_000_000.0)


def test_geofence_from_ring_buffer_and_kind():
    f = geofence_from_ring("c", "corridor", "C", _square(0.0, 0.0, 1000.0), 500.0)
    assert f.kind == "cable"
    assert f.buffer_m == 500.0
    assert GeofenceIndex([f]).query([1200.0, 0.0]) == ["c"]


def test_geofence_from_ring_rejects_unknown_kind():
    with pytest.raises(GeofenceContractError):
        geofence_from_ring("f", "kelp_bed", "F", _square(0.0, 0.0, 1000.0))


def test_geofence_from_ring_rejects_degenerate_ring():
    with pytest.raises(GeofenceContractError, match="at least 3"):
        geofence_from_ring("f", "mpa", "F", [[0.0, 0.0], [1.0, 1.0]])


# --------------------------------------------------------------------------
# geofence_from_mapping -- the written contract's dict, and Abhi's dataclass
# --------------------------------------------------------------------------

def test_from_mapping_written_contract_dict():
    # id / type / label / polygon / buffer_m -- the build plan's shape verbatim.
    m = {
        "id": "mpa_01",
        "type": "mpa",
        "label": "North Sanctuary",
        "polygon": _square(0.0, 0.0, 1000.0),
        "buffer_m": 0.0,
    }
    f = geofence_from_mapping(m)
    assert (f.id, f.kind, f.label, f.buffer_m) == ("mpa_01", "mpa", "North Sanctuary", 0.0)
    assert GeofenceIndex([f]).query([0.0, 0.0]) == ["mpa_01"]


def test_from_mapping_written_contract_dict_with_buffer():
    m = {
        "id": "cab_01",
        "type": "cable",
        "label": "Cable",
        "polygon": _square(0.0, 0.0, 1000.0),
        "buffer_m": 500.0,
    }
    f = geofence_from_mapping(m)
    assert f.buffer_m == 500.0
    # buffer_m on raw coordinates dilates: 1200 m out is inside a 1000+500 fence.
    assert GeofenceIndex([f]).query([1200.0, 0.0]) == ["cab_01"]


class _AbhiGeofence:
    """Structural stand-in for data/contracts.py Geofence on the unmerged
    branch: fence_id / name / kind / ring / geojson, and NO buffer_m.

    Deliberately a local stub, not an import -- this module must not depend on
    a teammate's branch being merged, which is the whole point of the adapters.
    """

    __slots__ = ("fence_id", "name", "kind", "ring", "geojson")

    def __init__(self, fence_id, name, kind, ring, geojson=None):
        self.fence_id = fence_id
        self.name = name
        self.kind = kind
        self.ring = tuple(tuple(float(c) for c in p) for p in ring)
        self.geojson = geojson or {}


def test_from_mapping_abhi_dataclass_shape():
    obj = _AbhiGeofence(
        fence_id="f_abhi",
        name="Monterey Bay NMS",
        kind="sanctuary",            # his vocabulary
        ring=_square(0.0, 0.0, 2000.0),
    )
    f = geofence_from_mapping(obj)
    assert f.id == "f_abhi"                     # fence_id -> id
    assert f.label == "Monterey Bay NMS"        # name -> label
    assert f.kind == "mpa"                      # sanctuary -> mpa
    assert f.buffer_m == 0.0                    # absent field -> 0.0
    assert f.geom.area == pytest.approx(4000.0 * 4000.0)
    assert GeofenceIndex([f]).query([0.0, 0.0]) == ["f_abhi"]


def test_from_mapping_abhi_shape_defaults_buffer_from_contracts():
    # UPDATED (was test_from_mapping_abhi_shape_buffer_must_be_passed_for_corridors):
    # that test pinned the exact zero-buffered-cable trap the contracts.py
    # reconciliation exists to close -- his dataclass carries no buffer_m field
    # at all, and defaulting it to a hardcoded 0.0 made a cable layer imported
    # through this shape a zero-area line that could never contain anything.
    # The corrected behaviour: absence now falls back to
    # contracts.default_buffer_m(kind), which is 500.0 for cable.
    obj = _AbhiGeofence("cab", "Cable Corridor", "cable", _square(0.0, 0.0, 1000.0))
    assert geofence_from_mapping(obj).buffer_m == 500.0
    assert GeofenceIndex([geofence_from_mapping(obj)]).query([1200.0, 0.0]) == ["cab"]

    # The buffer_m= argument still overrides the default in either direction.
    assert geofence_from_mapping(obj, buffer_m=0.0).buffer_m == 0.0
    assert GeofenceIndex([geofence_from_mapping(obj, buffer_m=0.0)]).query(
        [1200.0, 0.0]
    ) == []
    widened = geofence_from_mapping(obj, buffer_m=1500.0)
    assert widened.buffer_m == 1500.0
    assert GeofenceIndex([widened]).query([1200.0, 0.0]) == ["cab"]


def test_from_mapping_buffer_argument_overrides_field():
    m = {"id": "f", "kind": "mpa", "label": "F",
         "polygon": _square(0.0, 0.0, 1000.0), "buffer_m": 10.0}
    assert geofence_from_mapping(m, buffer_m=500.0).buffer_m == 500.0


@pytest.mark.parametrize(
    "ring_form",
    [
        _square(0.0, 0.0, 1000.0),                                   # list of lists
        tuple(tuple(p) for p in _square(0.0, 0.0, 1000.0)),          # tuples
        _square(0.0, 0.0, 1000.0) + [_square(0.0, 0.0, 1000.0)[0]],  # closed
        np.asarray(_square(0.0, 0.0, 1000.0), dtype=float),          # ndarray
        [_square(0.0, 0.0, 1000.0)],                                 # GeoJSON rings
    ],
)
def test_from_mapping_geometry_forms(ring_form):
    f = geofence_from_mapping({"id": "f", "kind": "mpa", "label": "F",
                               "coordinates": ring_form})
    assert f.geom.area == pytest.approx(4_000_000.0)


def test_from_mapping_missing_geometry_names_the_fields():
    with pytest.raises(GeofenceContractError) as ei:
        geofence_from_mapping({"id": "f", "kind": "mpa", "label": "F"})
    msg = str(ei.value)
    assert "missing a geometry" in msg
    assert "'f'" in msg                       # names the offending fence
    for field in ("geom", "polygon", "ring", "coordinates"):
        assert field in msg                   # names what it looked for


def test_from_mapping_missing_id_names_the_fields():
    with pytest.raises(GeofenceContractError) as ei:
        geofence_from_mapping({"kind": "mpa", "label": "F",
                               "polygon": _square(0.0, 0.0, 1000.0)})
    msg = str(ei.value)
    assert "missing an id" in msg
    for field in ("id", "fence_id", "geofence_id"):
        assert field in msg


def test_from_mapping_missing_kind_raises_rather_than_defaulting():
    with pytest.raises(GeofenceContractError, match="missing a kind"):
        geofence_from_mapping({"id": "f", "label": "F",
                               "polygon": _square(0.0, 0.0, 1000.0)})


def test_from_mapping_label_falls_back_to_id():
    f = geofence_from_mapping({"id": "f_only", "kind": "mpa",
                               "polygon": _square(0.0, 0.0, 1000.0)})
    assert f.label == "f_only"


def test_from_mapping_line_geometry_requires_buffer():
    from shapely.geometry import LineString

    line = LineString([(0.0, 0.0), (10000.0, 0.0)])
    with pytest.raises(GeofenceContractError, match="zero-buffered line"):
        geofence_from_mapping({"id": "c", "kind": "cable", "label": "C",
                               "geom": line})
    f = geofence_from_mapping({"id": "c", "kind": "cable", "label": "C",
                               "geom": line}, buffer_m=500.0)
    assert GeofenceIndex([f]).query([5000.0, 100.0]) == ["c"]


# --------------------------------------------------------------------------
# Round trip: my own Geofence out and back in, unchanged
# --------------------------------------------------------------------------

def test_round_trip_through_mapping_preserves_everything():
    original = corridor_fence("c", "East Cable", [[0.0, 0.0], [10000.0, 0.0]], 500.0)
    as_dict = {
        "id": original.id,
        "kind": original.kind,
        "label": original.label,
        "geom": original.geom,          # already-buffered shapely geometry
        "buffer_m": original.buffer_m,
    }
    back = geofence_from_mapping(as_dict)

    assert (back.id, back.kind, back.label, back.buffer_m) == (
        original.id, original.kind, original.label, original.buffer_m
    )
    # NOT re-buffered: geom is already dilated and buffer_m is provenance only.
    # A double buffer would inflate the area by ~2x here and silently widen
    # every corridor on every hop through the adapter.
    assert back.geom.area == pytest.approx(original.geom.area)
    assert back.geom.equals(original.geom)


def test_round_trip_polygon_fence_with_buffer_not_double_buffered():
    original = polygon_fence("p", "mpa", "P", _square(0.0, 0.0, 1000.0), buffer_m=500.0)
    back = geofence_from_mapping({
        "id": "p", "kind": "mpa", "label": "P",
        "geom": original.geom, "buffer_m": 500.0,
    })
    assert back.geom.area == pytest.approx(original.geom.area)
    # 2100 m out is beyond 1000 + 500; a double buffer would wrongly contain it.
    assert GeofenceIndex([back]).query([2100.0, 0.0]) == []


# --------------------------------------------------------------------------
# GeoJSON loading -- synthetic files (holes, lines, shapes of document)
# --------------------------------------------------------------------------

def _write_geojson(tmp_path, name, doc):
    p = tmp_path / name
    p.write_text(json.dumps(doc), encoding="utf-8")
    return str(p)


def _lonlat_square(clon, clat, half_deg):
    return [
        [clon - half_deg, clat - half_deg],
        [clon + half_deg, clat - half_deg],
        [clon + half_deg, clat + half_deg],
        [clon - half_deg, clat + half_deg],
        [clon - half_deg, clat - half_deg],
    ]


ORIGIN = [-122.0, 36.7]


def test_load_geojson_polygon_with_hole_excludes_the_hole(tmp_path):
    # THE RING TRAP: rings[1:] are interior boundaries, not extra polygons.
    # Dropping them (or unioning them) yields a fence that wrongly contains
    # points in its hole -- a "sanctuary" covering the harbour it excludes.
    shell = _lonlat_square(-122.0, 36.7, 0.20)
    hole = _lonlat_square(-122.0, 36.7, 0.05)
    path = _write_geojson(tmp_path, "holey.geojson", {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"kind": "sanctuary", "name": "Holey Sanctuary"},
            "geometry": {"type": "Polygon", "coordinates": [shell, hole]},
        }],
    })

    fences = load_geojson_fences(path, ORIGIN)
    assert len(fences) == 1
    assert fences[0].kind == "mpa"
    assert fences[0].label == "Holey Sanctuary"

    idx = GeofenceIndex(fences)
    in_hole = lonlat_to_enu([[-122.0, 36.7]], ORIGIN)[0]          # dead centre
    in_ring = lonlat_to_enu([[-122.0, 36.82]], ORIGIN)[0]         # shell, not hole
    outside = lonlat_to_enu([[-122.0, 37.5]], ORIGIN)[0]

    assert idx.query(in_hole) == []           # the hole is NOT inside
    assert idx.query(in_ring) == [fences[0].id]
    assert idx.query(outside) == []

    # And the area is shell-minus-hole, not shell.
    shell_only = load_geojson_fences(
        _write_geojson(tmp_path, "solid.geojson", {
            "type": "Feature",
            "properties": {"kind": "sanctuary"},
            "geometry": {"type": "Polygon", "coordinates": [shell]},
        }),
        ORIGIN,
    )[0]
    assert fences[0].geom.area < shell_only.geom.area


def test_load_geojson_elevation_ordinate_is_dropped_not_reinterpreted(tmp_path):
    # GeoJSON positions may legally be [lon, lat, alt]. lonlat_to_enu reshapes to
    # (-1, 2), which for an (N, 3) ring does NOT raise -- it silently rewrites
    # the buffer into garbage that still has positive area. Pin that the third
    # ordinate is dropped and the fence matches its 2D twin exactly.
    ring2d = _lonlat_square(-122.0, 36.7, 0.1)
    ring3d = [[lon, lat, 17.5] for lon, lat in ring2d]

    flat = load_geojson_fences(
        _write_geojson(tmp_path, "flat.geojson", {
            "type": "Feature", "properties": {"kind": "mpa"},
            "geometry": {"type": "Polygon", "coordinates": [ring2d]}}),
        ORIGIN)[0]
    tall = load_geojson_fences(
        _write_geojson(tmp_path, "tall.geojson", {
            "type": "Feature", "properties": {"kind": "mpa"},
            "geometry": {"type": "Polygon", "coordinates": [ring3d]}}),
        ORIGIN)[0]

    assert tall.geom.area == pytest.approx(flat.geom.area)
    assert tall.geom.equals(flat.geom)


def test_omar_pack_layer_entry_is_a_file_reference_not_a_fence():
    # Omar's pack.json geofence_layers entries are {"file", "kind"} references,
    # verified against origin/omar/scenario-packs. They are NOT Geofences: no
    # id, no geometry. The adapter must refuse them loudly rather than invent a
    # fence -- the pack loader is supposed to call load_geojson_fences instead.
    entry = {"file": "layers/monterey_bay_nms.geojson", "kind": "sanctuary"}
    with pytest.raises(GeofenceContractError, match="missing an id"):
        geofence_from_mapping(entry)


def test_load_geojson_bare_geometry_and_single_feature(tmp_path):
    ring = _lonlat_square(-122.0, 36.7, 0.1)
    bare = _write_geojson(tmp_path, "bare.geojson",
                          {"type": "Polygon", "coordinates": [ring]})
    feat = _write_geojson(tmp_path, "feat.geojson", {
        "type": "Feature", "properties": {}, "geometry":
            {"type": "Polygon", "coordinates": [ring]},
    })

    a = load_geojson_fences(bare, ORIGIN, kind="mpa")
    b = load_geojson_fences(feat, ORIGIN, kind="mpa")
    assert len(a) == len(b) == 1
    assert a[0].geom.area == pytest.approx(b[0].geom.area)
    # Label falls back to the filename stem when properties say nothing.
    assert a[0].label == "bare"
    assert b[0].label == "feat"


def test_load_geojson_multipolygon_is_one_fence(tmp_path):
    # DOCUMENTED CHOICE: one Geofence per feature, parts kept in a MultiPolygon.
    path = _write_geojson(tmp_path, "multi.geojson", {
        "type": "Feature",
        "properties": {"kind": "mpa", "name": "Two Part MPA", "id": "two_part"},
        "geometry": {"type": "MultiPolygon", "coordinates": [
            [_lonlat_square(-122.0, 36.7, 0.05)],
            [_lonlat_square(-121.5, 36.7, 0.05)],
        ]},
    })
    fences = load_geojson_fences(path, ORIGIN)
    assert len(fences) == 1
    assert fences[0].id == "two_part"
    assert fences[0].geom.geom_type == "MultiPolygon"

    idx = GeofenceIndex(fences)
    # Both parts answer with the SAME fence id -- that is the point of the
    # choice: crossing between parts must not look like leaving one fence.
    assert idx.query(lonlat_to_enu([[-122.0, 36.7]], ORIGIN)[0]) == ["two_part"]
    assert idx.query(lonlat_to_enu([[-121.5, 36.7]], ORIGIN)[0]) == ["two_part"]
    assert idx.query(lonlat_to_enu([[-121.75, 36.7]], ORIGIN)[0]) == []


def _line_layer(tmp_path, name="cable.geojson", multi=False):
    coords = [[-122.1, 36.7], [-121.9, 36.7]]
    geom = ({"type": "MultiLineString", "coordinates": [coords]} if multi
            else {"type": "LineString", "coordinates": coords})
    return _write_geojson(tmp_path, name, {
        "type": "FeatureCollection",
        "features": [{"type": "Feature",
                      "properties": {"id": "cable_1", "name": "Cable One"},
                      "geometry": geom}],
    })


@pytest.mark.parametrize("multi", [False, True])
def test_load_geojson_line_with_zero_buffer_raises(tmp_path, multi):
    # SILENT-FAILURE TRAP: a zero-buffered line has no area, so it can never
    # contain anything and the layer would never fire. Refuse it loudly.
    path = _line_layer(tmp_path, f"cable_{multi}.geojson", multi=multi)
    with pytest.raises(GeofenceContractError, match="never"):
        load_geojson_fences(path, ORIGIN)
    with pytest.raises(GeofenceContractError, match="default_buffer_m"):
        load_geojson_fences(path, ORIGIN, default_buffer_m=0.0)


@pytest.mark.parametrize("multi", [False, True])
def test_load_geojson_line_with_buffer_contains_nearby_point(tmp_path, multi):
    path = _line_layer(tmp_path, f"cable_b_{multi}.geojson", multi=multi)
    fences = load_geojson_fences(path, ORIGIN, default_buffer_m=800.0)
    assert len(fences) == 1
    assert fences[0].kind == "cable"      # geometry implies the kind for lines
    assert fences[0].buffer_m == 800.0
    assert fences[0].geom.area > 0.0

    idx = GeofenceIndex(fences)
    on_route = lonlat_to_enu([[-122.0, 36.7]], ORIGIN)[0]
    nearby = np.array([0.0, 400.0])       # 400 m off a route buffered by 800 m
    far = np.array([0.0, 5000.0])
    assert idx.query(on_route) == ["cable_1"]
    assert idx.query(nearby) == ["cable_1"]
    assert idx.query(far) == []


def test_load_geojson_mixed_collection_only_lines_need_buffer(tmp_path):
    # The buffer requirement is per-geometry, not per-file.
    path = _write_geojson(tmp_path, "mixed.geojson", {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"kind": "sanctuary", "id": "poly"},
             "geometry": {"type": "Polygon",
                          "coordinates": [_lonlat_square(-122.0, 36.7, 0.1)]}},
            {"type": "Feature", "properties": {"id": "line"},
             "geometry": {"type": "LineString",
                          "coordinates": [[-122.5, 36.9], [-122.2, 36.9]]}},
        ],
    })
    with pytest.raises(GeofenceContractError):
        load_geojson_fences(path, ORIGIN)
    fences = load_geojson_fences(path, ORIGIN, default_buffer_m=500.0)
    assert [(f.id, f.kind) for f in fences] == [("poly", "mpa"), ("line", "cable")]


def test_load_geojson_polygon_without_kind_raises(tmp_path):
    path = _write_geojson(tmp_path, "nokind.geojson", {
        "type": "Feature", "properties": {"source": "somewhere"},
        "geometry": {"type": "Polygon",
                     "coordinates": [_lonlat_square(-122.0, 36.7, 0.1)]},
    })
    # Never guessed -- not from the filename, not from an unrelated property.
    with pytest.raises(GeofenceContractError, match="declares no kind"):
        load_geojson_fences(path, ORIGIN)
    assert load_geojson_fences(path, ORIGIN, kind="mpa")[0].kind == "mpa"


def test_load_geojson_unknown_property_kind_raises(tmp_path):
    path = _write_geojson(tmp_path, "badkind.geojson", {
        "type": "Feature", "properties": {"kind": "kelp_forest"},
        "geometry": {"type": "Polygon",
                     "coordinates": [_lonlat_square(-122.0, 36.7, 0.1)]},
    })
    with pytest.raises(GeofenceContractError, match="unknown geofence kind"):
        load_geojson_fences(path, ORIGIN)


def test_load_geojson_duplicate_ids_are_disambiguated(tmp_path):
    feat = {"type": "Feature", "properties": {"id": "same", "kind": "mpa"},
            "geometry": {"type": "Polygon",
                         "coordinates": [_lonlat_square(-122.0, 36.7, 0.1)]}}
    path = _write_geojson(tmp_path, "dupes.geojson",
                          {"type": "FeatureCollection", "features": [feat, feat]})
    fences = load_geojson_fences(path, ORIGIN)
    ids = [f.id for f in fences]
    assert len(set(ids)) == 2, ids          # GeofenceIndex.get must stay unambiguous
    assert ids[0] == "same"


def test_load_geojson_unsupported_geometry_type_raises(tmp_path):
    path = _write_geojson(tmp_path, "pt.geojson", {
        "type": "Feature", "properties": {"kind": "port"},
        "geometry": {"type": "Point", "coordinates": [-122.0, 36.7]},
    })
    with pytest.raises(GeofenceContractError, match="unsupported"):
        load_geojson_fences(path, ORIGIN)


# --------------------------------------------------------------------------
# GeoJSON loading -- Omar's REAL files.
# They are gitignored, so every one of these is skipped when absent.
# --------------------------------------------------------------------------

_GEO_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "geo"
)


def _geo(name):
    return os.path.join(_GEO_DIR, name)


def _requires(name):
    return pytest.mark.skipif(
        not os.path.exists(_geo(name)),
        reason=f"geo/{name} is gitignored and not present on this machine",
    )


# A point in Monterey Bay water, well inside the sanctuary, and a point in the
# open Pacific ~450 km offshore. Hardcoded lon/lat literals, verified once
# against the loaded geometry -- deriving them from the geometry itself (a
# centroid, say) would assert nothing.
MBNMS_INSIDE_LONLAT = [-122.0, 36.7]
OPEN_OCEAN_LONLAT = [-127.0, 36.0]


@_requires("monterey_bay_nms_boundary.geojson")
def test_load_real_monterey_bay_sanctuary():
    fences = load_geojson_fences(
        _geo("monterey_bay_nms_boundary.geojson"), ORIGIN, kind="mpa"
    )
    # One MultiPolygon feature -> exactly one fence, by the documented choice.
    assert len(fences) == 1
    f = fences[0]
    assert f.kind == "mpa"
    assert f.geom.geom_type == "MultiPolygon"

    # INDEPENDENT GROUND TRUTH: the file's own properties carry AREA_KM.
    # Comparing against it (rather than against our own recomputation) is what
    # catches dropped holes and mis-nested rings, which would miss badly.
    with open(_geo("monterey_bay_nms_boundary.geojson"), encoding="utf-8") as fh:
        area_km2_truth = json.load(fh)["features"][0]["properties"]["AREA_KM"]
    area_km2 = f.geom.area / 1e6
    assert area_km2 == pytest.approx(area_km2_truth, rel=0.05), (
        f"computed {area_km2:.1f} km^2 vs the file's own {area_km2_truth:.1f} km^2"
    )

    idx = GeofenceIndex(fences)
    assert idx.query(lonlat_to_enu([MBNMS_INSIDE_LONLAT], ORIGIN)[0]) == [f.id]
    assert idx.query(lonlat_to_enu([OPEN_OCEAN_LONLAT], ORIGIN)[0]) == []


@_requires("ca_coastline_10m.geojson")
def test_load_real_ca_coastline():
    # Land, not a protected area: loaded as "danger" because for a vessel the
    # coastline is a grounding hazard. Explicit kind= precisely because the
    # file's properties (source/region) declare nothing the loader may guess at.
    fences = load_geojson_fences(_geo("ca_coastline_10m.geojson"), ORIGIN,
                                 kind="danger")
    assert len(fences) == 2              # 2 MultiPolygon features
    assert {f.kind for f in fences} == {"danger"}
    assert all(f.geom.area > 0.0 for f in fences)
    assert len({f.id for f in fences}) == 2

    # Open ocean is not land.
    idx = GeofenceIndex(fences)
    assert idx.query(lonlat_to_enu([OPEN_OCEAN_LONLAT], ORIGIN)[0]) == []


@_requires("ca_submarine_cables.geojson")
def test_load_real_submarine_cables():
    path = _geo("ca_submarine_cables.geojson")
    # It is a line layer, so zero buffer must be refused, not silently loaded.
    with pytest.raises(GeofenceContractError):
        load_geojson_fences(path, ORIGIN)

    fences = load_geojson_fences(path, ORIGIN, default_buffer_m=500.0)
    assert len(fences) >= 10
    assert {f.kind for f in fences} == {"cable"}     # inferred from geometry
    assert all(f.buffer_m == 500.0 for f in fences)
    assert all(f.geom.area > 0.0 for f in fences)    # buffered, so areal
    assert len({f.id for f in fences}) == len(fences)

    # A wider buffer strictly grows every corridor.
    wide = load_geojson_fences(path, ORIGIN, default_buffer_m=2000.0)
    for narrow_f, wide_f in zip(fences, wide):
        assert wide_f.geom.area > narrow_f.geom.area


@_requires("port_of_sf_geofence.geojson")
def test_load_official_port_of_sf_reference():
    with open(_geo("port_of_sf_geofence.geojson"), encoding="utf-8") as handle:
        feature = json.load(handle)["features"][0]

    assert feature["geometry"]["type"] == "Point"
    assert feature["properties"]["PORT_NAME"] == "SAN FRANCISCO"
    assert feature["properties"]["INDEX_NO"] == 16300
    assert feature["properties"]["source"] == "NGA World Port Index FeatureServer"
    lon, lat = feature["geometry"]["coordinates"]
    assert abs(lon - -122.416666666833) < 1e-10
    assert abs(lat - 37.8166666670331) < 1e-10


@_requires("monterey_bay_nms_boundary.geojson")
@_requires("ca_submarine_cables.geojson")
def test_real_layers_compose_in_one_index():
    # The demo's actual shape: several real layers, one index, one query.
    fences = (
        load_geojson_fences(_geo("monterey_bay_nms_boundary.geojson"), ORIGIN,
                            kind="mpa")
        + load_geojson_fences(_geo("ca_submarine_cables.geojson"), ORIGIN,
                              default_buffer_m=500.0)
    )
    idx = GeofenceIndex(fences)
    assert len(idx) == len(fences)
    hits = idx.query(lonlat_to_enu([MBNMS_INSIDE_LONLAT], ORIGIN)[0])
    assert len(hits) >= 1
    assert all(idx.get(h) is not None for h in hits)
    assert idx.query(lonlat_to_enu([OPEN_OCEAN_LONLAT], ORIGIN)[0]) == []


# ==========================================================================
# CONTRACTS RECONCILIATION -- geofence.py and tracker.contracts must agree
# ==========================================================================

# --------------------------------------------------------------------------
# "land" is a real fifth kind now, not a workaround
# --------------------------------------------------------------------------

@_requires("ca_coastline_10m.geojson")
def test_load_real_ca_coastline_as_land_kind():
    # THE GAP THIS CLOSES: geofence.CANONICAL_KINDS used to be the old 4-tuple
    # and normalize_kind raised on "land"/"coastline", so a coastline layer had
    # to be forced through as kind="danger" as a workaround. It now comes from
    # tracker.contracts (a 5-tuple including "land") and this succeeds directly.
    fences = load_geojson_fences(_geo("ca_coastline_10m.geojson"), ORIGIN, kind="land")
    assert len(fences) == 2
    assert {f.kind for f in fences} == {"land"}
    assert all(f.geom.area > 0.0 for f in fences)
    assert len({f.id for f in fences}) == 2

    idx = GeofenceIndex(fences)
    assert idx.query(lonlat_to_enu([OPEN_OCEAN_LONLAT], ORIGIN)[0]) == []


@_requires("ca_coastline_10m.geojson")
def test_load_real_ca_coastline_danger_workaround_still_works():
    # The old workaround (kind="danger") must keep working -- additive only.
    fences = load_geojson_fences(_geo("ca_coastline_10m.geojson"), ORIGIN,
                                 kind="danger")
    assert {f.kind for f in fences} == {"danger"}


def test_kind_vocabulary_includes_land_and_all_its_aliases():
    assert "land" in CANONICAL_KINDS
    for alias in ("land", "coastline", "coast", "shore", "shoreline",
                  "landmask", "terrain"):
        assert normalize_kind(alias) == "land"


# --------------------------------------------------------------------------
# GeofenceContractError isa contracts.ContractError
# --------------------------------------------------------------------------

def test_geofence_contract_error_is_a_contracts_contract_error():
    assert issubclass(GeofenceContractError, contracts.ContractError)
    assert issubclass(GeofenceContractError, ValueError)

    with pytest.raises(contracts.ContractError):
        normalize_kind("not_a_real_kind")

    with pytest.raises(contracts.ContractError):
        geofence_from_mapping({"kind": "mpa", "label": "F",
                               "polygon": _square(0.0, 0.0, 1000.0)})


# --------------------------------------------------------------------------
# geofence_from_mapping: buffer_m aliases from contracts.FIELD_ALIASES
# --------------------------------------------------------------------------

@pytest.mark.parametrize("buffer_key", ["buffer_m", "buffer", "bufferMeters"])
def test_from_mapping_buffer_field_aliases(buffer_key):
    m = {"id": "cab", "kind": "cable", "label": "Cable",
         "polygon": _square(0.0, 0.0, 1000.0), buffer_key: 500.0}
    f = geofence_from_mapping(m)
    assert f.buffer_m == 500.0
    assert GeofenceIndex([f]).query([1200.0, 0.0]) == ["cab"]


# --------------------------------------------------------------------------
# geofence_from_mapping: default_buffer_m fallback, not a hardcoded 0.0
# --------------------------------------------------------------------------

def test_from_mapping_no_buffer_field_defaults_by_kind():
    cable = geofence_from_mapping({"id": "cab", "kind": "cable", "label": "Cable",
                                   "polygon": _square(0.0, 0.0, 1000.0)})
    assert cable.buffer_m == 500.0                    # contracts.default_buffer_m("cable")
    assert GeofenceIndex([cable]).query([1200.0, 0.0]) == ["cab"]

    mpa = geofence_from_mapping({"id": "m", "kind": "mpa", "label": "MPA",
                                 "polygon": _square(0.0, 0.0, 1000.0)})
    assert mpa.buffer_m == 0.0                        # contracts.default_buffer_m("mpa")
    assert GeofenceIndex([mpa]).query([1200.0, 0.0]) == []


def test_from_mapping_raw_line_geometry_still_requires_explicit_buffer():
    # The per-kind default is deliberately NOT applied when the caller hands
    # in a raw, zero-area shapely LineString -- that is still a REQUIRED,
    # explicit ask (pinned by test_from_mapping_line_geometry_requires_buffer
    # above). Confirms the default fallback (previous test) does not leak into
    # this stricter path just because the kind is "cable".
    from shapely.geometry import LineString

    line = LineString([(0.0, 0.0), (10000.0, 0.0)])
    with pytest.raises(GeofenceContractError, match="zero-area"):
        geofence_from_mapping({"id": "c", "kind": "cable", "label": "C", "geom": line})


# --------------------------------------------------------------------------
# load_geojson_fences: layer namespacing (fixes the id-collision hazard)
# --------------------------------------------------------------------------

def _poly_geojson(tmp_path, name, poly_id):
    ring = _lonlat_square(-122.0, 36.7, 0.1)
    return _write_geojson(tmp_path, name, {
        "type": "Feature",
        "properties": {"kind": "mpa", "POLY_ID": poly_id},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    })


def test_load_geojson_layer_none_keeps_old_unprefixed_ids(tmp_path):
    path = _poly_geojson(tmp_path, "a.geojson", "1")
    fences = load_geojson_fences(path, ORIGIN)          # layer=None, the default
    assert fences[0].id == "1"


def test_load_geojson_layer_namespaces_ids(tmp_path):
    path = _poly_geojson(tmp_path, "b.geojson", "1")
    fences = load_geojson_fences(path, ORIGIN, layer="monterey")
    assert fences[0].id == contracts.namespaced_id("monterey", "1")
    assert fences[0].id == "monterey:1"


# --------------------------------------------------------------------------
# build_index: the real fix for two layers colliding on the same bare id
# --------------------------------------------------------------------------

def test_build_index_namespaces_colliding_bare_ids(tmp_path):
    a = _poly_geojson(tmp_path, "layer_a.geojson", "1")
    b = _poly_geojson(tmp_path, "layer_b.geojson", "1")
    fences_a = load_geojson_fences(a, ORIGIN)   # bare id "1"
    fences_b = load_geojson_fences(b, ORIGIN)   # bare id "1", would collide

    idx = build_index({"layer_a": fences_a, "layer_b": fences_b})
    assert len(idx) == 2
    ids = {f.id for f in idx.fences}
    assert ids == {"layer_a:1", "layer_b:1"}
    # Both distinctly reachable -- not one silently overwriting the other.
    assert idx.get("layer_a:1").id == "layer_a:1"
    assert idx.get("layer_b:1").id == "layer_b:1"


def test_build_index_raises_on_colliding_already_namespaced_ids():
    f1 = polygon_fence("dup:1", "mpa", "A", _square(0.0, 0.0, 1000.0))
    f2 = polygon_fence("dup:1", "port", "B", _square(5000.0, 5000.0, 1000.0))
    with pytest.raises(contracts.ContractError):
        build_index({"x": [f1], "y": [f2]})


def test_build_index_single_layer_matches_plain_index(tmp_path):
    path = _poly_geojson(tmp_path, "solo.geojson", "42")
    fences = load_geojson_fences(path, ORIGIN)
    idx = build_index({"solo": fences})
    assert len(idx) == 1
    assert idx.fences[0].id == "solo:42"

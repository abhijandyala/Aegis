"""Tests for tracker.contracts -- the shared vocabulary and coercion layer.

This module is the single source of truth for the three divergent geofence
contracts, so the tests here are written as *pins on the merge*: each one names
the shape it is defending. Nothing in this file imports shapely or touches disk,
which is itself part of the contract (contracts.py must stay importable from the
pipeline and loader without pulling in geometry).
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from tracker.contracts import (
    CANONICAL_KINDS,
    DEFAULT_BUFFER_M_BY_KIND,
    FIELD_ALIASES,
    KIND_ALIASES,
    ContractError,
    assert_unique_ids,
    dedupe_ids,
    default_buffer_m,
    namespaced_id,
    normalize_kind,
    parse_bbox,
    parse_epoch,
    parse_origin,
    parse_time_window,
    resolve_field,
)

# The real Monterey origin from Omar's scenario packs. Referenced in several
# tests because it is the exact pair whose swap is a ~1000 km silent bug.
MONTEREY = {"lat": 36.75, "lon": -121.95}

# 2024-06-15T06:00:00Z. Hardcoded on purpose: if a local-timezone regression
# creeps into parse_epoch, this literal is what fails.
EPOCH_20240615_0600Z = 1718431200.0


# ==========================================================================
# module hygiene
# ==========================================================================


def test_no_shapely_dependency():
    """contracts.py must stay importable without the geometry stack."""
    import ast

    import tracker.contracts as contracts

    with open(contracts.__file__, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())

    # An AST walk, not a substring grep: the docstrings legitimately *mention*
    # tracker.geofence (explaining why _norm_token is duplicated rather than
    # imported), so a grep would fail on an innocent prose edit at exactly the
    # moment this file matters.
    modules = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    modules |= {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    banned = {"shapely", "tracker"}
    offenders = sorted(m for m in modules if m.split(".")[0] in banned)
    assert not offenders, f"contracts.py must not import {offenders}"


def test_contract_error_is_a_value_error():
    """Both this module and tracker.geofence raise ValueError subclasses, and
    ValueError is the only handler that spans the two unrelated classes."""
    assert issubclass(ContractError, ValueError)


def test_contracts_public_api_is_native_python():
    """The public contract surface is implemented directly in this module."""
    import tracker.contracts as py_contracts

    functions = (
        py_contracts.normalize_kind,
        py_contracts.parse_origin,
        py_contracts.parse_bbox,
        py_contracts.parse_epoch,
        py_contracts.parse_time_window,
        py_contracts.resolve_field,
        py_contracts.default_buffer_m,
        py_contracts.namespaced_id,
        py_contracts.dedupe_ids,
        py_contracts.assert_unique_ids,
    )
    assert all(fn.__module__ == "tracker.contracts" for fn in functions)
    assert py_contracts.normalize_kind("Marine Sanctuary") == "mpa"
    with pytest.raises(py_contracts.ContractError):
        py_contracts.normalize_kind("not-a-real-kind")


# ==========================================================================
# 1. kind vocabulary
# ==========================================================================


def test_canonical_kinds_are_the_five():
    assert CANONICAL_KINDS == ("mpa", "cable", "port", "danger", "land")


def test_land_is_canonical_and_is_the_fifth_kind():
    """"land" is a real addition to the written contract's four kinds: the
    coastline layer is a grounding hazard and clips dark-vessel ellipses."""
    assert "land" in CANONICAL_KINDS
    assert normalize_kind("land") == "land"
    assert normalize_kind("coastline") == "land"


@pytest.mark.parametrize("raw,expected", sorted(KIND_ALIASES.items()))
def test_every_alias_maps_to_its_canonical_kind(raw, expected):
    """Exhaustive over KIND_ALIASES, so adding an entry cannot skip a test."""
    assert normalize_kind(raw) == expected
    assert expected in CANONICAL_KINDS


def test_every_canonical_kind_is_its_own_alias():
    """Canonical values must round-trip; otherwise normalizing twice breaks."""
    for kind in CANONICAL_KINDS:
        assert normalize_kind(kind) == kind


@pytest.mark.parametrize(
    "raw",
    [
        "MPA",
        "Marine Sanctuary",
        "marine-sanctuary",
        "MARINE_SANCTUARY",
        "  marine sanctuary  ",
        "National.Marine.Sanctuary",
        "cable/corridor",
        "Restricted-Area",
        "  COASTLINE ",
    ],
)
def test_case_space_hyphen_dot_slash_variants_normalize(raw):
    assert normalize_kind(raw) in CANONICAL_KINDS


def test_abhis_vocabulary_maps_onto_the_canonical_four():
    """Abhi's kind values are a different vocabulary for the same concepts."""
    assert normalize_kind("sanctuary") == "mpa"
    assert normalize_kind("restricted") == "danger"
    assert normalize_kind("anchorage") == "port"


def test_omars_pack_kind_maps():
    """Omar's layer entries are {"file": ..., "kind": "sanctuary"}."""
    entry = {"file": "layers/monterey_bay_nms.geojson", "kind": "sanctuary"}
    assert normalize_kind(entry["kind"]) == "mpa"


def test_unknown_kind_raises_and_never_guesses():
    with pytest.raises(ContractError) as exc:
        normalize_kind("sancturay")
    msg = str(exc.value)
    assert "sancturay" in msg
    # The message must name the vocabulary and where to extend it.
    assert "KIND_ALIASES" in msg


def test_unknown_kind_does_not_silently_become_mpa():
    """Defaulting to "mpa" would manufacture the demo's top-severity alert."""
    with pytest.raises(ContractError):
        normalize_kind("whatever")


def test_explicit_default_is_returned_verbatim_and_unvalidated():
    assert normalize_kind("nonsense", default="danger") == "danger"
    assert normalize_kind("nonsense", default=None) is None
    # Unvalidated on purpose: the caller owns the fallback.
    assert normalize_kind("nonsense", default="not_a_kind") == "not_a_kind"


def test_default_is_not_used_when_the_kind_is_known():
    assert normalize_kind("sanctuary", default="danger") == "mpa"


# ==========================================================================
# 2. field-name reconciliation
# ==========================================================================


@dataclass(frozen=True)
class AbhiGeofence:
    """Abhi's exact shape (data/contracts.py, unmerged). No buffer_m member."""

    fence_id: str
    name: str
    kind: str
    ring: tuple
    geojson: dict


def test_field_aliases_include_the_canonical_names_themselves():
    for canonical, spellings in FIELD_ALIASES.items():
        assert canonical in spellings, f"{canonical} cannot read its own name"


def test_resolve_field_over_the_written_contract_shape():
    """The build plan: {"id", "type", "label", "polygon", "buffer_m"}."""
    fence = {
        "id": "mb-1",
        "type": "mpa",
        "label": "Monterey Bay NMS",
        "polygon": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]],
        "buffer_m": 250.0,
    }
    assert resolve_field(fence, "id") == "mb-1"
    assert resolve_field(fence, "kind") == "mpa"
    assert resolve_field(fence, "label") == "Monterey Bay NMS"
    assert resolve_field(fence, "geometry") == fence["polygon"]
    assert resolve_field(fence, "buffer_m") == 250.0


def test_resolve_field_over_abhis_exact_shape():
    """fence_id / name / kind / ring, and NO buffer_m -> needs the default."""
    fence = AbhiGeofence(
        fence_id="1",
        name="Monterey Bay",
        kind="sanctuary",
        ring=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        geojson={},
    )
    assert resolve_field(fence, "id") == "1"
    assert resolve_field(fence, "label") == "Monterey Bay"
    assert normalize_kind(resolve_field(fence, "kind")) == "mpa"
    assert resolve_field(fence, "geometry") == fence.ring
    # The whole reason default_buffer_m exists.
    assert resolve_field(fence, "buffer_m", default=None) is None
    with pytest.raises(ContractError):
        resolve_field(fence, "buffer_m")


def test_resolve_field_over_a_plain_object():
    obj = SimpleNamespace(geofence_id="g7", title="Pier 39", category="anchorage")
    assert resolve_field(obj, "id") == "g7"
    assert resolve_field(obj, "label") == "Pier 39"
    assert normalize_kind(resolve_field(obj, "kind")) == "port"


def test_resolve_field_camelcase_spellings_are_reachable():
    """Field names are matched VERBATIM -- token-folding would make camelCase
    entries such as fenceId / bufferMeters unreachable forever."""
    assert resolve_field({"fenceId": "abc"}, "id") == "abc"
    assert resolve_field({"bufferMeters": 500.0}, "buffer_m") == 500.0
    assert resolve_field(SimpleNamespace(fenceId="xyz"), "id") == "xyz"


def test_resolve_field_priority_order():
    """First accepted spelling wins, so a shape carrying both is deterministic."""
    both = {"id": "canonical", "fence_id": "foreign"}
    assert resolve_field(both, "id") == "canonical"
    assert resolve_field({"kind": "mpa", "type": "port"}, "kind") == "mpa"


def test_resolve_field_geometry_accepts_every_real_spelling():
    ring = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    for spelling in ("geometry", "geom", "polygon", "ring", "coordinates", "rings"):
        assert resolve_field({spelling: ring}, "geometry") == ring


def test_resolve_field_missing_raises_naming_every_spelling_tried():
    with pytest.raises(ContractError) as exc:
        resolve_field({"unrelated": 1}, "id")
    msg = str(exc.value)
    assert "'id'" in msg
    for spelling in FIELD_ALIASES["id"]:
        assert spelling in msg, f"error message does not mention {spelling}"


def test_resolve_field_missing_returns_explicit_default():
    assert resolve_field({}, "buffer_m", default=0.0) == 0.0
    assert resolve_field({}, "label", default=None) is None


@pytest.mark.parametrize("falsy", [0, 0.0, "", [], (), {}, False])
def test_falsy_values_are_present_not_absent(falsy):
    """0.0 is a meaningful buffer; "" is a (bad) label; [] is an empty ring.
    Treating any of them as missing would silently substitute a default."""
    assert resolve_field({"buffer_m": falsy}, "buffer_m") == falsy
    assert resolve_field(SimpleNamespace(buffer_m=falsy), "buffer_m") == falsy


def test_zero_buffer_is_not_replaced_by_a_default():
    assert resolve_field({"buffer_m": 0.0}, "buffer_m", default=500.0) == 0.0


def test_none_is_treated_as_absent():
    assert resolve_field({"buffer_m": None}, "buffer_m", default=500.0) == 500.0
    assert resolve_field(SimpleNamespace(id=None, fence_id="f"), "id") == "f"
    with pytest.raises(ContractError):
        resolve_field({"id": None}, "id")


def test_resolve_field_rejects_an_unknown_canonical_name():
    with pytest.raises(ContractError):
        resolve_field({"anything": 1}, "not_a_canonical_field")


# ==========================================================================
# 3. origin parsing -- the ~1000 km silent bug
# ==========================================================================


def test_parse_origin_omars_real_dict():
    """The unambiguous form: the keys say which value is which."""
    assert parse_origin(MONTEREY) == (-121.95, 36.75)


def test_parse_origin_latitude_longitude_spelling():
    assert parse_origin({"latitude": 36.75, "longitude": -121.95}) == (-121.95, 36.75)


def test_parse_origin_dict_key_order_is_irrelevant():
    assert parse_origin({"lon": -121.95, "lat": 36.75}) == (-121.95, 36.75)


def test_parse_origin_equivalent_geojson_sequence_matches_the_dict():
    assert parse_origin([-121.95, 36.75]) == parse_origin(MONTEREY)
    assert parse_origin((-121.95, 36.75)) == (-121.95, 36.75)
    assert parse_origin(np.array([-121.95, 36.75])) == (-121.95, 36.75)


def test_parse_origin_swapped_monterey_raises_and_does_not_swap():
    """(36.75, -121.95) is valid ONLY read as (lat, lon). Silently reordering it
    is exactly how the 1000 km bug ships, so it must raise instead."""
    with pytest.raises(ContractError) as exc:
        parse_origin((36.75, -121.95))
    msg = str(exc.value)
    assert "ambiguous" in msg.lower()
    # The message must tell the caller the fix.
    assert "lat" in msg and "lon" in msg


def test_parse_origin_genuinely_ambiguous_pair_resolves_as_lon_lat():
    """Both readings valid -> GeoJSON order wins, with no error. That is the
    standard, and it is what every .geojson file in geo/ uses."""
    assert parse_origin((10.0, 20.0)) == (10.0, 20.0)
    assert parse_origin([20.0, 10.0]) == (20.0, 10.0)
    assert parse_origin((0.0, 0.0)) == (0.0, 0.0)


def test_parse_origin_boundary_values_are_inclusive():
    assert parse_origin((180.0, 90.0)) == (180.0, 90.0)
    assert parse_origin((-180.0, -90.0)) == (-180.0, -90.0)


@pytest.mark.parametrize("bad", [(200.0, 100.0), (-181.0, 91.0), (400.0, 400.0)])
def test_parse_origin_out_of_range_under_either_reading_raises(bad):
    with pytest.raises(ContractError) as exc:
        parse_origin(bad)
    assert "range" in str(exc.value).lower()


def test_parse_origin_dict_out_of_range_raises():
    with pytest.raises(ContractError):
        parse_origin({"lat": 100.0, "lon": -121.95})
    with pytest.raises(ContractError):
        parse_origin({"lat": 36.75, "lon": -181.0})


def test_parse_origin_dict_may_be_out_of_geojson_order_without_error():
    """A dict is unambiguous, so a latitude larger than any longitude is fine."""
    assert parse_origin({"lat": 89.0, "lon": 1.0}) == (1.0, 89.0)


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"lat": 36.75},
        {"lon": -121.95},
        {"x": 1, "y": 2},
        (1.0,),
        (1.0, 2.0, 3.0),
        "36.75,-121.95",
        None,
        36.75,
    ],
)
def test_parse_origin_rejects_unusable_input(bad):
    with pytest.raises(ContractError):
        parse_origin(bad)


def test_parse_origin_rejects_non_finite():
    with pytest.raises(ContractError):
        parse_origin({"lat": float("nan"), "lon": 0.0})
    with pytest.raises(ContractError):
        parse_origin((float("inf"), 0.0))


def test_parse_origin_always_returns_floats():
    lon, lat = parse_origin({"lat": 36, "lon": -121})
    assert isinstance(lon, float) and isinstance(lat, float)


# ==========================================================================
# 3b. bbox parsing
# ==========================================================================

BBOX_TUPLE = (-122.6, 36.4, -121.6, 37.1)


def test_parse_bbox_omars_mapping_shape():
    got = parse_bbox(
        {"lon_min": -122.6, "lat_min": 36.4, "lon_max": -121.6, "lat_max": 37.1}
    )
    assert got == BBOX_TUPLE


def test_parse_bbox_build_plan_array_shape():
    assert parse_bbox([-122.6, 36.4, -121.6, 37.1]) == BBOX_TUPLE
    assert parse_bbox(np.array([-122.6, 36.4, -121.6, 37.1])) == BBOX_TUPLE


def test_parse_bbox_both_shapes_agree():
    as_map = {"lon_min": -122.6, "lat_min": 36.4, "lon_max": -121.6, "lat_max": 37.1}
    assert parse_bbox(as_map) == parse_bbox(list(BBOX_TUPLE))


def test_parse_bbox_low_latitude_box_is_not_treated_as_ambiguous():
    """No swap heuristic here: a box whose corners are valid both ways round is
    still perfectly legal, and rejecting it would be a false positive."""
    assert parse_bbox([10.0, 20.0, 30.0, 40.0]) == (10.0, 20.0, 30.0, 40.0)


@pytest.mark.parametrize(
    "bad",
    [
        [-121.6, 36.4, -122.6, 37.1],  # lon_min > lon_max
        [-122.6, 37.1, -121.6, 36.4],  # lat_min > lat_max
        [-122.6, 36.4, -122.6, 37.1],  # degenerate in lon
        [-122.6, 36.4, -121.6, 36.4],  # degenerate in lat
    ],
)
def test_parse_bbox_requires_strict_min_less_than_max(bad):
    with pytest.raises(ContractError):
        parse_bbox(bad)


def test_parse_bbox_min_greater_than_max_mapping_shape_raises():
    with pytest.raises(ContractError):
        parse_bbox(
            {"lon_min": 10.0, "lat_min": 0.0, "lon_max": -10.0, "lat_max": 1.0}
        )


@pytest.mark.parametrize(
    "bad",
    [
        [-181.0, 36.4, -121.6, 37.1],
        [-122.6, -91.0, -121.6, 37.1],
        [-122.6, 36.4, 181.0, 37.1],
        [-122.6, 36.4, -121.6, 91.0],
    ],
)
def test_parse_bbox_out_of_range_raises(bad):
    with pytest.raises(ContractError):
        parse_bbox(bad)


@pytest.mark.parametrize(
    "bad",
    [
        [-122.6, 36.4, -121.6],
        [-122.6, 36.4, -121.6, 37.1, 0.0],
        {"lon_min": -122.6, "lat_min": 36.4, "lon_max": -121.6},
        None,
        "bbox",
    ],
)
def test_parse_bbox_rejects_unusable_input(bad):
    with pytest.raises(ContractError):
        parse_bbox(bad)


# ==========================================================================
# 4. buffer defaults by kind
# ==========================================================================


def test_cable_default_buffer_is_non_zero():
    """A zero-buffered cable corridor is a zero-area line that can never contain
    anything: it loads clean, indexes clean, and never fires."""
    assert default_buffer_m("cable") > 0.0
    assert DEFAULT_BUFFER_M_BY_KIND["cable"] == 500.0


@pytest.mark.parametrize("kind", ["mpa", "port", "danger", "land"])
def test_areal_kinds_default_to_zero_buffer(kind):
    """A polygon already has area; an unrequested buffer would quietly enlarge a
    sanctuary and over-report intrusions."""
    assert default_buffer_m(kind) == 0.0


def test_every_canonical_kind_has_a_default():
    assert set(DEFAULT_BUFFER_M_BY_KIND) == set(CANONICAL_KINDS)


def test_default_buffer_m_normalizes_its_kind_first():
    """Omar's layer entry says "cable_corridor"; a raw dict lookup would raise
    KeyError and re-open the zero-buffer trap this table exists to close."""
    assert default_buffer_m("cable_corridor") == 500.0
    assert default_buffer_m("Submarine Cable") == 500.0
    assert default_buffer_m("sanctuary") == 0.0


def test_default_buffer_m_unknown_kind_raises_contract_error_not_keyerror():
    with pytest.raises(ContractError):
        default_buffer_m("sancturay")


def test_omars_cable_layer_entry_becomes_loadable():
    """The end-to-end point of section 4: a pack layer entry with no buffer_m
    field can now produce a usable cable fence."""
    entry = {"file": "layers/ca_submarine_cables.geojson", "kind": "cable_corridor"}
    kind = normalize_kind(entry["kind"])
    buf = resolve_field(entry, "buffer_m", default=default_buffer_m(kind))
    assert kind == "cable"
    assert buf == 500.0


# ==========================================================================
# 5. id namespacing -- the silent fence swap
# ==========================================================================


def test_namespaced_id_shape():
    assert namespaced_id("monterey_bay_nms", 1) == "monterey_bay_nms:1"
    assert namespaced_id("monterey_bay_nms", "1") == "monterey_bay_nms:1"


@pytest.mark.parametrize(
    "layer",
    [
        "monterey_bay_nms",
        "Monterey Bay NMS",
        "monterey-bay-nms",
        "monterey/bay/nms",
        "  MONTEREY  BAY  NMS  ",
    ],
)
def test_namespaced_id_layer_part_is_stable_across_spellings(layer):
    """A file stem, a display name and a path fragment for the same layer must
    not produce three different id namespaces."""
    assert namespaced_id(layer, 1) == "monterey_bay_nms:1"


def test_namespaced_id_does_not_strip_a_file_extension():
    """Sanitising folds separators but does not parse paths -- pass the stem, not
    the filename, or the extension becomes part of the namespace."""
    assert namespaced_id("monterey_bay_nms.geojson", 1) == (
        "monterey_bay_nms_geojson:1"
    )


def test_namespaced_id_keeps_the_raw_id_verbatim():
    """The raw id must stay greppable against the source file's property."""
    assert namespaced_id("layer", "POLY_ID-7") == "layer:POLY_ID-7"


@pytest.mark.parametrize("bad_layer", ["", "   ", "---", "///"])
def test_namespaced_id_rejects_a_layer_that_sanitises_to_nothing(bad_layer):
    with pytest.raises(ContractError):
        namespaced_id(bad_layer, 1)


def test_namespaced_id_rejects_an_empty_raw_id():
    with pytest.raises(ContractError):
        namespaced_id("layer", "")


def test_namespacing_fixes_the_real_poly_id_collision():
    """Monterey's fence id resolves to "1" from POLY_ID, and so does another
    layer's first polygon. Namespacing makes the collision impossible."""
    raw = ["1", "1"]
    with pytest.raises(ContractError):
        assert_unique_ids(raw)
    namespaced = [
        namespaced_id("monterey_bay_nms", "1"),
        namespaced_id("port_of_sf", "1"),
    ]
    assert namespaced == ["monterey_bay_nms:1", "port_of_sf:1"]
    assert_unique_ids(namespaced)  # does not raise


def test_dedupe_ids_suffixes_collisions_deterministically():
    assert dedupe_ids(["1", "1", "1"]) == ["1", "1#1", "1#2"]
    assert dedupe_ids(["a", "b", "a", "b", "a"]) == ["a", "b", "a#1", "b#1", "a#2"]


def test_dedupe_ids_preserves_order_and_leaves_unique_input_alone():
    ids = ["c", "a", "b"]
    assert dedupe_ids(ids) == ids


def test_dedupe_ids_handles_the_suffix_itself_colliding():
    """["1", "1", "1#1"] -- a naive counter emits "1#1" twice and reintroduces
    exactly the silent fence swap this section exists to prevent."""
    out = dedupe_ids(["1", "1", "1#1"])
    assert len(set(out)) == 3, out
    assert out == ["1", "1#1", "1#1#1"]


def test_dedupe_ids_output_is_always_unique():
    messy = ["1", "1", "1#1", "1#1", "1#2", "1", "2"]
    out = dedupe_ids(messy)
    assert len(out) == len(messy)
    assert_unique_ids(out)  # does not raise


def test_dedupe_ids_empty():
    assert dedupe_ids([]) == []


def test_assert_unique_ids_passes_on_unique_input():
    assert assert_unique_ids(["a", "b", "c"]) is None
    assert assert_unique_ids([]) is None


def test_assert_unique_ids_lists_every_duplicate_sorted():
    with pytest.raises(ContractError) as exc:
        assert_unique_ids(["b", "a", "b", "a", "c"])
    msg = str(exc.value)
    assert "'a'" in msg and "'b'" in msg
    assert "'c'" not in msg
    # Deterministic ordering: sorted, so the message is testable.
    assert msg.index("'a'") < msg.index("'b'")


def test_assert_unique_ids_reports_counts():
    with pytest.raises(ContractError) as exc:
        assert_unique_ids(["1", "1", "1"])
    assert "x3" in str(exc.value)


def test_assert_unique_ids_accepts_any_iterable():
    with pytest.raises(ContractError):
        assert_unique_ids(iter(["1", "1"]))


# ==========================================================================
# 6. time parsing
# ==========================================================================


def test_parse_epoch_passthrough_int_and_float():
    assert parse_epoch(EPOCH_20240615_0600Z) == EPOCH_20240615_0600Z
    assert parse_epoch(1718431200) == EPOCH_20240615_0600Z
    assert parse_epoch(0) == 0.0
    assert isinstance(parse_epoch(0), float)


def test_parse_epoch_numpy_scalars():
    assert parse_epoch(np.float64(EPOCH_20240615_0600Z)) == EPOCH_20240615_0600Z
    assert parse_epoch(np.int64(1718431200)) == EPOCH_20240615_0600Z


def test_parse_epoch_z_suffix():
    assert parse_epoch("2024-06-15T06:00:00Z") == EPOCH_20240615_0600Z


def test_parse_epoch_explicit_utc_offset():
    """Omar's real time_window spelling."""
    assert parse_epoch("2024-06-15T06:00:00+00:00") == EPOCH_20240615_0600Z


def test_parse_epoch_non_utc_offset_is_honoured():
    assert parse_epoch("2024-06-15T08:00:00+02:00") == EPOCH_20240615_0600Z
    assert parse_epoch("2024-06-14T23:00:00-07:00") == EPOCH_20240615_0600Z


def test_parse_epoch_naive_string_is_utc_not_local():
    """THE local-timezone regression test. If parse_epoch ever lets
    datetime.timestamp() apply the machine's zone, this exact number changes."""
    assert parse_epoch("2024-06-15T06:00:00") == EPOCH_20240615_0600Z


def test_parse_epoch_naive_datetime_is_utc_not_local():
    naive = datetime(2024, 6, 15, 6, 0, 0)
    assert parse_epoch(naive) == EPOCH_20240615_0600Z


def test_parse_epoch_aware_datetime():
    aware = datetime(2024, 6, 15, 6, 0, 0, tzinfo=timezone.utc)
    assert parse_epoch(aware) == EPOCH_20240615_0600Z
    shifted = datetime(2024, 6, 15, 8, 0, 0, tzinfo=timezone(timedelta(hours=2)))
    assert parse_epoch(shifted) == EPOCH_20240615_0600Z


def test_naive_and_aware_spellings_of_the_same_instant_agree():
    """The four ways a scenario pack can write 06:00 UTC must all be one number,
    regardless of the machine's local zone."""
    values = [
        "2024-06-15T06:00:00",
        "2024-06-15T06:00:00Z",
        "2024-06-15T06:00:00+00:00",
        datetime(2024, 6, 15, 6, 0, 0),
        datetime(2024, 6, 15, 6, 0, 0, tzinfo=timezone.utc),
        EPOCH_20240615_0600Z,
    ]
    assert {parse_epoch(v) for v in values} == {EPOCH_20240615_0600Z}


def test_parse_epoch_date_only_string_is_midnight_utc():
    assert parse_epoch("2024-06-15") == parse_epoch("2024-06-15T00:00:00Z")


def test_parse_epoch_accepts_a_numeric_string():
    assert parse_epoch("1718431200") == EPOCH_20240615_0600Z


@pytest.mark.parametrize(
    "bad", ["", "   ", "not a date", None, {}, [], True, float("nan")]
)
def test_parse_epoch_rejects_unusable_input(bad):
    with pytest.raises(ContractError):
        parse_epoch(bad)


def test_parse_time_window_omars_iso_shape():
    tw = {"start": "2024-06-15T06:00:00+00:00", "end": "2024-06-15T07:00:00+00:00"}
    start, end = parse_time_window(tw)
    assert start == EPOCH_20240615_0600Z
    assert end == EPOCH_20240615_0600Z + 3600.0
    assert start < end


def test_parse_time_window_build_plan_epoch_shape():
    tw = {"start_t": EPOCH_20240615_0600Z, "end_t": EPOCH_20240615_0600Z + 3600.0}
    assert parse_time_window(tw) == (
        EPOCH_20240615_0600Z,
        EPOCH_20240615_0600Z + 3600.0,
    )


def test_parse_time_window_both_shapes_agree():
    iso = {"start": "2024-06-15T06:00:00Z", "end": "2024-06-15T07:00:00Z"}
    epochs = {"start_t": EPOCH_20240615_0600Z, "end_t": EPOCH_20240615_0600Z + 3600.0}
    assert parse_time_window(iso) == parse_time_window(epochs)


def test_parse_time_window_mixed_spellings():
    """Each bound goes through parse_epoch independently."""
    tw = {"start": "2024-06-15T06:00:00Z", "end_t": EPOCH_20240615_0600Z + 60.0}
    assert parse_time_window(tw) == (
        EPOCH_20240615_0600Z,
        EPOCH_20240615_0600Z + 60.0,
    )


def test_parse_time_window_sequence_shape():
    assert parse_time_window(["2024-06-15T06:00:00Z", "2024-06-15T07:00:00Z"]) == (
        EPOCH_20240615_0600Z,
        EPOCH_20240615_0600Z + 3600.0,
    )


def test_parse_time_window_requires_start_strictly_before_end():
    with pytest.raises(ContractError):
        parse_time_window({"start": 100.0, "end": 100.0})
    with pytest.raises(ContractError):
        parse_time_window({"start": 200.0, "end": 100.0})
    with pytest.raises(ContractError):
        parse_time_window(
            {"start": "2024-06-15T07:00:00Z", "end": "2024-06-15T06:00:00Z"}
        )


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"start": 0.0},
        {"end_t": 1.0},
        {"from": 0.0, "to": 1.0},
        [0.0],
        [0.0, 1.0, 2.0],
        None,
        "06:00 to 07:00",
    ],
)
def test_parse_time_window_rejects_unusable_input(bad):
    with pytest.raises(ContractError):
        parse_time_window(bad)


def test_parse_time_window_none_bound_is_absent():
    with pytest.raises(ContractError):
        parse_time_window({"start": None, "end": 1.0})

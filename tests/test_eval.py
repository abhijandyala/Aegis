"""Tests for the scenario-pack regression harness (``tracker.eval``).

``scenarios/`` is not present in this working tree -- the packs live on an
unmerged branch -- so every fixture here builds a pack on disk with ``tmp_path``
and ``json.dump``. That is not a workaround: it is how the harness is meant to
be tested, since it reads pack JSON directly and must work against any root.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tracker.eval import (
    CHECKERS,
    FAIL,
    MARGIN_WARN_FRAC,
    OFFSET_TOLERANCE_FRAC,
    PASS,
    SKIP,
    CheckResult,
    _bbox_bounds,
    _collect_file_refs,
    _ellipsoidal_km,
    _haversine_km,
    _parse_epoch,
    _window_bounds,
    discover_packs,
    evaluate_all,
    evaluate_pack,
    format_report,
    load_pack,
    main,
)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


# --------------------------------------------------------------------------- #
# fixtures: synthetic packs shaped like Omar's real ones
# --------------------------------------------------------------------------- #


def s01_pack() -> dict:
    """An s01-shaped pack whose every structural key should PASS."""
    return {
        "id": "s01_dark_in_sanctuary",
        "name": "Dark vessel loitering in the sanctuary",
        "time_window": {"start": "2024-06-15T06:00:00+00:00", "end": "2024-06-15T08:00:00+00:00"},
        "frame_interval_s": 30.0,
        "geofence_layers": [{"file": "layers/monterey_bay_nms.geojson", "kind": "sanctuary"}],
        "synthetic_actors": [
            {
                "actor_id": "target_1",
                "track": "gen:transit_then_loiter",
                "params": {"start_latlon": [36.75, -122.90], "speed_mps": 6.0},
            }
        ],
        "events": [
            {"id": "ev_dark", "t": 3135.0, "kind": "ais_off", "actor": "target_1"},
            {"id": "ev_radar", "t": 5535.0, "kind": "radar_contact", "actor": "target_1"},
        ],
        "expected": {
            "id_switches_max": 0,
            "reassoc_p_min": 0.85,
            "geofence_alert": True,
            "dark_actor": "target_1",
            "sanctuary_entry_s": 1350.0,
            "dark_at_s": 3135.0,
            "radar_contact_at_s": 5535.0,
            "intrusion_fence": "monterey_bay_nms",
            "predicted_vs_true_offset_verified_m": 1794,
            "notes": "provenance text that must never become a check",
        },
    }


def s02_pack() -> dict:
    """An s02-shaped pack: two actors 40 km apart sharing one MMSI."""
    return {
        "id": "s02_mmsi_spoof",
        "name": "Simultaneous MMSI spoof",
        "time_window": {"start": "2024-06-15T06:00:00+00:00", "end": "2024-06-15T07:00:00+00:00"},
        "synthetic_actors": [
            {"actor_id": "vessel_a", "track": "gen:transit", "params": {"start_latlon": [36.6, -121.9]}},
            {"actor_id": "vessel_b", "track": "gen:transit", "params": {"start_latlon": [36.95, -122.0]}},
        ],
        "events": [
            {
                "id": "ev_spoof_a",
                "t": 0.0,
                "kind": "identity_change",
                "actor": "vessel_a",
                "params": {"new_display_id": "mmsi:412345678"},
            },
            {
                "id": "ev_spoof_b",
                "t": 0.0,
                "kind": "identity_change",
                "actor": "vessel_b",
                "params": {"new_display_id": "mmsi:412345678"},
            },
        ],
        "expected": {
            "spoof_detected": True,
            "spoof_mmsi": "mmsi:412345678",
            "actors": ["vessel_a", "vessel_b"],
            "min_separation_km": 39.9,
            "retraction_cascade": "expected only if JTMS is implemented (Raj); if not, "
            "spoof_detected alone is the pack's minimum bar",
            "notes": "both actors broadcast the same display id from t=0",
        },
    }


def s03_pack() -> dict:
    """An s03-shaped pack: every expected key is an unbuilt capability."""
    return {
        "id": "s03_ghost_fleet",
        "name": "Ghost fleet re-flagging",
        "time_window": {"start": "2024-06-15T06:00:00+00:00", "end": "2024-06-15T07:00:00+00:00"},
        "synthetic_actors": [{"actor_id": "actor_1", "params": {"start_latlon": [36.5, -122.2]}}],
        "events": [
            {"id": "ev_reflag", "t": 1800.0, "kind": "identity_change", "actor": "actor_1"},
        ],
        "expected": {
            "relink_same_track": True,
            "ofac_hit": True,
            "ofac_hit_name": "ARTAVIL",
            "ofac_hit_mmsi": "mmsi:572469210",
            "severity_escalates_on_relink": True,
            "notes": "roadmap tier",
        },
    }


def good_s01_results() -> dict:
    """Measured run data that satisfies every measured s01 key."""
    return {
        "id_switches": 0,
        "reassoc_p": 0.93,
        "geofence_alerts": [{"fence": "monterey_bay_nms", "t": 1350.0}],
        "offset_m": 1750.0,
        "reassoc_track_id": "target_1",
    }


def write_pack(
    root: Path, pack: dict, *, dirname: str | None = None, create_refs: bool = True
) -> Path:
    """Write ``pack`` to ``<root>/scenarios/<dirname>/pack.json``.

    By default the files the pack references (``geofence_layers[].file`` and
    friends) are created alongside it, because a pack on disk that points at a
    file that is not there is now itself a finding -- see
    ``packcheck_referenced_files``. Pass ``create_refs=False`` to write a pack
    with dangling references on purpose.
    """
    d = root / "scenarios" / (dirname or str(pack["id"]))
    d.mkdir(parents=True, exist_ok=True)
    path = d / "pack.json"
    path.write_text(json.dumps(pack, indent=2), encoding="utf-8")
    if create_refs:
        for ref in referenced_paths(pack):
            target = d / ref
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_text("{}", encoding="utf-8")
    return path


def referenced_paths(pack: dict) -> list[str]:
    """Every file path the pack references, using the harness's own walker."""
    found: list[tuple[str, str]] = []
    _collect_file_refs(pack, found)
    return [ref for _, ref in found]


def outcomes(result) -> dict[str, str]:
    return {c.key: c.outcome for c in result.checks}


def detail_of(result, key: str) -> str:
    return next(c.detail for c in result.checks if c.key == key)


# --------------------------------------------------------------------------- #
# discovery and loading
# --------------------------------------------------------------------------- #


def test_discover_packs_absent_scenarios_dir_returns_empty(tmp_path: Path) -> None:
    """An absent scenarios/ is the normal state of this tree. No raise."""
    assert not (tmp_path / "scenarios").exists()
    assert discover_packs(tmp_path) == []


def test_discover_packs_finds_and_sorts(tmp_path: Path) -> None:
    write_pack(tmp_path, s02_pack())
    write_pack(tmp_path, s01_pack())
    write_pack(tmp_path, s03_pack())
    found = discover_packs(tmp_path)
    assert [p.parent.name for p in found] == [
        "s01_dark_in_sanctuary",
        "s02_mmsi_spoof",
        "s03_ghost_fleet",
    ]


def test_discover_packs_ignores_other_json(tmp_path: Path) -> None:
    write_pack(tmp_path, s01_pack())
    (tmp_path / "scenarios" / "s01_dark_in_sanctuary" / "notes.json").write_text("{}", encoding="utf-8")
    assert [p.name for p in discover_packs(tmp_path)] == ["pack.json"]


def test_load_pack_records_source_path(tmp_path: Path) -> None:
    path = write_pack(tmp_path, s01_pack())
    pack = load_pack(path)
    assert evaluate_pack(pack).path == str(path)


def test_malformed_pack_json_error_names_the_file(tmp_path: Path) -> None:
    d = tmp_path / "scenarios" / "broken"
    d.mkdir(parents=True)
    path = d / "pack.json"
    path.write_text('{"id": "broken", "expected": {', encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_pack(path)
    msg = str(exc.value)
    assert "pack.json" in msg
    assert "broken" in msg
    assert "malformed" in msg.lower()


def test_pack_json_that_is_not_an_object_is_rejected(tmp_path: Path) -> None:
    d = tmp_path / "scenarios" / "listy"
    d.mkdir(parents=True)
    path = d / "pack.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="pack.json"):
        load_pack(path)


# --------------------------------------------------------------------------- #
# structural checks on a well-formed pack
# --------------------------------------------------------------------------- #


def test_wellformed_s01_structural_keys_all_pass() -> None:
    res = evaluate_pack(s01_pack())
    got = outcomes(res)
    for key in (
        "dark_actor",
        "intrusion_fence",
        "sanctuary_entry_s",
        "dark_at_s",
        "radar_contact_at_s",
    ):
        assert got[key] == PASS, f"{key}: {detail_of(res, key)}"
    assert res.failed == 0
    assert res.ok is True


def test_notes_produces_no_row_at_all() -> None:
    res = evaluate_pack(s01_pack())
    assert "notes" not in outcomes(res)
    assert all(c.key != "notes" for c in res.checks)


def test_dark_actor_not_in_synthetic_actors_fails() -> None:
    pack = s01_pack()
    pack["expected"]["dark_actor"] = "ghost_9"
    res = evaluate_pack(pack)
    assert outcomes(res)["dark_actor"] == FAIL
    assert "ghost_9" in detail_of(res, "dark_actor")
    assert res.ok is False


def test_actors_list_with_one_undefined_member_fails() -> None:
    pack = s02_pack()
    pack["expected"]["actors"] = ["vessel_a", "vessel_z"]
    res = evaluate_pack(pack)
    assert outcomes(res)["actors"] == FAIL
    assert "vessel_z" in detail_of(res, "actors")


def test_actors_all_defined_passes() -> None:
    assert outcomes(evaluate_pack(s02_pack()))["actors"] == PASS


def test_intrusion_fence_unknown_layer_fails() -> None:
    pack = s01_pack()
    pack["expected"]["intrusion_fence"] = "central_bay_sanctuary"
    res = evaluate_pack(pack)
    assert outcomes(res)["intrusion_fence"] == FAIL
    assert "monterey_bay_nms" in detail_of(res, "intrusion_fence")


def test_intrusion_fence_with_no_layers_declared_fails() -> None:
    pack = s01_pack()
    del pack["geofence_layers"]
    assert outcomes(evaluate_pack(pack))["intrusion_fence"] == FAIL


def test_spoof_mmsi_present_passes_and_counts_broadcasters() -> None:
    res = evaluate_pack(s02_pack())
    assert outcomes(res)["spoof_mmsi"] == PASS
    detail = detail_of(res, "spoof_mmsi")
    assert "vessel_a" in detail and "vessel_b" in detail


def test_spoof_mmsi_never_broadcast_fails() -> None:
    pack = s02_pack()
    pack["expected"]["spoof_mmsi"] = "mmsi:000000000"
    assert outcomes(evaluate_pack(pack))["spoof_mmsi"] == FAIL


# --------------------------------------------------------------------------- #
# timings: window bounds, ordering, event cross-reference
# --------------------------------------------------------------------------- #


def test_out_of_order_radar_contact_before_dark_fails() -> None:
    pack = s01_pack()
    pack["expected"]["radar_contact_at_s"] = 2000.0  # before dark_at_s=3135
    res = evaluate_pack(pack)
    assert outcomes(res)["radar_contact_at_s"] == FAIL
    assert "ordering" in detail_of(res, "radar_contact_at_s").lower()


def test_dark_before_sanctuary_entry_fails() -> None:
    pack = s01_pack()
    pack["expected"]["sanctuary_entry_s"] = 4000.0
    res = evaluate_pack(pack)
    assert outcomes(res)["dark_at_s"] == FAIL
    assert "ordering" in detail_of(res, "dark_at_s").lower()


def test_timing_after_end_of_time_window_fails() -> None:
    pack = s01_pack()
    pack["expected"]["radar_contact_at_s"] = 9999.0  # window is 7200 s
    res = evaluate_pack(pack)
    assert outcomes(res)["radar_contact_at_s"] == FAIL
    assert "OUTSIDE" in detail_of(res, "radar_contact_at_s")


def test_negative_timing_fails() -> None:
    pack = s01_pack()
    pack["expected"]["sanctuary_entry_s"] = -10.0
    res = evaluate_pack(pack)
    assert outcomes(res)["sanctuary_entry_s"] == FAIL


def test_dark_at_s_must_match_an_ais_off_event() -> None:
    pack = s01_pack()
    pack["expected"]["dark_at_s"] = 3200.0  # the ais_off event is at 3135
    res = evaluate_pack(pack)
    assert outcomes(res)["dark_at_s"] == FAIL
    assert "ais_off" in detail_of(res, "dark_at_s")


def test_dark_at_s_fails_when_no_ais_off_event_exists() -> None:
    pack = s01_pack()
    pack["events"] = [e for e in pack["events"] if e["kind"] != "ais_off"]
    res = evaluate_pack(pack)
    assert outcomes(res)["dark_at_s"] == FAIL
    assert "no 'ais_off' event" in detail_of(res, "dark_at_s")


def test_events_keyed_on_kind_and_on_type_both_work() -> None:
    """Omar's packs use `kind`; `type` is accepted and the detail says so."""
    kind_pack = s01_pack()
    assert outcomes(evaluate_pack(kind_pack))["dark_at_s"] == PASS

    type_pack = s01_pack()
    type_pack["events"] = [
        {"id": e["id"], "t": e["t"], "type": e["kind"], "actor": e["actor"]} for e in type_pack["events"]
    ]
    res = evaluate_pack(type_pack)
    assert outcomes(res)["dark_at_s"] == PASS
    assert "type" in detail_of(res, "dark_at_s")


def test_unparseable_time_window_does_not_fail_the_timing_check() -> None:
    pack = s01_pack()
    pack["time_window"] = {"start": "not-a-date", "end": "also-not"}
    res = evaluate_pack(pack)
    assert outcomes(res)["dark_at_s"] == PASS
    assert "not parseable" in detail_of(res, "dark_at_s")


def test_non_numeric_timing_fails_cleanly() -> None:
    pack = s01_pack()
    pack["expected"]["dark_at_s"] = "half past three"
    assert outcomes(evaluate_pack(pack))["dark_at_s"] == FAIL


# --------------------------------------------------------------------------- #
# min_separation_km
# --------------------------------------------------------------------------- #


def test_min_separation_passes_and_reports_the_computed_value() -> None:
    res = evaluate_pack(s02_pack())
    assert outcomes(res)["min_separation_km"] == PASS
    detail = detail_of(res, "min_separation_km")
    assert "haversine" in detail
    assert "39.9" in detail  # both the computed value and the threshold


def test_min_separation_below_threshold_fails() -> None:
    pack = s02_pack()
    pack["expected"]["min_separation_km"] = 100.0
    res = evaluate_pack(pack)
    assert outcomes(res)["min_separation_km"] == FAIL
    assert "SHORT" in detail_of(res, "min_separation_km")


def test_min_separation_skips_when_positions_not_derivable() -> None:
    pack = s02_pack()
    for actor in pack["synthetic_actors"]:
        actor["params"] = {"speed_mps": 5.0}
    res = evaluate_pack(pack)
    assert outcomes(res)["min_separation_km"] == SKIP
    assert detail_of(res, "min_separation_km")


def test_min_separation_skips_with_a_single_actor() -> None:
    pack = s02_pack()
    pack["synthetic_actors"] = pack["synthetic_actors"][:1]
    assert outcomes(evaluate_pack(pack))["min_separation_km"] == SKIP


# --------------------------------------------------------------------------- #
# measured keys: results=None vs good vs bad
# --------------------------------------------------------------------------- #


MEASURED_S01 = (
    "id_switches_max",
    "reassoc_p_min",
    "geofence_alert",
    "predicted_vs_true_offset_verified_m",
)


def test_results_none_measured_keys_skip_structural_keys_still_pass() -> None:
    res = evaluate_pack(s01_pack(), results=None)
    got = outcomes(res)
    for key in MEASURED_S01:
        assert got[key] == SKIP, f"{key} should SKIP without run data"
        assert "no run data supplied" in detail_of(res, key)
    assert got["dark_actor"] == PASS
    assert got["dark_at_s"] == PASS
    assert res.failed == 0
    assert res.ok is True


def test_good_results_make_measured_keys_pass() -> None:
    res = evaluate_pack(s01_pack(), results=good_s01_results())
    got = outcomes(res)
    for key in MEASURED_S01:
        assert got[key] == PASS, f"{key}: {detail_of(res, key)}"
    assert res.failed == 0


def test_bad_id_switches_fails() -> None:
    results = good_s01_results() | {"id_switches": 3}
    res = evaluate_pack(s01_pack(), results=results)
    assert outcomes(res)["id_switches_max"] == FAIL
    assert "OVER" in detail_of(res, "id_switches_max")


def test_low_reassoc_p_fails() -> None:
    results = good_s01_results() | {"reassoc_p": 0.42}
    res = evaluate_pack(s01_pack(), results=results)
    assert outcomes(res)["reassoc_p_min"] == FAIL
    assert "BELOW" in detail_of(res, "reassoc_p_min")


def test_no_geofence_alerts_fails() -> None:
    results = good_s01_results() | {"geofence_alerts": []}
    res = evaluate_pack(s01_pack(), results=results)
    assert outcomes(res)["geofence_alert"] == FAIL


def test_geofence_alert_on_the_wrong_fence_fails() -> None:
    results = good_s01_results() | {"geofence_alerts": [{"fence": "gate_approach_restricted"}]}
    res = evaluate_pack(s01_pack(), results=results)
    assert outcomes(res)["geofence_alert"] == FAIL
    assert "monterey_bay_nms" in detail_of(res, "geofence_alert")


def test_geofence_alert_accepts_plain_string_alerts() -> None:
    results = good_s01_results() | {"geofence_alerts": ["monterey_bay_nms"]}
    assert outcomes(evaluate_pack(s01_pack(), results=results))["geofence_alert"] == PASS


def test_offset_within_documented_tolerance_passes() -> None:
    want = 1794
    edge = want * (1.0 + OFFSET_TOLERANCE_FRAC * 0.99)
    results = good_s01_results() | {"offset_m": edge}
    assert outcomes(evaluate_pack(s01_pack(), results=results))["predicted_vs_true_offset_verified_m"] == PASS


def test_offset_outside_tolerance_fails() -> None:
    results = good_s01_results() | {"offset_m": 400.0}
    res = evaluate_pack(s01_pack(), results=results)
    assert outcomes(res)["predicted_vs_true_offset_verified_m"] == FAIL
    assert "TOLERANCE" in detail_of(res, "predicted_vs_true_offset_verified_m").upper()


def test_results_supplied_but_missing_a_key_skips_with_that_reason() -> None:
    results = {k: v for k, v in good_s01_results().items() if k != "reassoc_p"}
    res = evaluate_pack(s01_pack(), results=results)
    assert outcomes(res)["reassoc_p_min"] == SKIP
    assert "reassoc_p" in detail_of(res, "reassoc_p_min")


def test_reassoc_correct_uses_dark_actor_as_ground_truth() -> None:
    pack = s01_pack()
    pack["expected"]["reassoc_correct"] = True

    good = evaluate_pack(pack, results=good_s01_results())
    assert outcomes(good)["reassoc_correct"] == PASS

    bad = evaluate_pack(pack, results=good_s01_results() | {"reassoc_track_id": "target_2"})
    assert outcomes(bad)["reassoc_correct"] == FAIL
    assert "MISATTRIBUTED" in detail_of(bad, "reassoc_correct")


# --------------------------------------------------------------------------- #
# not-built capabilities and unknown keys
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "key",
    [
        "retraction_cascade",
        "spoof_detected",
        "ofac_hit",
        "ofac_hit_mmsi",
        "ofac_hit_name",
        "relink_same_track",
        "severity_escalates_on_relink",
    ],
)
def test_unbuilt_capabilities_always_skip_with_a_reason(key: str) -> None:
    """Even with perfect run data these SKIP -- the capability does not exist."""
    pack = s01_pack()
    pack["expected"] = {key: True}
    res = evaluate_pack(pack, results=good_s01_results())
    check = res.checks[0]
    assert check.key == key
    assert check.outcome == SKIP
    assert check.detail.strip()
    assert "not implemented" in check.detail


def test_jtms_skips_name_the_retraction_layer() -> None:
    res = evaluate_pack(s02_pack())
    for key in ("spoof_detected", "retraction_cascade"):
        assert "JTMS" in detail_of(res, key)


def test_ofac_skips_name_the_watchlist_lookup() -> None:
    res = evaluate_pack(s03_pack())
    assert "OFAC" in detail_of(res, "ofac_hit")
    assert res.skipped == 5
    assert res.passed == 0
    assert res.failed == 0
    assert res.ok is True


def test_verbose_string_expected_value_is_not_echoed_into_the_detail() -> None:
    """s02's retraction_cascade value is a 100-char sentence; keep it out."""
    res = evaluate_pack(s02_pack())
    detail = detail_of(res, "retraction_cascade")
    assert "minimum bar" not in detail
    assert len(detail) < 120


def test_unknown_expected_key_skips_and_does_not_raise() -> None:
    pack = s01_pack()
    pack["expected"]["wormhole_detected"] = True
    res = evaluate_pack(pack)
    assert outcomes(res)["wormhole_detected"] == SKIP
    assert "no checker registered" in detail_of(res, "wormhole_detected")
    assert res.failed == 0


def test_pack_with_no_expected_block_skips_rather_than_crashing() -> None:
    pack = s01_pack()
    del pack["expected"]
    res = evaluate_pack(pack)
    assert res.failed == 0
    # One SKIP for the absent `expected` block, plus one for the pack-level
    # referenced-files check: this dict was never loaded from disk, so its
    # geofence layer path cannot be resolved against anything.
    assert res.skipped == 2
    assert outcomes(res)["referenced_files"] == SKIP


# --------------------------------------------------------------------------- #
# a broken checker must not take the suite down
# --------------------------------------------------------------------------- #


def test_checker_that_raises_becomes_a_fail_and_the_run_continues(monkeypatch) -> None:
    def exploding(expected, pack, results):
        raise RuntimeError("boom: geofence layer file went missing")

    monkeypatch.setitem(CHECKERS, "dark_actor", exploding)
    res = evaluate_pack(s01_pack())
    got = outcomes(res)
    assert got["dark_actor"] == FAIL
    assert "boom: geofence layer file went missing" in detail_of(res, "dark_actor")
    assert "RuntimeError" in detail_of(res, "dark_actor")
    # every other key still got checked
    assert got["dark_at_s"] == PASS
    assert got["intrusion_fence"] == PASS
    # minus notes, plus the pack-level referenced_files row appended at the end
    assert len(res.checks) == len(s01_pack()["expected"]) - 1 + 1
    assert res.checks[-1].key == "referenced_files"


def test_checker_returning_the_wrong_type_becomes_a_fail(monkeypatch) -> None:
    monkeypatch.setitem(CHECKERS, "dark_actor", lambda e, p, r: "sure, looks fine")
    res = evaluate_pack(s01_pack())
    assert outcomes(res)["dark_actor"] == FAIL
    assert "CheckResult" in detail_of(res, "dark_actor")


# --------------------------------------------------------------------------- #
# evaluate_all
# --------------------------------------------------------------------------- #


def test_evaluate_all_covers_every_pack(tmp_path: Path) -> None:
    for pack in (s01_pack(), s02_pack(), s03_pack()):
        write_pack(tmp_path, pack)
    results = evaluate_all(tmp_path)
    assert [r.pack_id for r in results] == [
        "s01_dark_in_sanctuary",
        "s02_mmsi_spoof",
        "s03_ghost_fleet",
    ]
    assert all(r.ok for r in results)
    assert sum(r.failed for r in results) == 0
    assert sum(r.passed for r in results) > 0


def test_evaluate_all_routes_results_by_pack_id(tmp_path: Path) -> None:
    write_pack(tmp_path, s01_pack())
    results = evaluate_all(tmp_path, results_by_pack={"s01_dark_in_sanctuary": good_s01_results()})
    got = outcomes(results[0])
    assert got["id_switches_max"] == PASS
    assert got["reassoc_p_min"] == PASS


def test_evaluate_all_on_empty_root(tmp_path: Path) -> None:
    assert evaluate_all(tmp_path) == []


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #


def test_report_has_no_ansi_escapes_when_colour_disabled(tmp_path: Path) -> None:
    for pack in (s01_pack(), s02_pack(), s03_pack()):
        write_pack(tmp_path, pack)
    text = format_report(evaluate_all(tmp_path), use_colour=False)
    assert "\x1b" not in text
    assert ANSI_RE.search(text) is None


def test_report_emits_ansi_when_colour_forced(tmp_path: Path) -> None:
    write_pack(tmp_path, s01_pack())
    text = format_report(evaluate_all(tmp_path), use_colour=True)
    assert ANSI_RE.search(text) is not None


def test_report_contains_a_row_per_check_and_a_total(tmp_path: Path) -> None:
    write_pack(tmp_path, s01_pack())
    results = evaluate_all(tmp_path)
    text = format_report(results, use_colour=False)
    for check in results[0].checks:
        assert check.key in text
    assert "notes" not in text
    assert "TOTAL" in text
    assert "s01_dark_in_sanctuary" in text


def test_report_key_column_is_aligned_across_packs(tmp_path: Path) -> None:
    for pack in (s01_pack(), s02_pack(), s03_pack()):
        write_pack(tmp_path, pack)
    text = format_report(evaluate_all(tmp_path), use_colour=False)
    rows = [ln for ln in text.splitlines() if re.match(r"^  (PASS|FAIL|SKIP)  ", ln)]
    assert len(rows) >= 15

    detail_starts = set()
    for row in rows:
        body = row[8:]  # past "  PASS  "
        key = body.split()[0]
        rest = body[len(key):]
        detail_starts.add(8 + len(key) + (len(rest) - len(rest.lstrip())))
    assert len(detail_starts) == 1, f"detail column is ragged: {sorted(detail_starts)}"


def test_report_on_no_packs_says_so(tmp_path: Path) -> None:
    text = format_report(evaluate_all(tmp_path), use_colour=False)
    assert "no packs found" in text


def test_report_flags_a_failing_pack() -> None:
    pack = s01_pack()
    pack["expected"]["dark_actor"] = "nobody"
    text = format_report([evaluate_pack(pack)], use_colour=False)
    assert "FAIL" in text
    assert "1 failed" in text
    assert "FAILURES IN THIS PACK" in text


def test_report_explains_what_skip_means() -> None:
    text = format_report([evaluate_pack(s03_pack())], use_colour=False)
    assert "Not a failure" in text


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_main_returns_zero_when_only_pass_and_skip(tmp_path: Path, capsys) -> None:
    for pack in (s01_pack(), s02_pack(), s03_pack()):
        write_pack(tmp_path, pack)
    assert main(["--all", "--root", str(tmp_path), "--colour", "never"]) == 0
    out = capsys.readouterr().out
    assert "0 failed" in out


def test_main_returns_nonzero_when_a_check_failed(tmp_path: Path) -> None:
    pack = s01_pack()
    pack["expected"]["dark_actor"] = "nobody"
    write_pack(tmp_path, pack)
    assert main(["--root", str(tmp_path), "--colour", "never"]) == 1


def test_main_on_absent_scenarios_dir_exits_zero(tmp_path: Path, capsys) -> None:
    assert main(["--all", "--root", str(tmp_path)]) == 0
    assert "no packs found" in capsys.readouterr().out


def test_main_pack_filter(tmp_path: Path, capsys) -> None:
    for pack in (s01_pack(), s02_pack()):
        write_pack(tmp_path, pack)
    assert main(["--pack", "s02_mmsi_spoof", "--root", str(tmp_path), "--colour", "never"]) == 0
    out = capsys.readouterr().out
    assert "s02_mmsi_spoof" in out
    assert "s01_dark_in_sanctuary" not in out


def test_main_unknown_pack_id_is_a_usage_error(tmp_path: Path, capsys) -> None:
    write_pack(tmp_path, s01_pack())
    assert main(["--pack", "s99_nope", "--root", str(tmp_path)]) == 2
    assert "s99_nope" in capsys.readouterr().err


def test_main_malformed_pack_reports_a_usage_error_naming_the_file(tmp_path: Path, capsys) -> None:
    d = tmp_path / "scenarios" / "broken"
    d.mkdir(parents=True)
    (d / "pack.json").write_text("{oops", encoding="utf-8")
    assert main(["--root", str(tmp_path)]) == 2
    assert "pack.json" in capsys.readouterr().err


def test_main_json_output_is_parseable(tmp_path: Path, capsys) -> None:
    for pack in (s01_pack(), s02_pack(), s03_pack()):
        write_pack(tmp_path, pack)
    assert main(["--json", "--root", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["packs"] == 3
    assert payload["summary"]["failed"] == 0
    assert payload["summary"]["ok"] is True
    ids = [p["pack_id"] for p in payload["packs"]]
    assert ids == ["s01_dark_in_sanctuary", "s02_mmsi_spoof", "s03_ghost_fleet"]
    for p in payload["packs"]:
        assert all(c["key"] != "notes" for c in p["checks"])
        assert all(c["outcome"] in {PASS, FAIL, SKIP} for c in p["checks"])


def test_main_quiet_prints_one_line(tmp_path: Path, capsys) -> None:
    write_pack(tmp_path, s01_pack())
    assert main(["--quiet", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out.strip()
    assert out.count("\n") == 0
    assert "passed" in out and "failed" in out and "skipped" in out


def test_main_default_root_is_the_repo_root_and_does_not_crash(capsys) -> None:
    """`python -m tracker.eval --all` with no --root must be a clean exit."""
    assert main(["--all", "--colour", "never"]) == 0


# --------------------------------------------------------------------------- #
# CheckResult / PackResult shape
# --------------------------------------------------------------------------- #


def test_check_result_is_frozen() -> None:
    c = CheckResult(key="k", outcome=PASS, detail="d")
    with pytest.raises(Exception):
        c.key = "other"  # type: ignore[misc]


def test_pack_result_counts_sum_to_the_number_of_checks() -> None:
    res = evaluate_pack(s01_pack())
    assert res.passed + res.failed + res.skipped == len(res.checks)


# --------------------------------------------------------------------------- #
# The registries remain plain, callable Python objects and are read live by
# evaluate_pack, preserving the monkeypatch contract.
# --------------------------------------------------------------------------- #


def test_checkers_registry_values_are_all_plain_callables() -> None:
    for key, checker in CHECKERS.items():
        assert callable(checker), f"CHECKERS[{key!r}] is not callable"


def test_pack_checks_tuple_values_are_all_plain_callables() -> None:
    import tracker.eval as ev

    for pack_check in ev.PACK_CHECKS:
        assert callable(pack_check)


def test_eval_public_api_is_native_python() -> None:
    """Discovery and evaluation are implemented directly in tracker.eval."""
    import tracker.eval as ev

    assert ev.discover_packs.__module__ == "tracker.eval"
    assert ev.evaluate_pack.__module__ == "tracker.eval"
    result = ev.evaluate_pack({"id": "native", "expected": {"dark_actor": "a"},
                               "synthetic_actors": [{"actor_id": "a"}]})
    assert result.pack_id == "native"
    assert outcomes(result)["dark_actor"] == PASS


def test_two_not_built_checkers_are_independent_closures() -> None:
    """Each not-built key gets its own capability message, not a shared one."""
    assert CHECKERS["ofac_hit"] is not CHECKERS["relink_same_track"]
    a = CHECKERS["ofac_hit"](True, {}, None)
    b = CHECKERS["relink_same_track"](True, {}, None)
    assert isinstance(a, CheckResult)
    assert isinstance(b, CheckResult)
    assert a.detail != b.detail
    assert "OFAC" in a.detail
    assert "relink" in b.detail.lower()


# =========================================================================== #
# pack-level check: referenced files must exist
# =========================================================================== #


def test_referenced_file_that_exists_passes(tmp_path: Path) -> None:
    """s01 points at layers/monterey_bay_nms.geojson, which really is there."""
    write_pack(tmp_path, s01_pack())  # create_refs=True writes the layer file
    res = evaluate_all(tmp_path)[0]
    assert outcomes(res)["referenced_files"] == PASS
    assert "layers/monterey_bay_nms.geojson" in detail_of(res, "referenced_files")
    assert res.failed == 0


def test_missing_referenced_file_fails_naming_path_and_resolved_dir(tmp_path: Path) -> None:
    path = write_pack(tmp_path, s01_pack(), create_refs=False)
    res = evaluate_all(tmp_path)[0]
    detail = detail_of(res, "referenced_files")
    assert outcomes(res)["referenced_files"] == FAIL
    assert "layers/monterey_bay_nms.geojson" in detail
    assert "MISSING" in detail
    # the directory the path was resolved against must be named
    assert path.parent.name in detail
    assert res.ok is False


def test_missing_referenced_file_is_visible_in_the_rendered_report(tmp_path: Path) -> None:
    """The missing path must survive detail elision -- it is the whole point."""
    write_pack(tmp_path, s01_pack(), create_refs=False)
    text = format_report(evaluate_all(tmp_path), use_colour=False)
    assert "MISSING layers/monterey_bay_nms.geojson" in text
    assert "referenced_files" in text
    # the resolved directory is the other half of the finding; it must survive too
    assert "resolved against" in text
    assert "s01_dark_in_sanctuary" in text


def test_file_refs_are_found_at_any_depth_and_under_several_key_names(tmp_path: Path) -> None:
    pack = s01_pack()
    pack["ais"] = {"file": "ais_window.csv"}
    pack["watchlists"] = [{"csv": "ofac/subset.csv"}, {"path": "deep/nested/thing.geojson"}]
    write_pack(tmp_path, pack, create_refs=False)
    detail = detail_of(evaluate_all(tmp_path)[0], "referenced_files")
    for ref in ("ais_window.csv", "ofac/subset.csv", "deep/nested/thing.geojson"):
        assert ref in detail


def test_partially_present_refs_report_the_ratio(tmp_path: Path) -> None:
    pack = s01_pack()
    pack["ais"] = {"file": "ais_window.csv"}
    d = tmp_path / "scenarios" / pack["id"]
    write_pack(tmp_path, pack, create_refs=False)
    (d / "layers").mkdir(parents=True, exist_ok=True)
    (d / "layers" / "monterey_bay_nms.geojson").write_text("{}", encoding="utf-8")
    res = evaluate_all(tmp_path)[0]
    detail = detail_of(res, "referenced_files")
    assert outcomes(res)["referenced_files"] == FAIL
    assert "ais_window.csv" in detail
    assert "1/2" in detail


def test_dict_loaded_pack_with_refs_skips_instead_of_crashing() -> None:
    """No source path means nothing to resolve against. SKIP, never a raise."""
    res = evaluate_pack(s01_pack())
    assert outcomes(res)["referenced_files"] == SKIP
    detail = detail_of(res, "referenced_files")
    assert "no source path" in detail
    assert res.failed == 0


def test_pack_with_no_file_refs_at_all_gets_no_referenced_files_row() -> None:
    assert "referenced_files" not in outcomes(evaluate_pack(s02_pack()))
    assert "referenced_files" not in outcomes(evaluate_pack(s03_pack()))


def test_url_valued_ref_is_not_treated_as_a_local_path(tmp_path: Path) -> None:
    pack = s02_pack()
    pack["source"] = "https://example.org/ais/feed.csv"
    write_pack(tmp_path, pack, create_refs=False)
    assert "referenced_files" not in outcomes(evaluate_all(tmp_path)[0])


def test_collect_file_refs_ignores_the_private_source_path_key(tmp_path: Path) -> None:
    pack = load_pack(write_pack(tmp_path, s01_pack()))
    refs = []
    _collect_file_refs(pack, refs)
    assert [r for _, r in refs] == ["layers/monterey_bay_nms.geojson"]


# =========================================================================== #
# pack-level check: path-looking tokens in free-text notes
# =========================================================================== #


def s03_pack_with_stale_note_path() -> dict:
    """s03 as committed: notes point at geo/ofac_sdn_vessels_subset.csv, while
    the csv is actually a sibling of pack.json, not under geo/."""
    pack = s03_pack()
    pack["expected"]["notes"] = (
        "mmsi:572469210 / ARTAVIL is a real OFAC SDN vessel entry (Iran program, "
        "crude/oil tanker) pulled into geo/ofac_sdn_vessels_subset.csv"
    )
    return pack


def test_stale_path_in_notes_is_a_skip_that_says_it_may_be_stale(tmp_path: Path) -> None:
    d = tmp_path / "scenarios" / "s03_ghost_fleet"
    write_pack(tmp_path, s03_pack_with_stale_note_path())
    (d / "ofac_sdn_vessels_subset.csv").write_text("mmsi,name\n", encoding="utf-8")
    res = evaluate_all(tmp_path)[0]
    assert outcomes(res)["free_text_paths"] == SKIP
    detail = detail_of(res, "free_text_paths")
    assert "geo/ofac_sdn_vessels_subset.csv" in detail
    assert "may be stale" in detail
    # free text must never turn a pack red
    assert res.failed == 0
    assert res.ok is True


def test_prose_with_a_slash_but_no_extension_is_not_a_path(tmp_path: Path) -> None:
    """'crude/oil tanker' is English, not a filename."""
    pack = s03_pack()
    pack["expected"]["notes"] = "a crude/oil tanker, re-flagged mid-replay"
    write_pack(tmp_path, pack)
    assert "free_text_paths" not in outcomes(evaluate_all(tmp_path)[0])


def test_note_path_that_really_exists_produces_no_row(tmp_path: Path) -> None:
    pack = s03_pack()
    pack["expected"]["notes"] = "pulled from geo/ofac_sdn_vessels_subset.csv"
    d = tmp_path / "scenarios" / pack["id"]
    write_pack(tmp_path, pack)
    (d / "geo").mkdir(parents=True, exist_ok=True)
    (d / "geo" / "ofac_sdn_vessels_subset.csv").write_text("mmsi,name\n", encoding="utf-8")
    res = evaluate_all(tmp_path)[0]
    assert "free_text_paths" not in outcomes(res)  # pinned: no row, not a PASS


def test_free_text_row_never_prints_the_word_the_report_forbids(tmp_path: Path) -> None:
    """The report test asserts `notes` never appears; this row must not smuggle
    it back in. Both branches are exercised: on-disk (what the projector shows)
    and dict-loaded (no source path)."""
    write_pack(tmp_path, s03_pack_with_stale_note_path())
    on_disk = format_report(evaluate_all(tmp_path), use_colour=False)
    assert "free_text_paths" in on_disk
    assert "notes" not in on_disk

    dict_loaded = evaluate_pack(s03_pack_with_stale_note_path())
    assert outcomes(dict_loaded)["free_text_paths"] == SKIP
    assert "no source path" in detail_of(dict_loaded, "free_text_paths")
    assert "notes" not in format_report([dict_loaded], use_colour=False)


# =========================================================================== #
# min_separation_km: margins, both earth models, the replay caveat
# =========================================================================== #


def test_comfortable_margin_passes_and_is_not_marginal() -> None:
    pack = s02_pack()
    pack["expected"]["min_separation_km"] = 30.0  # actual is ~39.92 km
    res = evaluate_pack(pack)
    check = next(c for c in res.checks if c.key == "min_separation_km")
    assert check.outcome == PASS
    assert check.marginal is False
    assert res.marginal == 0
    assert "MARGIN" not in check.detail
    assert "margin" in check.detail  # the margin is still reported
    assert "haversine" in check.detail


def test_knifeedge_margin_passes_but_is_flagged_marginal() -> None:
    """s02 as written: 39.9 km required, ~39.924 km actual -- 24 m of margin."""
    res = evaluate_pack(s02_pack())
    check = next(c for c in res.checks if c.key == "min_separation_km")
    assert check.outcome == PASS
    assert check.marginal is True
    assert res.marginal == 1
    assert "MARGIN" in check.detail
    assert "earth-model dependent" in check.detail
    assert "24 m" in check.detail
    assert "haversine" in check.detail and "6371.0088" in check.detail


def test_marginal_does_not_affect_the_exit_code(tmp_path: Path, capsys) -> None:
    write_pack(tmp_path, s02_pack())
    assert main(["--all", "--root", str(tmp_path), "--colour", "never"]) == 0
    out = capsys.readouterr().out
    assert "1 marginal" in out
    assert "0 failed" in out


def test_marginal_appears_in_the_json_summary(tmp_path: Path, capsys) -> None:
    write_pack(tmp_path, s02_pack())
    assert main(["--json", "--root", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["marginal"] == 1
    assert payload["summary"]["failed"] == 0
    row = next(c for p in payload["packs"] for c in p["checks"] if c["key"] == "min_separation_km")
    assert row["marginal"] is True
    assert row["outcome"] == PASS


def test_margin_threshold_boundary_is_one_percent() -> None:
    """Just inside 1% is marginal; comfortably outside it is not."""
    actual = _haversine_km((36.6, -121.9), (36.95, -122.0))

    marginal_pack = s02_pack()
    marginal_pack["expected"]["min_separation_km"] = actual / (1.0 + MARGIN_WARN_FRAC * 0.5)
    assert (
        next(c for c in evaluate_pack(marginal_pack).checks if c.key == "min_separation_km").marginal
        is True
    )

    roomy_pack = s02_pack()
    roomy_pack["expected"]["min_separation_km"] = actual / (1.0 + MARGIN_WARN_FRAC * 2.0)
    assert (
        next(c for c in evaluate_pack(roomy_pack).checks if c.key == "min_separation_km").marginal
        is False
    )


def test_separation_genuinely_below_the_requirement_still_fails() -> None:
    pack = s02_pack()
    pack["expected"]["min_separation_km"] = 100.0
    res = evaluate_pack(pack)
    check = next(c for c in res.checks if c.key == "min_separation_km")
    assert check.outcome == FAIL
    assert "SHORT" in check.detail
    assert check.marginal is False
    assert res.ok is False


def test_two_earth_models_straddling_the_threshold_are_reported() -> None:
    """The sharpest statement of the s02 problem: the models disagree."""
    res = evaluate_pack(s02_pack())
    detail = detail_of(res, "separation_earth_model")
    assert outcomes(res)["separation_earth_model"] == SKIP
    assert "STRADDLE" in detail
    assert "WGS-84" in detail and "sphere" in detail


def test_agreeing_earth_models_say_so() -> None:
    pack = s02_pack()
    pack["expected"]["min_separation_km"] = 10.0
    detail = detail_of(evaluate_pack(pack), "separation_earth_model")
    assert "models agree" in detail
    assert "STRADDLE" not in detail


def test_ellipsoidal_model_is_a_different_answer_from_the_sphere() -> None:
    a, b = (36.6, -121.9), (36.95, -122.0)
    sphere, ellipsoid = _haversine_km(a, b), _ellipsoidal_km(a, b)
    assert ellipsoid < sphere  # WGS-84 is smaller here, by tens of metres
    assert 39.80 < ellipsoid < 39.90
    assert 39.90 < sphere < 39.95
    assert abs(sphere - ellipsoid) * 1000.0 > 50.0


def test_replay_minimum_is_recorded_as_not_derivable() -> None:
    res = evaluate_pack(s02_pack())
    assert outcomes(res)["separation_over_replay"] == SKIP
    detail = detail_of(res, "separation_over_replay")
    assert "t=0" in detail
    assert "not derivable" in detail


def test_separation_companion_rows_absent_when_the_key_is_absent() -> None:
    pack = s02_pack()
    del pack["expected"]["min_separation_km"]
    got = outcomes(evaluate_pack(pack))
    assert "separation_earth_model" not in got
    assert "separation_over_replay" not in got


# =========================================================================== #
# both contract spellings: time windows and bbox
# =========================================================================== #


def test_parse_epoch_treats_a_naive_iso_string_as_utc() -> None:
    """A local-timezone regression must fail here, so pin the exact epoch."""
    epoch, error = _parse_epoch("2024-06-15T06:00:00")
    assert error is None
    assert epoch == 1718431200.0  # 2024-06-15T06:00:00Z
    assert _parse_epoch("2024-06-15T06:00:00Z")[0] == 1718431200.0
    assert _parse_epoch("2024-06-15T06:00:00+00:00")[0] == 1718431200.0


def test_parse_epoch_accepts_numbers_and_rejects_nonsense() -> None:
    assert _parse_epoch(1718431200)[0] == 1718431200.0
    assert _parse_epoch(1718431200.5)[0] == 1718431200.5
    assert _parse_epoch("half past three")[0] is None
    assert "ISO-8601" in _parse_epoch("half past three")[1]
    assert _parse_epoch(None)[0] is None
    assert _parse_epoch(True)[0] is None  # bool is an int, but not a time


@pytest.mark.parametrize(
    "window",
    [
        {"start": "2024-06-15T06:00:00+00:00", "end": "2024-06-15T08:00:00+00:00"},
        {"start": "2024-06-15T06:00:00Z", "end": "2024-06-15T08:00:00Z"},
        {"start": "2024-06-15T06:00:00", "end": "2024-06-15T08:00:00"},
        {"start_t": 1718431200, "end_t": 1718438400},
        {"start": 1718431200.0, "end": 1718438400.0},
    ],
)
def test_timing_checks_accept_every_window_spelling(window: dict) -> None:
    """The same 7200 s window five ways; the timings must pass in all of them."""
    pack = s01_pack()
    pack["time_window"] = window
    res = evaluate_pack(pack)
    got = outcomes(res)
    for key in ("sanctuary_entry_s", "dark_at_s", "radar_contact_at_s"):
        assert got[key] == PASS, f"{key} failed for window {window}"
    assert "in window (0-7200s)" in detail_of(res, "dark_at_s")


def test_epoch_window_still_catches_an_out_of_window_timing() -> None:
    pack = s01_pack()
    pack["time_window"] = {"start_t": 1718431200, "end_t": 1718438400}
    pack["expected"]["radar_contact_at_s"] = 9999.0
    res = evaluate_pack(pack)
    assert outcomes(res)["radar_contact_at_s"] == FAIL
    assert "OUTSIDE" in detail_of(res, "radar_contact_at_s")


def test_window_bounds_reads_a_bare_start_t_end_t_at_the_pack_root() -> None:
    bounds, error = _window_bounds({"start_t": 100.0, "end_t": 460.0})
    assert error is None
    assert bounds == (100.0, 460.0)


def test_window_bounds_reads_an_ais_window() -> None:
    bounds, error = _window_bounds({"ais_window": {"start_t": 0.0, "end_t": 60.0}})
    assert error is None
    assert bounds == (0.0, 60.0)


def test_unparseable_time_value_fails_the_check_without_raising() -> None:
    """One end parses, the other is garbage: a claim the pack gets wrong."""
    pack = s01_pack()
    pack["time_window"] = {"start": "2024-06-15T06:00:00Z", "end": "half past three"}
    res = evaluate_pack(pack)  # must not raise
    assert outcomes(res)["dark_at_s"] == FAIL
    detail = detail_of(res, "dark_at_s")
    assert "unparseable" in detail
    assert "half past three" in detail


def test_unparseable_epoch_key_also_fails() -> None:
    pack = s01_pack()
    pack["time_window"] = {"start_t": 0.0, "end_t": "banana"}
    res = evaluate_pack(pack)
    assert outcomes(res)["dark_at_s"] == FAIL
    assert "end_t" in detail_of(res, "dark_at_s")


def test_wholly_unparseable_window_stays_lenient() -> None:
    """Preserved on purpose: a pack speaking no time dialect this harness knows
    cannot have its timings range-checked, so the check is omitted, not failed."""
    pack = s01_pack()
    pack["time_window"] = {"start": "not-a-date", "end": "also-not"}
    res = evaluate_pack(pack)
    assert outcomes(res)["dark_at_s"] == PASS
    assert "not parseable" in detail_of(res, "dark_at_s")


def test_bbox_array_and_object_forms_agree() -> None:
    array_form = {"bbox": [-123.2, 35.4, -121.0, 38.0]}
    object_form = {"bbox": {"lon_min": -123.2, "lat_min": 35.4, "lon_max": -121.0, "lat_max": 38.0}}
    assert _bbox_bounds(array_form) == _bbox_bounds(object_form)
    assert _bbox_bounds(array_form)[0] == (-123.2, 35.4, -121.0, 38.0)
    assert _bbox_bounds({})[0] is None


@pytest.mark.parametrize(
    "bbox",
    [
        [-123.2, 35.4, -121.0, 38.0],
        {"lon_min": -123.2, "lat_min": 35.4, "lon_max": -121.0, "lat_max": 38.0},
    ],
)
def test_both_bbox_spellings_pass_the_pack_level_check(bbox) -> None:
    pack = s01_pack()
    pack["bbox"] = bbox
    res = evaluate_pack(pack)
    assert outcomes(res)["bbox"] == PASS
    assert "lon -123.2..-121" in detail_of(res, "bbox")


def test_inverted_bbox_fails_in_either_spelling() -> None:
    pack = s01_pack()
    pack["bbox"] = [-121.0, 35.4, -123.2, 38.0]  # lon_min east of lon_max
    res = evaluate_pack(pack)
    assert outcomes(res)["bbox"] == FAIL
    assert "west of" in detail_of(res, "bbox")

    pack["bbox"] = {"lon_min": -123.2, "lat_min": 38.0, "lon_max": -121.0, "lat_max": 35.4}
    assert outcomes(evaluate_pack(pack))["bbox"] == FAIL


def test_malformed_bbox_fails_rather_than_raising() -> None:
    pack = s01_pack()
    pack["bbox"] = [-123.2, 35.4, -121.0]
    assert outcomes(evaluate_pack(pack))["bbox"] == FAIL
    pack["bbox"] = {"lon_min": -123.2, "lat_min": 35.4}
    assert outcomes(evaluate_pack(pack))["bbox"] == FAIL
    pack["bbox"] = "the whole ocean"
    res = evaluate_pack(pack)
    assert outcomes(res)["bbox"] == FAIL
    assert "expected an array of 4 or an object" in detail_of(res, "bbox")


def test_pack_without_bbox_gets_no_bbox_row() -> None:
    assert "bbox" not in outcomes(evaluate_pack(s01_pack()))


# =========================================================================== #
# pack-level checks are contained the same way keyed checkers are
# =========================================================================== #


def test_a_raising_pack_check_becomes_a_fail_and_the_run_continues(monkeypatch) -> None:
    import tracker.eval as ev

    def exploding(pack):
        raise RuntimeError("boom: pack walker fell over")

    monkeypatch.setattr(ev, "PACK_CHECKS", (exploding, ev.packcheck_bbox))
    pack = s01_pack()
    pack["bbox"] = [-123.2, 35.4, -121.0, 38.0]
    res = evaluate_pack(pack)
    details = " ".join(c.detail for c in res.checks)
    assert "boom: pack walker fell over" in details
    assert "RuntimeError" in details
    assert outcomes(res)["bbox"] == PASS  # the next pack check still ran
    assert outcomes(res)["dark_actor"] == PASS  # and the keyed checks are intact


def test_pack_level_rows_come_after_the_expected_rows(tmp_path: Path) -> None:
    pack = s01_pack()
    pack["bbox"] = [-123.2, 35.4, -121.0, 38.0]
    write_pack(tmp_path, pack)
    keys = [c.key for c in evaluate_all(tmp_path)[0].checks]
    assert keys[0] == "id_switches_max"  # first key of the expected block
    assert keys[-2:] == ["referenced_files", "bbox"]


def test_stale_note_path_and_its_verdict_survive_elision_in_the_report(tmp_path: Path) -> None:
    """Both halves of the row -- the path and "may be stale" -- must reach the
    projector, not be cut off by the detail width limit."""
    write_pack(tmp_path, s03_pack_with_stale_note_path())
    text = format_report(evaluate_all(tmp_path), use_colour=False)
    assert "geo/ofac_sdn_vessels_subset.csv" in text
    assert "may be stale" in text


def test_straddle_verdict_survives_elision_in_the_report(tmp_path: Path) -> None:
    write_pack(tmp_path, s02_pack())
    text = format_report(evaluate_all(tmp_path), use_colour=False)
    assert "STRADDLE" in text
    assert "WGS-84 39.853 km" in text
    assert "MARGIN 24 m" in text
    assert "haversine R=6371.0088 km" in text

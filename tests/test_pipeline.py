"""End-to-end tests and WebSocket contracts for the Aegis pipeline."""

import json
import os

import pytest

from aegis.graph import alerts_of, tracks_of
from aegis.main import run_pack

S01_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scenarios",
    "s01_dark_in_sanctuary",
    "ais_window.csv",
)


@pytest.fixture(scope="module")
def s02_run():
    return run_pack("s02_synthetic_demo")


def test_pipeline_tells_the_whole_story(s02_run):
    alerts = alerts_of(s02_run["mission"])
    kinds = {alert.kind for alert in alerts}
    assert {"went_dark", "intrusion", "reacquired"} <= kinds
    assert any(
        alert.kind == "intrusion" and alert.severity == "critical"
        for alert in alerts
    )
    assert (
        "assoc" in s02_run["assoc_source"]
        or "tracker" in s02_run["assoc_source"]
    )


def test_every_alert_has_headline_and_provenance(s02_run):
    for alert in alerts_of(s02_run["mission"]):
        assert alert.headline
        assert alert.detail
        assert alert.model_used


def test_deltas_are_render_ready(s02_run):
    deltas = s02_run["deltas"]
    assert len(deltas) == 121
    assert deltas[0]["zones"], "zones must ship on frame 0"
    for delta in deltas:
        assert set(delta) >= {
            "frame_idx",
            "t",
            "clock",
            "tracks",
            "measurements",
            "alerts",
            "log",
            "removed_track_ids",
            "stats",
            "financial_risk",
        }
        for track in delta["tracks"]:
            assert track["ellipse"][0] == track["ellipse"][-1]
            assert track["color"].startswith("#")
            assert {"lat", "speed_kn", "label"} <= set(track)


def test_dark_ellipse_grows_while_coasting(s02_run):
    def ghost_extent(delta):
        for track in delta["tracks"]:
            if track["dark"]:
                xs = [point[0] for point in track["ellipse"]]
                return max(xs) - min(xs)
        return None

    run = []
    for delta in s02_run["deltas"]:
        extent = ghost_extent(delta)
        if extent is not None:
            run.append(extent)
        elif run:
            break

    assert len(run) >= 3
    assert run[-1] > 3 * run[0]


def test_identity_never_reaches_a_delta(s02_run):
    blob = json.dumps(s02_run["deltas"])
    assert "ghost_1" not in blob
    assert "mmsi:" not in blob


@pytest.mark.skipif(not os.path.isfile(S01_CSV), reason="s01 window not extracted")
def test_s01_real_traffic_smoke():
    result = run_pack("s01_dark_in_sanctuary", max_frames=40)
    assert len(result["deltas"]) == 40
    assert len(tracks_of(result["mission"])) > 200
    zones = result["deltas"][0]["zones"]
    assert {zone["zone_id"] for zone in zones} == {"monterey_bay_nms"}

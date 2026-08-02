import math

from data.dark_prediction import predict_dark_vessel


def _dark_vessel(**overrides):
    vessel = {
        "mmsi": 123456789,
        "lat": 37.7,
        "lon": -123.0,
        "last_seen": 1_700_000_000.0,
        "age_s": 180.0,
        "dark": True,
        "course": 95.0,
        "heading": 96.0,
        "speed_kn": 12.0,
        "rate_of_turn": 0.1,
        "history": [[37.7, -123.02], [37.7, -123.01], [37.7, -123.0]],
    }
    return {**vessel, **overrides}


def test_prediction_is_deterministic_and_probability_is_normalized():
    first = predict_dark_vessel(_dark_vessel())
    second = predict_dark_vessel(_dark_vessel())

    assert first == second
    assert first["samples"] == 600
    assert 2 <= first["scenario_count"] <= 6
    assert abs(sum(row["probability"] for row in first["scenarios"]) - 1.0) < 0.001
    expected_points = math.ceil(first["path_minutes_from_last_fix"] / 2) + 1
    assert all(len(row["path"]) == expected_points for row in first["scenarios"])


def test_missing_navigation_data_creates_more_uncertainty_branches():
    complete = predict_dark_vessel(_dark_vessel())
    sparse = predict_dark_vessel(_dark_vessel(
        course=None,
        heading=None,
        speed_kn=None,
        rate_of_turn=None,
        history=[],
        age_s=900.0,
    ))

    assert sparse["scenario_count"] > complete["scenario_count"]
    assert len(sparse["uncertainty_drivers"]) > len(complete["uncertainty_drivers"])
    assert sparse["signal_availability"]["satellite_observation"] is False


def test_active_vessels_are_not_predicted():
    try:
        predict_dark_vessel(_dark_vessel(dark=False))
    except ValueError as exc:
        assert "AIS-silent" in str(exc)
    else:
        raise AssertionError("active vessel should not receive a dark prediction")


def test_copernicus_current_changes_paths_and_is_reported():
    vessel = _dark_vessel()
    baseline = predict_dark_vessel(vessel)
    with_current = predict_dark_vessel(vessel, {
        "configured": True,
        "available": True,
        "source": "Copernicus Marine Service",
        "center": {
            "east_mps": 0.4,
            "north_mps": -0.2,
            "speed_mps": 0.4472,
            "bearing_deg": 116.6,
        },
        "vectors": [],
    })

    assert with_current["inputs"]["copernicus_surface_current"] is True
    assert with_current["signal_availability"]["ocean_currents"] is True
    assert with_current["scenarios"] != baseline["scenarios"]


def test_behavior_hypotheses_include_turnaround_drift_and_curved_routes():
    prediction = predict_dark_vessel(_dark_vessel(age_s=180.0))
    scenarios = prediction["scenarios"]
    behaviors = {scenario["behavior"] for scenario in scenarios}

    assert "course_reversal" in behaviors
    assert "drift" in behaviors
    assert "maneuver" in behaviors
    assert abs(sum(row["probability"] for row in scenarios) - 1.0) < 0.001

    reversal = next(
        scenario for scenario in scenarios
        if scenario["behavior"] == "course_reversal"
    )
    path = reversal["path"]
    segment_bearings = [
        math.degrees(math.atan2(
            (end[1] - start[1]) * math.cos(math.radians(start[0])),
            end[0] - start[0],
        )) % 360
        for start, end in zip(path, path[1:])
        if start != end
    ]
    total_turn = sum(
        abs((end - start + 180) % 360 - 180)
        for start, end in zip(segment_bearings, segment_bearings[1:])
    )
    assert total_turn > 80


def test_noaa_wind_changes_paths_and_is_reported():
    vessel = _dark_vessel(age_s=900.0)
    baseline = predict_dark_vessel(vessel)
    with_wind = predict_dark_vessel(vessel, weather_conditions={
        "configured": True,
        "available": True,
        "source": "NOAA Global Forecast System",
        "center": {
            "east_mps": 8.0,
            "north_mps": -2.0,
            "speed_mps": 8.246,
            "bearing_deg": 104.0,
        },
    })

    assert with_wind["inputs"]["noaa_gfs_wind"] is True
    assert with_wind["signal_availability"]["wind_forcing"] is True
    assert with_wind["scenarios"] != baseline["scenarios"]


def test_extreme_reported_turn_rate_settles_instead_of_spiraling():
    prediction = predict_dark_vessel(_dark_vessel(
        rate_of_turn=10.0,
        age_s=600.0,
    ))

    for scenario in prediction["scenarios"]:
        if scenario["behavior"] == "drift":
            continue
        path = scenario["path"]
        mean_lat = math.radians(sum(point[0] for point in path) / len(path))

        def distance(start, end):
            return math.hypot(
                (end[1] - start[1]) * 111_320 * math.cos(mean_lat),
                (end[0] - start[0]) * 111_320,
            )

        segment_bearings = [
            math.degrees(math.atan2(
                (end[1] - start[1]) * math.cos(mean_lat),
                end[0] - start[0],
            )) % 360
            for start, end in zip(path, path[1:])
            if distance(start, end) > 1
        ]
        total_turn = sum(
            abs((end - start + 180) % 360 - 180)
            for start, end in zip(segment_bearings, segment_bearings[1:])
        )
        assert total_turn < 220

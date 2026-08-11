"""Monte Carlo prediction for AIS-silent vessels using observed navigation data."""

from __future__ import annotations

import math
import random
from typing import Any

from data.maritime_context import maritime_context

EARTH_RADIUS_M = 6_371_008.8
KNOT_TO_MPS = 0.514444
HORIZON_MINUTES = 30
STEP_MINUTES = 2
SAMPLE_COUNT = 600
MAX_SILENCE_MINUTES = 24 * 60
CONFIDENCE_LEVELS = (0.50, 0.80, 0.95)


def _destination(lat: float, lon: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    angular = distance_m / EARTH_RADIUS_M
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular)
        + math.cos(lat1) * math.sin(angular) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular) * math.cos(lat1),
        math.cos(angular) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), ((math.degrees(lon2) + 180) % 360) - 180


def _bearing_between(start: tuple[float, float], end: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, start)
    lat2, lon2 = map(math.radians, end)
    delta_lon = lon2 - lon1
    y = math.sin(delta_lon) * math.cos(lat2)
    x = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    )
    return math.degrees(math.atan2(y, x)) % 360


def _angle_delta(target: float, source: float) -> float:
    return (target - source + 180) % 360 - 180


def _history_navigation(vessel: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    samples = vessel.get("history_samples") or []
    valid = [
        sample for sample in samples
        if isinstance(sample, dict)
        and "lat" in sample and "lon" in sample
    ]
    if len(valid) < 2:
        points = vessel.get("history") or []
        if len(points) < 2:
            return None, None, None
        return (
            _bearing_between(tuple(points[-2]), tuple(points[-1])),
            None,
            None,
        )

    latest = valid[-1]
    previous = valid[-2]
    course = _bearing_between(
        (float(previous["lat"]), float(previous["lon"])),
        (float(latest["lat"]), float(latest["lon"])),
    )
    speed = None
    elapsed = float(latest.get("time", 0)) - float(previous.get("time", 0))
    if elapsed > 1:
        mean_lat = math.radians((float(previous["lat"]) + float(latest["lat"])) / 2)
        north_m = (float(latest["lat"]) - float(previous["lat"])) * 111_320
        east_m = (
            (float(latest["lon"]) - float(previous["lon"]))
            * 111_320
            * max(0.1, math.cos(mean_lat))
        )
        speed = math.hypot(east_m, north_m) / elapsed / KNOT_TO_MPS

    history_turn = None
    if len(valid) >= 3:
        older = valid[-3]
        prior_course = _bearing_between(
            (float(older["lat"]), float(older["lon"])),
            (float(previous["lat"]), float(previous["lon"])),
        )
        turn_elapsed_min = (
            float(latest.get("time", 0)) - float(older.get("time", 0))
        ) / 120
        if turn_elapsed_min > 0:
            history_turn = _angle_delta(course, prior_course) / turn_elapsed_min
    return course, speed, history_turn


def _valid_bearing(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and 0 <= number < 360


def _valid_speed(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and 0 <= number < 80


def _valid_turn(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and -20 <= number <= 20


def _nearest_vector(
    conditions: dict[str, Any] | None,
    lat: float,
    lon: float,
) -> dict[str, Any]:
    if not conditions or not conditions.get("available"):
        return {}
    vectors = conditions.get("vectors") or []
    if vectors:
        return min(
            vectors,
            key=lambda vector: (
                (float(vector["lat"]) - lat) ** 2
                + (float(vector["lon"]) - lon) ** 2
            ),
        )
    return conditions.get("center") or {}


def _weighted_choice(
    rng: random.Random,
    weights: dict[str, float],
) -> str:
    draw = rng.random() * sum(weights.values())
    cumulative = 0.0
    for name, weight in weights.items():
        cumulative += weight
        if draw <= cumulative:
            return name
    return next(reversed(weights))


def _behavior_priors(
    *,
    base_speed: float,
    navigation_status: int,
    turn_rate: float,
    age_s: float,
    missing_count: int,
) -> dict[str, float]:
    """Return visible maneuver priors conditioned on the latest AIS state."""
    constrained = navigation_status in {1, 5, 6}
    if constrained or base_speed < 0.8:
        return {
            "drift": 0.43,
            "stopped": 0.20,
            "slow_maneuver": 0.20,
            "maintain_course": 0.12,
            "course_reversal": 0.05,
        }

    uncertainty = min(1.0, age_s / 3600 + missing_count / 8)
    observed_turn = min(1.0, abs(turn_rate) / 1.5)
    maneuver = 0.18 + uncertainty * 0.12 + observed_turn * 0.12
    reversal = 0.035 + uncertainty * 0.045
    drift = 0.035 + uncertainty * 0.04
    stopped = 0.025 + uncertainty * 0.035
    slow_maneuver = 0.04 + observed_turn * 0.04
    maintain = max(
        0.2,
        1.0 - maneuver - reversal - drift - stopped - slow_maneuver,
    )
    weights = {
        "maintain_course": maintain,
        "maneuver": maneuver,
        "slow_maneuver": slow_maneuver,
        "course_reversal": reversal,
        "drift": drift,
        "stopped": stopped,
    }
    total = sum(weights.values())
    return {name: weight / total for name, weight in weights.items()}


def _windage_range(vessel: dict[str, Any]) -> tuple[float, float]:
    """Fraction of 10-meter wind transferred to an unpowered vessel."""
    try:
        ship_type = int(vessel.get("ship_type"))
    except (TypeError, ValueError):
        ship_type = -1
    try:
        draught = float(vessel.get("draught_m") or 0.0)
    except (TypeError, ValueError):
        draught = 0.0
    if 70 <= ship_type <= 89 or draught >= 7:
        return 0.012, 0.025
    if 30 <= ship_type <= 39 or 60 <= ship_type <= 69:
        return 0.02, 0.04
    return 0.015, 0.04


def _adaptive_time_steps(
    elapsed_minutes: float,
    forecast_minutes: float = HORIZON_MINUTES,
) -> list[dict[str, float | str]]:
    """Build bounded simulation steps without losing near-fix detail."""
    elapsed = max(0.0, min(MAX_SILENCE_MINUTES, elapsed_minutes))
    target = elapsed + max(0.0, forecast_minutes)
    minute = 0.0
    steps: list[dict[str, float | str]] = []
    while minute < target - 1e-9:
        if minute < elapsed - 1e-9:
            phase = "elapsed"
            if minute < min(elapsed, 60.0):
                nominal = 2.0
                boundary = min(elapsed, 60.0)
            elif minute < min(elapsed, 360.0):
                nominal = 5.0
                boundary = min(elapsed, 360.0)
            else:
                nominal = 15.0
                boundary = elapsed
        else:
            phase = "forecast"
            nominal = 2.0
            boundary = target
        duration = min(nominal, boundary - minute, target - minute)
        if duration <= 1e-9:
            break
        minute += duration
        steps.append({
            "minutes": duration,
            "phase": phase,
            "end_minute": minute,
        })
    return steps


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return a deterministic monotonic-chain hull in local meter space."""
    unique = sorted(set(points))
    if len(unique) <= 1:
        return unique

    def cross(
        origin: tuple[float, float],
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> float:
        return (
            (a[0] - origin[0]) * (b[1] - origin[1])
            - (a[1] - origin[1]) * (b[0] - origin[0])
        )

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _confidence_regions(
    samples: list[dict[str, Any]],
    origin: tuple[float, float],
    path_index: int,
) -> list[dict[str, Any]]:
    """Build nested endpoint regions from central Monte Carlo samples."""
    lat0, lon0 = origin
    cos_lat = max(0.1, math.cos(math.radians(lat0)))

    def to_xy(point: list[float]) -> tuple[float, float]:
        return (
            (float(point[1]) - lon0) * 111_320 * cos_lat,
            (float(point[0]) - lat0) * 111_320,
        )

    points = [to_xy(sample["path"][path_index]) for sample in samples]
    center_x = sorted(point[0] for point in points)[len(points) // 2]
    center_y = sorted(point[1] for point in points)[len(points) // 2]
    ordered = sorted(
        points,
        key=lambda point: math.hypot(
            point[0] - center_x,
            point[1] - center_y,
        ),
    )
    regions = []
    for level in CONFIDENCE_LEVELS:
        count = max(3, min(len(ordered), math.ceil(len(ordered) * level)))
        included = ordered[:count]
        hull = _convex_hull(included)
        spread_radius = max(
            100.0,
            max(
                math.hypot(point[0] - center_x, point[1] - center_y)
                for point in included
            ),
        )
        if len(hull) < 3:
            hull = [
                (
                    center_x + math.sin(index * math.pi / 4) * spread_radius,
                    center_y + math.cos(index * math.pi / 4) * spread_radius,
                )
                for index in range(8)
            ]
        polygon = [
            [
                round(lat0 + north / 111_320, 5),
                round(lon0 + east / (111_320 * cos_lat), 5),
            ]
            for east, north in hull
        ]
        if polygon:
            polygon.append(polygon[0])
        regions.append({
            "level": int(level * 100),
            "center": [
                round(lat0 + center_y / 111_320, 5),
                round(lon0 + center_x / (111_320 * cos_lat), 5),
            ],
            "spread_radius_m": round(spread_radius),
            "reach_radius_m": round(
                max(math.hypot(east, north) for east, north in included)
            ),
            "polygon": polygon,
            "sample_count": count,
        })
    return regions


def _desired_bearing_for_behavior(
    rng: random.Random,
    behavior: str,
    bearing: float,
    heading_sigma: float,
) -> float:
    if behavior == "course_reversal":
        return (bearing + 180 + rng.gauss(0, 18 + heading_sigma * 0.2)) % 360
    if behavior == "maneuver":
        offset = max(
            -120.0,
            min(120.0, rng.gauss(0, max(32.0, heading_sigma * 1.8))),
        )
        if abs(offset) < 20:
            offset = 20 if rng.random() < 0.5 else -20
        return (bearing + offset) % 360
    if behavior == "slow_maneuver":
        return (
            bearing
            + (1 if rng.random() < 0.5 else -1) * rng.uniform(25, 65)
        ) % 360
    return (bearing + rng.gauss(0, heading_sigma * 0.35)) % 360


def _segment_intersects_land(
    context: Any,
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    """Check long adaptive steps at intervals small enough to catch coastlines."""
    end_status = context.terrain_status(*end)
    if not end_status["available"]:
        return False
    if end_status["on_land"]:
        return True
    mean_lat = math.radians((start[0] + end[0]) / 2)
    north_m = (end[0] - start[0]) * 111_320
    lon_delta = _angle_delta(end[1], start[1])
    east_m = lon_delta * 111_320 * max(0.1, math.cos(mean_lat))
    checks = max(1, math.ceil(math.hypot(east_m, north_m) / 1500))
    for index in range(1, checks):
        fraction = index / checks
        point = (
            start[0] + (end[0] - start[0]) * fraction,
            ((start[1] + lon_delta * fraction + 180) % 360) - 180,
        )
        if context.terrain_status(*point)["on_land"]:
            return True
    return False


def _cluster(samples: list[dict[str, Any]], count: int, origin: tuple[float, float]) -> list[dict[str, Any]]:
    lat0, lon0 = origin
    cos_lat = max(0.1, math.cos(math.radians(lat0)))

    def xy(sample: dict[str, Any]) -> tuple[float, float]:
        lat, lon = sample["path"][-1]
        return ((lon - lon0) * 111_320 * cos_lat, (lat - lat0) * 111_320)

    points = [xy(sample) for sample in samples]
    behavior_groups: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        behavior_groups.setdefault(sample["behavior"], []).append(index)

    allocations = {behavior: 1 for behavior in behavior_groups}
    while sum(allocations.values()) < count:
        behavior = max(
            behavior_groups,
            key=lambda name: len(behavior_groups[name]) / allocations[name],
        )
        allocations[behavior] += 1

    scenarios = []
    for behavior, group_indices in behavior_groups.items():
        cluster_count = min(allocations[behavior], len(group_indices))
        ordered = sorted(
            group_indices,
            key=lambda index: math.atan2(points[index][0], points[index][1]),
        )
        centroids = [
            points[
                ordered[
                    min(
                        len(ordered) - 1,
                        int((cluster + 0.5) * len(ordered) / cluster_count),
                    )
                ]
            ]
            for cluster in range(cluster_count)
        ]
        assignments = [0] * len(group_indices)
        for _ in range(8):
            for local_index, sample_index in enumerate(group_indices):
                point = points[sample_index]
                assignments[local_index] = min(
                    range(cluster_count),
                    key=lambda cluster: (
                        (point[0] - centroids[cluster][0]) ** 2
                        + (point[1] - centroids[cluster][1]) ** 2
                    ),
                )
            for cluster in range(cluster_count):
                members = [
                    points[group_indices[local_index]]
                    for local_index, assigned in enumerate(assignments)
                    if assigned == cluster
                ]
                if members:
                    centroids[cluster] = (
                        sum(point[0] for point in members) / len(members),
                        sum(point[1] for point in members) / len(members),
                    )

        for cluster in range(cluster_count):
            members = [
                group_indices[local_index]
                for local_index, assigned in enumerate(assignments)
                if assigned == cluster
            ]
            if not members:
                continue
            center = centroids[cluster]
            representative = min(
                members,
                key=lambda index: (
                    (points[index][0] - center[0]) ** 2
                    + (points[index][1] - center[1]) ** 2
                ),
            )
            distances = sorted(
                math.hypot(
                    points[index][0] - center[0],
                    points[index][1] - center[1],
                )
                for index in members
            )
            radius = distances[
                min(len(distances) - 1, int(len(distances) * 0.9))
            ]
            path = samples[representative]["path"]
            distance_m = sum(
                math.hypot(
                    (path[i][1] - path[i - 1][1]) * 111_320 * cos_lat,
                    (path[i][0] - path[i - 1][0]) * 111_320,
                )
                for i in range(1, len(path))
            )
            scenarios.append({
                "probability": round(len(members) / len(samples), 4),
                "path": [
                    [round(lat, 5), round(lon, 5)]
                    for lat, lon in path
                ],
                "distance_nm": round(distance_m / 1852, 1),
                "uncertainty_radius_m": round(max(100.0, radius)),
                "behavior": behavior,
                "behavior_sequence": samples[representative].get(
                    "behavior_sequence",
                    [behavior],
                ),
            })
    scenarios.sort(key=lambda scenario: scenario["probability"], reverse=True)
    return scenarios


def _simulation_ensemble(
    samples: list[dict[str, Any]],
    timeline: list[float],
    current_path_index: int,
    max_points: int = 72,
) -> dict[str, Any]:
    """Return all samples in a compact, canvas-friendly representation."""
    path_length = len(samples[0]["path"])
    if path_length <= max_points:
        indices = list(range(path_length))
    else:
        indices = sorted({
            0,
            current_path_index,
            path_length - 1,
            *[
                round(index * (path_length - 1) / (max_points - 1))
                for index in range(max_points)
            ],
        })
    compact_paths = []
    min_lat = 90.0
    min_lon = 180.0
    max_lat = -90.0
    max_lon = -180.0
    for sample in samples:
        coordinates: list[float] = []
        for index in indices:
            point = sample["path"][index]
            point_lat = round(float(point[0]), 5)
            point_lon = round(float(point[1]), 5)
            coordinates.extend((point_lat, point_lon))
            min_lat = min(min_lat, point_lat)
            min_lon = min(min_lon, point_lon)
            max_lat = max(max_lat, point_lat)
            max_lon = max(max_lon, point_lon)
        compact_paths.append({
            "behavior": sample["behavior"],
            "path": coordinates,
        })
    return {
        "count": len(compact_paths),
        "source": "same deterministic Monte Carlo model as the map prediction",
        "temporally_downsampled": len(indices) < path_length,
        "original_points_per_path": path_length,
        "display_points_per_path": len(indices),
        "current_path_index": indices.index(current_path_index),
        "timeline_minutes": [round(float(timeline[index]), 2) for index in indices],
        "bounds": [[min_lat, min_lon], [max_lat, max_lon]],
        "paths": compact_paths,
    }


def predict_dark_vessel(
    vessel: dict[str, Any],
    ocean_conditions: dict[str, Any] | None = None,
    weather_conditions: dict[str, Any] | None = None,
    include_samples: bool = False,
) -> dict[str, Any]:
    """Return deterministic full-gap Monte Carlo branches from the last AIS fix."""
    if not vessel.get("dark"):
        raise ValueError("prediction is only valid for AIS-silent vessels")

    lat = float(vessel["lat"])
    lon = float(vessel["lon"])
    age_s = max(0.0, float(vessel.get("age_s", 0.0)))
    modeled_silence_minutes = min(MAX_SILENCE_MINUTES, age_s / 60)
    unmodeled_silence_minutes = max(0.0, age_s / 60 - modeled_silence_minutes)
    has_course = _valid_bearing(vessel.get("course"))
    has_heading = _valid_bearing(vessel.get("heading"))
    has_speed = _valid_speed(vessel.get("speed_kn"))
    history_points = len(vessel.get("history") or [])
    history_course, history_speed, history_turn = _history_navigation(vessel)
    base_bearing = (
        float(vessel["course"]) if has_course
        else float(vessel["heading"]) if has_heading
        else history_course if history_course is not None
        else 0.0
    )
    if history_course is not None and (has_course or has_heading):
        difference = _angle_delta(history_course, base_bearing)
        if abs(difference) <= 45:
            base_bearing = (base_bearing + difference * 0.2) % 360
    history_speed_valid = history_speed is not None and _valid_speed(history_speed)
    base_speed = (
        float(vessel["speed_kn"]) if has_speed
        else float(history_speed) if history_speed_valid
        else 5.0
    )
    raw_turn = vessel.get("rate_of_turn")
    has_reported_turn = _valid_turn(raw_turn)
    has_history_turn = history_turn is not None and math.isfinite(history_turn)
    if has_reported_turn and has_history_turn:
        turn_rate = float(raw_turn) * 0.7 + float(history_turn) * 0.3
    elif has_reported_turn:
        turn_rate = float(raw_turn)
    elif has_history_turn:
        turn_rate = float(history_turn)
    else:
        turn_rate = 0.0
    turn_rate = max(-1.5, min(1.5, turn_rate))
    has_turn_rate = has_reported_turn or has_history_turn
    try:
        navigation_status = int(vessel.get("navigation_status"))
    except (TypeError, ValueError):
        navigation_status = -1
    constrained_status = navigation_status in {1, 5, 6}
    if constrained_status and base_speed < 3:
        base_speed = min(base_speed, 0.5)

    missing = [
        name
        for name, available in (
            ("course_over_ground", has_course),
            ("true_heading", has_heading),
            ("speed_over_ground", has_speed),
            ("rate_of_turn", has_turn_rate),
            (
                "track_history",
                history_points >= 3
                or len(vessel.get("history_samples") or []) >= 3,
            ),
        )
        if not available
    ]
    heading_sigma = 4.0 + len(missing) * 4.0
    speed_sigma = max(0.35, base_speed * 0.1) + len(missing) * 0.3
    if constrained_status:
        heading_sigma += 8.0
        speed_sigma = min(speed_sigma, 0.5)
    desired_scenarios = max(
        5,
        min(6, 5 + len(missing) + int(modeled_silence_minutes // 15)),
    )
    seed = int(vessel.get("mmsi", 0)) ^ int(float(vessel.get("last_seen", 0.0)))
    context = maritime_context()
    behavior_priors = _behavior_priors(
        base_speed=base_speed,
        navigation_status=navigation_status,
        turn_rate=turn_rate,
        age_s=age_s,
        missing_count=len(missing),
    )
    wind = (
        weather_conditions.get("center", {})
        if weather_conditions and weather_conditions.get("available")
        else {}
    )
    wind_east = float(wind.get("east_mps", 0.0))
    wind_north = float(wind.get("north_mps", 0.0))
    wind_speed = math.hypot(wind_east, wind_north)
    wind_bearing = math.degrees(math.atan2(wind_east, wind_north)) % 360
    windage_low, windage_high = _windage_range(vessel)
    wave = (ocean_conditions or {}).get("wave") or {}
    wave_height = (
        float(wave.get("height_m", 0.0)) if wave.get("available") else 0.0
    )
    time_steps = _adaptive_time_steps(modeled_silence_minutes)
    current_path_index = sum(
        step["phase"] == "elapsed" for step in time_steps
    )
    samples: list[dict[str, Any]] = []
    constrained_samples = 0
    transition_count = 0

    for sample_index in range(SAMPLE_COUNT):
        # Isolate each sample's random stream so optional forecast forcing
        # cannot alter another sample's historical propagation.
        rng = random.Random(seed ^ ((sample_index + 1) * 0x9E3779B1))
        behavior = _weighted_choice(rng, behavior_priors)
        behavior_sequence = [behavior]
        behavior_minutes = {name: 0.0 for name in behavior_priors}
        bearing = (
            base_bearing + rng.gauss(0, max(1.0, heading_sigma * 0.15))
        ) % 360
        desired_bearing = _desired_bearing_for_behavior(
            rng,
            behavior,
            base_bearing,
            heading_sigma,
        )
        sampled_base_speed = max(0.0, rng.gauss(base_speed, speed_sigma))
        speed = sampled_base_speed
        sample_turn = max(
            -2.5,
            min(2.5, turn_rate + rng.gauss(0, 0.08 + heading_sigma / 180)),
        )
        windage = rng.uniform(windage_low, windage_high)
        sample_lat, sample_lon = lat, lon
        path = [[sample_lat, sample_lon]]
        constrained = False
        next_behavior_decision = rng.uniform(20.0, 45.0)

        for step_index, step in enumerate(time_steps):
            duration = float(step["minutes"])
            minute = float(step["end_minute"])
            phase = str(step["phase"])
            scale = math.sqrt(max(0.1, duration / STEP_MINUTES))
            behavior_minutes[behavior] += duration

            if minute >= next_behavior_decision:
                transition_probability = min(
                    0.80,
                    0.16
                    + 0.44 * minute / MAX_SILENCE_MINUTES
                    + 0.035 * len(missing),
                )
                if rng.random() < transition_probability:
                    transition_weights = dict(behavior_priors)
                    transition_weights[behavior] *= 0.35
                    next_behavior = _weighted_choice(rng, transition_weights)
                    if next_behavior != behavior:
                        behavior = next_behavior
                        behavior_sequence.append(behavior)
                        transition_count += 1
                        desired_bearing = _desired_bearing_for_behavior(
                            rng,
                            behavior,
                            bearing,
                            heading_sigma + min(30.0, minute / 30),
                        )
                decision_window = 45.0 if minute < 360 else 120.0
                next_behavior_decision += rng.uniform(
                    decision_window * 0.55,
                    decision_window,
                )

            if behavior == "maintain_course":
                desired_bearing = (
                    desired_bearing
                    + rng.gauss(0, max(0.08, heading_sigma / 160) * scale)
                ) % 360
            elif behavior == "slow_maneuver":
                desired_bearing = (
                    desired_bearing
                    + math.sin((step_index + 1) * 0.55)
                    * min(3.0, heading_sigma / 12)
                    * scale
                ) % 360

            target_speed = (
                0.0
                if behavior in {"drift", "stopped"}
                else min(sampled_base_speed, 3.0)
                if behavior == "slow_maneuver"
                else sampled_base_speed
            )
            max_turn = 7.0 if max(speed, target_speed) < 8 else 5.0
            desired_turn = max(
                -max_turn,
                min(
                    max_turn,
                    _angle_delta(desired_bearing, bearing)
                    / (5.0 if behavior == "course_reversal" else 7.5),
                ),
            )
            historical_gap_noise = (
                min(0.16, minute / MAX_SILENCE_MINUTES * 0.16)
                if phase == "elapsed"
                else 0.0
            )
            wave_turn_noise = (
                min(0.12, wave_height / 30)
                if phase == "forecast"
                else 0.0
            )
            turn_noise = (
                0.04
                + heading_sigma / 240
                + historical_gap_noise
                + wave_turn_noise
            ) * scale
            turn_response = 1 - 0.58 ** (duration / STEP_MINUTES)
            sample_turn = max(
                -max_turn,
                min(
                    max_turn,
                    sample_turn
                    + (desired_turn - sample_turn) * turn_response
                    + rng.gauss(0, turn_noise),
                ),
            )
            bearing = (bearing + sample_turn * duration) % 360

            if behavior == "drift":
                speed = max(0.0, speed * 0.58 ** (duration / STEP_MINUTES))
            elif behavior == "stopped":
                speed = max(0.0, speed * 0.35 ** (duration / STEP_MINUTES))
            else:
                speed_response = 1 - 0.75 ** (duration / STEP_MINUTES)
                speed = max(
                    0.0,
                    speed
                    + (target_speed - speed) * speed_response
                    + rng.gauss(0, speed_sigma * 0.035 * scale),
                )
            next_lat, next_lon = sample_lat, sample_lon
            if speed > 0:
                next_lat, next_lon = _destination(
                    next_lat,
                    next_lon,
                    bearing,
                    speed * KNOT_TO_MPS * duration * 60,
                )

            # Cached observations describe current conditions, so they are only
            # evidence for the forward phase, never the historical AIS gap.
            if phase == "forecast":
                current = _nearest_vector(
                    ocean_conditions,
                    sample_lat,
                    sample_lon,
                )
                current_east = float(current.get("east_mps", 0.0))
                current_north = float(current.get("north_mps", 0.0))
                current_speed = math.hypot(current_east, current_north)
                if current_speed > 0:
                    current_bearing = (
                        math.degrees(
                            math.atan2(current_east, current_north)
                        ) % 360
                    )
                    sampled_current = max(
                        0.0,
                        rng.gauss(
                            current_speed,
                            max(0.02, current_speed * 0.1),
                        ),
                    )
                    next_lat, next_lon = _destination(
                        next_lat,
                        next_lon,
                        current_bearing + rng.gauss(0, 3.0),
                        sampled_current * duration * 60,
                    )

                if wind_speed > 0:
                    leeway_factor = (
                        windage
                        if behavior in {"drift", "stopped"}
                        else windage * 0.2
                    )
                    sampled_leeway = max(
                        0.0,
                        rng.gauss(
                            wind_speed * leeway_factor,
                            wind_speed * leeway_factor * 0.2,
                        ),
                    )
                    next_lat, next_lon = _destination(
                        next_lat,
                        next_lon,
                        wind_bearing + rng.gauss(0, 8.0),
                        sampled_leeway * duration * 60,
                    )

            if _segment_intersects_land(
                context,
                (sample_lat, sample_lon),
                (next_lat, next_lon),
            ):
                constrained = True
                bearing = (
                    base_bearing + (35 if rng.random() < 0.5 else -35)
                ) % 360
                next_lat, next_lon = _destination(
                    sample_lat,
                    sample_lon,
                    bearing,
                    speed * KNOT_TO_MPS * duration * 30,
                )
                if _segment_intersects_land(
                    context,
                    (sample_lat, sample_lon),
                    (next_lat, next_lon),
                ):
                    next_lat, next_lon = sample_lat, sample_lon
                    speed *= 0.5
            sample_lat, sample_lon = next_lat, next_lon
            path.append([sample_lat, sample_lon])

        constrained_samples += int(constrained)
        dominant_behavior = max(
            behavior_minutes,
            key=behavior_minutes.get,
        )
        # A stopped vessel is represented as the stationary end of the drift
        # hypothesis while its transition sequence still records the stop.
        cluster_behavior = (
            "drift" if dominant_behavior == "stopped" else dominant_behavior
        )
        samples.append({
            "path": path,
            "behavior": cluster_behavior,
            "behavior_sequence": behavior_sequence,
        })

    scenarios = _cluster(samples, desired_scenarios, (lat, lon))
    for scenario in scenarios:
        scenario["current_path_index"] = current_path_index
    elapsed_regions = _confidence_regions(
        samples,
        (lat, lon),
        current_path_index,
    )
    forecast_regions = _confidence_regions(
        samples,
        (lat, lon),
        len(time_steps),
    )
    terrain_here = context.terrain_status(lat, lon)
    behavior_counts = {
        behavior: sum(
            behavior in sample["behavior_sequence"] for sample in samples
        )
        for behavior in behavior_priors
    }
    has_ocean = bool(ocean_conditions and ocean_conditions.get("available"))
    has_waves = bool(wave.get("available"))
    has_wind = bool(weather_conditions and weather_conditions.get("available"))
    if modeled_silence_minutes >= 360:
        confidence_label = "low"
        confidence_reason = "AIS gap exceeds 6 hours"
    elif modeled_silence_minutes >= 60:
        confidence_label = "moderate"
        confidence_reason = "AIS gap exceeds 1 hour"
    else:
        confidence_label = "higher"
        confidence_reason = "AIS gap is under 1 hour"
    step_counts: dict[str, int] = {}
    for step in time_steps:
        label = f"{float(step['minutes']):g}"
        step_counts[label] = step_counts.get(label, 0) + 1
    path_timeline = [
        0.0,
        *[round(float(step["end_minute"]), 2) for step in time_steps],
    ]
    path_minutes = modeled_silence_minutes + HORIZON_MINUTES
    result = {
        "model": "monte_carlo_navigation_v4",
        "samples": SAMPLE_COUNT,
        "horizon_minutes": HORIZON_MINUTES,
        "step_minutes": STEP_MINUTES,
        "adaptive_step_counts": step_counts,
        "path_timeline_minutes": path_timeline,
        "modeled_silence_minutes": round(modeled_silence_minutes, 1),
        "unmodeled_silence_minutes": round(unmodeled_silence_minutes, 1),
        "path_minutes_from_last_fix": round(path_minutes, 1),
        "current_path_index": current_path_index,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "elapsed_confidence_regions": elapsed_regions,
        "forecast_confidence_regions": forecast_regions,
        "confidence_regions": forecast_regions,
        "confidence": {
            "label": confidence_label,
            "reason": confidence_reason,
        },
        "inputs": {
            "mmsi": vessel.get("mmsi"),
            "last_position": [lat, lon],
            "ais_silence_s": round(age_s, 1),
            "course_deg": (
                base_bearing
                if has_course or has_heading or history_course is not None
                else None
            ),
            "speed_kn": (
                base_speed if has_speed or history_speed_valid else None
            ),
            "history_points": history_points,
            "history_course_deg": (
                round(history_course, 1) if history_course is not None else None
            ),
            "history_speed_kn": (
                round(float(history_speed), 1) if history_speed_valid else None
            ),
            "turn_rate_deg_min": round(turn_rate, 3) if has_turn_rate else None,
            "navigation_status": (
                navigation_status if navigation_status >= 0 else None
            ),
            "regional_terrain_constraint": terrain_here["available"],
            "copernicus_surface_current": has_ocean,
            "copernicus_wave_forcing": has_waves,
            "noaa_gfs_wind": has_wind,
            "windage_range": [windage_low, windage_high],
        },
        "behavior_priors": {
            name: round(weight, 4)
            for name, weight in behavior_priors.items()
        },
        "behavior_sample_counts": behavior_counts,
        "behavior_transition_count": transition_count,
        "uncertainty_drivers": [
            *missing,
            "historical_environment_unavailable",
        ],
        "environment_evidence": {
            "historical_gap": {
                "available": False,
                "applied": False,
                "reason": (
                    "Historical current, wind, and wave observations are not "
                    "available from the configured live-data clients"
                ),
            },
            "forward_outlook": {
                "current_applied": has_ocean,
                "wave_applied": has_waves,
                "wind_applied": has_wind,
            },
        },
        "signal_availability": {
            "ais_vhf_navigation": True,
            "independent_radio_frequency": False,
            "satellite_observation": False,
            "peer_vessel_observation": False,
            "global_terrain": False,
            "ocean_currents": has_ocean,
            "wave_forcing": has_waves,
            "wind_forcing": has_wind,
        },
        "terrain_constrained_samples": constrained_samples,
        "ocean_conditions": ocean_conditions or {
            "configured": False,
            "available": False,
            "source": "Copernicus Marine Service",
        },
        "weather_conditions": weather_conditions or {
            "configured": True,
            "available": False,
            "source": "NOAA Global Forecast System",
        },
    }
    if include_samples:
        result["simulation_ensemble"] = _simulation_ensemble(
            samples,
            path_timeline,
            current_path_index,
        )
    return result

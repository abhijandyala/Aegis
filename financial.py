"""Sourced, conditional maritime response-cost options."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_RATES_PATH = Path(__file__).with_name("data") / "uscg_fy26_rates.json"
_RATES = json.loads(_RATES_PATH.read_text(encoding="utf-8"))
_SOURCE = _RATES["source"]

_SIGNAL_LABELS = {
    "intrusion": "Protected-water alert",
    "went_dark": "AIS transmission lost",
    "dark_monitoring": "AIS-silent contact",
    "reacquired": "AIS transmission resumed",
    "identity_change": "Vessel identity changed",
}


def _review_option(track_id: str, severity: str) -> dict[str, Any]:
    personnel = _RATES["personnel_hourly_usd"]
    return {
        "tier": "desk_review",
        "track_id": track_id,
        "severity": severity,
        "label": f"Desk review{f' · {track_id}' if track_id else ''}",
        "low_usd": 2 * personnel["E-5"],
        "high_usd": 4 * personnel["E-6"] + personnel["O-3"],
        "conditional": False,
        "assumption": "2 E-5 analyst-hours to 4 E-6 analyst-hours plus 1 O-3 review hour.",
        "rate_inputs": [
            {"resource": "E-5", "unit": "hour", "rate_usd": personnel["E-5"]},
            {"resource": "E-6", "unit": "hour", "rate_usd": personnel["E-6"]},
            {"resource": "O-3", "unit": "hour", "rate_usd": personnel["O-3"]},
        ],
        "basis": f"{_SOURCE['rate_schedule']} effective {_SOURCE['effective_date']}.",
    }


def _boat_option(track_id: str, severity: str) -> dict[str, Any]:
    boats = _RATES["boats_hourly_usd"]
    small_rate = boats["Response Boat - Small (II)"]
    medium_rate = boats["Response Boat - Medium"]
    return {
        "tier": "on_water_verification",
        "track_id": track_id,
        "severity": severity,
        "label": f"On-water verification, if dispatched{f' · {track_id}' if track_id else ''}",
        "low_usd": small_rate,
        "high_usd": 3 * medium_rate,
        "conditional": True,
        "assumption": "1 hour using a small response boat to 3 hours using a medium response boat; staffing and transit are not added.",
        "rate_inputs": [
            {"resource": "Response Boat - Small (II)", "unit": "hour", "rate_usd": small_rate},
            {"resource": "Response Boat - Medium", "unit": "hour", "rate_usd": medium_rate},
        ],
        "basis": f"{_SOURCE['rate_schedule']} effective {_SOURCE['effective_date']}.",
    }


def _air_option(track_id: str, severity: str) -> dict[str, Any]:
    aircraft = _RATES["aircraft_hourly_usd"]
    dolphin_rate = aircraft["MH-65 Dolphin"]
    jayhawk_rate = aircraft["MH-60 Jayhawk"]
    return {
        "tier": "air_support",
        "track_id": track_id,
        "severity": severity,
        "label": f"Aircraft support, if authorized{f' · {track_id}' if track_id else ''}",
        "low_usd": dolphin_rate,
        "high_usd": 3 * jayhawk_rate,
        "conditional": True,
        "assumption": "1 MH-65 flight-hour to 3 MH-60 flight-hours; mission-specific staffing and transit are not added.",
        "rate_inputs": [
            {"resource": "MH-65 Dolphin", "unit": "hour", "rate_usd": dolphin_rate},
            {"resource": "MH-60 Jayhawk", "unit": "hour", "rate_usd": jayhawk_rate},
        ],
        "basis": f"{_SOURCE['rate_schedule']} effective {_SOURCE['effective_date']}.",
    }


def response_plan(
    signals: list[str],
    *,
    severity: str = "warning",
    track_id: str = "",
    include_on_water: bool = False,
    include_air: bool = False,
) -> dict[str, Any]:
    """Build mutually exclusive response options from official unit rates."""
    unique_signals = list(dict.fromkeys(signal for signal in signals if signal))
    items: list[dict[str, Any]] = []
    if unique_signals:
        items.append(_review_option(track_id, severity))
        if include_on_water:
            items.append(_boat_option(track_id, severity))
        if include_air:
            items.append(_air_option(track_id, severity))

    return {
        "currency": "USD",
        "low_usd": min((item["low_usd"] for item in items), default=0),
        "high_usd": max((item["high_usd"] for item in items), default=0),
        "items": items,
        "signals": unique_signals,
        "range_semantics": "Each item is a separate response option; do not add the ranges.",
        "source": _SOURCE,
        "disclaimer": (
            "Conditional planning options, not incurred costs, damages, or a dispatch recommendation. "
            + _SOURCE["scope_note"]
        ),
    }


def estimate_response_cost(frame_delta: dict[str, Any]) -> dict[str, Any]:
    """Return sourced response options for safety signals in one frame."""
    contacts: dict[str, dict[str, Any]] = {}
    severity_rank = {"info": 0, "warning": 1, "critical": 2}

    for alert in frame_delta.get("alerts", ()):
        kind = str(alert.get("kind", ""))
        if kind not in _SIGNAL_LABELS:
            continue
        track_id = str(alert.get("track_id", ""))
        contact = contacts.setdefault(track_id, {
            "signals": [],
            "severity": "info",
            "include_on_water": False,
            "include_air": False,
        })
        contact["signals"].append(_SIGNAL_LABELS[kind])
        severity = str(alert.get("severity", "info"))
        if severity_rank.get(severity, 0) > severity_rank.get(contact["severity"], 0):
            contact["severity"] = severity
        if kind in {"intrusion", "went_dark"}:
            contact["include_on_water"] = True
        if kind == "intrusion" and severity == "critical":
            contact["include_air"] = True

    for track in frame_delta.get("tracks", ()):
        if not track.get("dark"):
            continue
        track_id = str(track.get("track_id", ""))
        contact = contacts.setdefault(track_id, {
            "signals": [],
            "severity": "warning",
            "include_on_water": True,
            "include_air": False,
        })
        if _SIGNAL_LABELS["dark_monitoring"] not in contact["signals"]:
            contact["signals"].append(_SIGNAL_LABELS["dark_monitoring"])
        contact["include_on_water"] = True

    items: list[dict[str, Any]] = []
    signals: list[str] = []
    for track_id, contact in contacts.items():
        plan = response_plan(
            contact["signals"],
            severity=contact["severity"],
            track_id=track_id,
            include_on_water=contact["include_on_water"],
            include_air=contact["include_air"],
        )
        items.extend(plan["items"])
        signals.extend(plan["signals"])

    result = response_plan([])
    result["items"] = items
    result["signals"] = list(dict.fromkeys(signals))
    result["low_usd"] = min((item["low_usd"] for item in items), default=0)
    result["high_usd"] = max((item["high_usd"] for item in items), default=0)
    return result

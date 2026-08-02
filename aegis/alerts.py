"""Alert policy, evidence relationships, and operator briefs."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .driver import fmt_t
from .graph import Alert, Brief, JustifiedBy, Mission, Track, Zone, mint_alert_id

LLM_MODEL_NAME = os.environ.get("AEGIS_LLM", "mockllm")

_MOCK_HEADLINES = (
    "Dark vessel intrusion: an untransmitting contact is loitering inside the sanctuary.",
    "Track lost AIS and is now coasting on prediction only.",
    "Radar contact re-attributed to a previously dark track.",
    "Vessel entered a restricted approach zone.",
    "Displayed identity changed mid-transit; possible spoofing.",
    "Contact re-acquired inside the protected area with high confidence.",
    "Two tracks held through a close crossing without identity swap.",
    "Watch brief: one contact dark in protected water, prediction confidence degrading.",
)


def severity_for(kind: str, zone_kind: str, dark: bool) -> str:
    if kind == "intrusion":
        if dark:
            return "critical"
        return "warning" if zone_kind in ("sanctuary", "restricted") else "info"
    if kind == "went_dark":
        return "warning"
    if kind == "reacquired":
        return "warning" if dark else "info"
    if kind == "identity_change":
        return "warning"
    return "info"


def _detail_line(
    kind: str, severity: str, track: Track, zone_name: str, t: float
) -> str:
    stamp = fmt_t(t)
    label = track.track_id
    if kind == "intrusion":
        state = "DARK, position predicted" if track.status == "coasting" else "tracked"
        return f"[{stamp}] {severity.upper()}: {label} ({state}) entered {zone_name}"
    if kind == "went_dark":
        return (
            f"[{stamp}] {severity.upper()}: {label} stopped transmitting; "
            "coasting on prediction, uncertainty growing"
        )
    if kind == "reacquired":
        return (
            f"[{stamp}] {severity.upper()}: unlabelled detection fused into "
            f"{label} after dark period"
        )
    if kind == "identity_change":
        return f"[{stamp}] {severity.upper()}: displayed identity of {label} changed"
    return f"[{stamp}] {severity.upper()}: {kind} on {label}"


def raise_alert(
    mission: Mission,
    track: Track,
    kind: str,
    zone: Zone | None,
    t: float,
    evidence_meas_ids: list[str],
) -> Alert:
    dark = track.status == "coasting" or track.was_dark
    severity = severity_for(kind, zone.kind if zone else "", dark)
    detail = _detail_line(kind, severity, track, zone.name if zone else "", t)
    alert = Alert(
        alert_id=mint_alert_id(mission),
        kind=kind,
        severity=severity,
        t=t,
        track_id=track.track_id,
        zone_id=zone.zone_id if zone else "",
        headline=detail,
        detail=detail,
        model_used="template",
        raised_on=track,
        concerning=zone,
    )
    wanted = set(evidence_meas_ids)
    alert.justified_by = [
        JustifiedBy(contact)
        for contact in mission.contacts
        if contact.meas_id in wanted
    ]
    mission.alerts.append(alert)
    return alert


def _litellm_headline(alert: Alert, zone_name: str) -> str:
    import litellm

    litellm.suppress_debug_info = True
    response = litellm.completion(
        model=LLM_MODEL_NAME,
        messages=[{
            "role": "user",
            "content": (
                "One-sentence maritime watch headline, under 20 words. "
                f"Alert kind: {alert.kind}; severity: {alert.severity}; "
                f"track: {alert.track_id}; zone: {zone_name or 'none'}; "
                f"time: {fmt_t(alert.t)}."
            ),
        }],
        max_tokens=60,
    )
    return response.choices[0].message.content.strip()


@dataclass
class AlertScribe:
    composed: int = 0

    def compose(self, alert: Alert) -> "AlertScribe":
        if LLM_MODEL_NAME == "mockllm":
            try:
                index = int(alert.alert_id.rsplit("-", 1)[1]) - 1
                alert.headline = _MOCK_HEADLINES[index]
            except (IndexError, ValueError):
                alert.headline = alert.detail
                alert.model_used = "template"
            else:
                alert.model_used = "mockllm"
        else:
            try:
                alert.headline = _litellm_headline(
                    alert, alert.concerning.name if alert.concerning else ""
                )
                alert.model_used = LLM_MODEL_NAME
            except Exception:
                alert.headline = alert.detail
                alert.model_used = "template"
        self.composed += 1
        return self


def _digest(lines: list[str], n_tracks: int, n_dark: int, clock: str) -> str:
    head = (
        f"Situation at {clock}: {n_tracks} tracks held, "
        f"{n_dark} coasting without AIS."
    )
    return head + (" Alerts: " + " | ".join(lines) if lines else " No alerts raised.")


@dataclass
class Briefer:
    brief: Brief | None = None

    def assemble(self, mission: Mission) -> "Briefer":
        alerts = sorted(mission.alerts, key=lambda alert: alert.t)
        live = [track for track in mission.tracks if track.status != "dead"]
        digest = _digest(
            [alert.detail for alert in alerts],
            len(live),
            sum(track.status == "coasting" for track in live),
            fmt_t(mission.t),
        )
        # Brief generation stays deterministic and offline in the Python runtime.
        model_used = "mockllm" if LLM_MODEL_NAME == "mockllm" else "template"
        self.brief = Brief(
            t=mission.t,
            text=digest,
            alert_count=len(alerts),
            model_used=model_used,
            summarizes=list(alerts),
        )
        mission.briefs.append(self.brief)
        return self

"""Render-ready frame delta construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from data.geometry import cov_ellipse_points, heading_speed
from data.scenario import enu_to_lla
from financial import estimate_response_cost

from .driver import fmt_t
from .graph import LIVE_STATUSES, STATUS_COASTING, Alert, Mission, Track

TRACK_COLORS = {
    "tentative": "#5b6673",
    "confirmed": "#2dd4bf",
    "coasting": "#f59e0b",
    "dead": "#374151",
}
ZONE_STYLES = {
    "sanctuary": {
        "color": "#34d399",
        "fillOpacity": 0.12,
        "dashArray": "6 4",
    },
    "restricted": {
        "color": "#f87171",
        "fillOpacity": 0.15,
        "dashArray": "2 4",
    },
    "zone": {"color": "#93c5fd", "fillOpacity": 0.08, "dashArray": ""},
}
SEVERITY_COLORS = {
    "info": "#93c5fd",
    "warning": "#f59e0b",
    "critical": "#ef4444",
}
MEAS_STYLE = {
    "ais": {"color": "#94a3b8", "radius": 3},
    "radar": {"color": "#e879f9", "radius": 6},
}


def track_color(status: str, in_zone: bool) -> str:
    if status == STATUS_COASTING and in_zone:
        return "#ef4444"
    return TRACK_COLORS.get(status, "#5b6673")


def ellipse_for(track: Track) -> list[list[float]]:
    return [
        [round(x, 1), round(y, 1)]
        for x, y in cov_ellipse_points(
            track.x,
            track.y,
            track.p00,
            track.p01,
            track.p11,
            k_sigma=2.0,
            n_points=24,
        )
    ]


def ellipse_latlon_for(
    track: Track, origin: tuple[float, float]
) -> list[list[float]]:
    return [
        [round(lat, 6), round(lon, 6)]
        for lat, lon in (
            enu_to_lla(x, y, origin[0], origin[1])
            for x, y in cov_ellipse_points(
                track.x,
                track.y,
                track.p00,
                track.p01,
                track.p11,
                k_sigma=2.0,
                n_points=24,
            )
        )
    ]


@dataclass
class Renderer:
    origin: tuple[float, float] = (0.0, 0.0)
    payloads: list[dict[str, Any]] = field(default_factory=list)

    def run(self, mission: Mission) -> "Renderer":
        for track in mission.tracks:
            if track.status not in LIVE_STATUSES:
                continue
            heading, speed = heading_speed(track.vx, track.vy)
            lat, lon = enu_to_lla(
                track.x, track.y, self.origin[0], self.origin[1]
            )
            in_zone = bool(track.zones_in)
            speed_kn = round(speed * 1.94384, 1)
            self.payloads.append({
                "track_id": track.track_id,
                "x": round(track.x, 1),
                "y": round(track.y, 1),
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "heading_deg": round(heading, 1),
                "speed_mps": round(speed, 2),
                "speed_kn": speed_kn,
                "status": track.status,
                "dark": track.status == STATUS_COASTING,
                "color": track_color(track.status, in_zone),
                "ellipse": ellipse_for(track),
                "ellipse_latlon": ellipse_latlon_for(track, self.origin),
                "zones": list(track.zones_in),
                "last_source": track.last_source,
                "label": f"{track.track_id} · {speed_kn} kn",
            })
        return self


def zone_payloads(mission: Mission) -> list[dict[str, Any]]:
    return [
        {
            "zone_id": zone.zone_id,
            "name": zone.name,
            "kind": zone.kind,
            "ring": [[round(p[0], 1), round(p[1], 1)] for p in zone.ring],
            "geojson": zone.geojson,
            "style": ZONE_STYLES.get(zone.kind, ZONE_STYLES["zone"]),
        }
        for zone in mission.zones
    ]


def alert_payload(alert: Alert) -> dict[str, Any]:
    return {
        "alert_id": alert.alert_id,
        "kind": alert.kind,
        "severity": alert.severity,
        "color": SEVERITY_COLORS.get(alert.severity, "#93c5fd"),
        "t": alert.t,
        "stamp": fmt_t(alert.t),
        "track_id": alert.track_id,
        "zone_id": alert.zone_id,
        "headline": alert.headline,
        "detail": alert.detail,
        "model_used": alert.model_used,
    }


def meas_payloads(measurements: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "meas_id": measurement.meas_id,
            "x": round(measurement.x, 1),
            "y": round(measurement.y, 1),
            "source": measurement.source,
            "style": MEAS_STYLE.get(measurement.source, MEAS_STYLE["ais"]),
        }
        for measurement in measurements
    ]


def frame_delta(
    mission: Mission,
    driver: Any,
    fusion: Any,
    renderer: Renderer,
    include_zones: bool,
) -> dict[str, Any]:
    delta = {
        "frame_idx": mission.frame_idx,
        "t": mission.t,
        "clock": fmt_t(mission.t),
        "tracks": renderer.payloads,
        "measurements": meas_payloads(driver.effective),
        "alerts": [alert_payload(alert) for alert in fusion.new_alerts],
        "log": driver.log + fusion.log,
        "removed_track_ids": fusion.dropped_ids,
        "zones": zone_payloads(mission) if include_zones else [],
        "stats": {
            "n_tracks": len(renderer.payloads),
            "n_dark": sum(payload["dark"] for payload in renderer.payloads),
            "n_meas": len(driver.effective),
            "n_suppressed": driver.suppressed_count,
            "n_matched": fusion.n_matched,
        },
    }
    delta["financial_risk"] = estimate_response_cost(delta)
    return delta

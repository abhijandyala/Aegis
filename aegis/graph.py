"""In-memory mission graph used by the pure-Python Aegis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from data.geometry import make_zone_polygon, polygon_contains

STATUS_TENTATIVE = "tentative"
STATUS_CONFIRMED = "confirmed"
STATUS_COASTING = "coasting"
STATUS_DEAD = "dead"
LIVE_STATUSES = (STATUS_TENTATIVE, STATUS_CONFIRMED, STATUS_COASTING)

DARK_MIN_HITS = 6
REACQ_MIN_COAST = 4


@dataclass
class Zone:
    zone_id: str = ""
    name: str = ""
    kind: str = "zone"
    ring: list[list[float]] = field(default_factory=list)
    geojson: dict[str, Any] = field(default_factory=dict)
    poly: Any = field(default=None, repr=False)

    def contains(self, x: float, y: float) -> bool:
        if self.poly is None:
            self.poly = make_zone_polygon(self.ring)
        return polygon_contains(self.poly, x, y)


@dataclass
class ActorState:
    actor: str = ""
    ais_suppressed: bool = False
    dark_since: float = -1.0
    display_id: str = ""


@dataclass
class InZone:
    zone: Zone
    since_t: float = 0.0
    active: bool = True


@dataclass
class Contact:
    meas_id: str = ""
    t: float = 0.0
    x: float = 0.0
    y: float = 0.0
    source: str = ""
    sigma: float = 0.0


@dataclass
class Track:
    track_id: str = ""
    _rt: Any = field(default=None, repr=False)
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    p00: float = 0.0
    p01: float = 0.0
    p11: float = 0.0
    status: str = STATUS_TENTATIVE
    hits: int = 0
    misses: int = 0
    llr: float = 0.0
    born_t: float = 0.0
    last_t: float = 0.0
    last_source: str = ""
    last_meas_id: str = ""
    last_fix_x: float = 0.0
    last_fix_y: float = 0.0
    zones_in: list[str] = field(default_factory=list)
    was_dark: bool = False
    in_zone_edges: list[InZone] = field(default_factory=list)

    @property
    def rt(self) -> Any:
        return self._rt

    def sync_from_tracker(self) -> None:
        state = self._rt.kf.state
        cov = self._rt.kf.cov
        self.x, self.y, self.vx, self.vy = map(float, state)
        self.p00 = float(cov[0][0])
        self.p01 = float(cov[0][1])
        self.p11 = float(cov[1][1])
        self.status = self._rt.status
        self.hits = self._rt.hits
        self.misses = self._rt.misses
        self.llr = float(self._rt.llr)


@dataclass
class JustifiedBy:
    contact: Contact
    weight: float = 1.0


@dataclass
class Alert:
    alert_id: str = ""
    kind: str = ""
    severity: str = "info"
    t: float = 0.0
    track_id: str = ""
    zone_id: str = ""
    headline: str = ""
    detail: str = ""
    model_used: str = ""
    raised_on: Track | None = field(default=None, repr=False)
    concerning: Zone | None = field(default=None, repr=False)
    justified_by: list[JustifiedBy] = field(default_factory=list, repr=False)

    @property
    def evidence(self) -> list[Contact]:
        return [edge.contact for edge in self.justified_by]


@dataclass
class Brief:
    t: float = 0.0
    text: str = ""
    alert_count: int = 0
    model_used: str = ""
    summarizes: list[Alert] = field(default_factory=list, repr=False)


@dataclass
class Mission:
    pack_id: str = ""
    name: str = ""
    frame_interval_s: float = 30.0
    frame_idx: int = -1
    t: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)
    zones: list[Zone] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    contacts: list[Contact] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    briefs: list[Brief] = field(default_factory=list)
    actor_states: list[ActorState] = field(default_factory=list)


def build_mission(scenario: Any) -> Mission:
    mission = Mission(
        pack_id=scenario.pack_id,
        name=scenario.name,
        frame_interval_s=scenario.frame_interval_s,
    )
    mission.zones = [
        Zone(
            zone_id=gf.fence_id,
            name=gf.name,
            kind=gf.kind,
            ring=[[p[0], p[1]] for p in gf.ring],
            geojson=gf.geojson,
        )
        for gf in scenario.geofences
    ]
    mission.meta.update(
        actor_states={},
        pending_births=[],
        next_track=1,
        next_alert=1,
    )
    return mission


def zones_of(mission: Mission) -> list[Zone]:
    return list(mission.zones)


def tracks_of(mission: Mission) -> list[Track]:
    return [track for track in mission.tracks if track.status in LIVE_STATUSES]


def alerts_of(mission: Mission) -> list[Alert]:
    return list(mission.alerts)


def actor_state(mission: Mission, actor: str) -> ActorState:
    cache = mission.meta.setdefault("actor_states", {})
    if actor not in cache:
        state = ActorState(actor=actor, display_id=actor)
        cache[actor] = state
        mission.actor_states.append(state)
    return cache[actor]


def mint_track_id(mission: Mission) -> str:
    number = mission.meta.setdefault("next_track", 1)
    mission.meta["next_track"] = number + 1
    return f"T-{number:03d}"


def mint_alert_id(mission: Mission) -> str:
    number = mission.meta.setdefault("next_alert", 1)
    mission.meta["next_alert"] = number + 1
    return f"A-{number:03d}"

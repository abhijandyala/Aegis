"""Stateful geofence and AIS transition analysis in ordinary Python."""

from __future__ import annotations

from dataclasses import dataclass, field

from tracker.geofence import GeofenceIndex, corridor_fence, polygon_fence

SEV_RANK = {"info": 0, "warning": 1, "critical": 2, "emergency": 3}
ENTER_SEV = {
    "mpa": "warning",
    "danger": "warning",
    "cable": "warning",
    "port": "info",
}
DARK_INSIDE_SEV = {
    "mpa": "emergency",
    "danger": "critical",
    "cable": "critical",
    "port": "critical",
}
EXIT_SEV = "info"
DARK_OUTSIDE_SEV = "warning"
NO_FENCE = ""

EVENTS: list[dict] = []
HUB: list["Watchtower"] = []
INDEX: list[GeofenceIndex] = []
FRAME_NO = [0]


@dataclass
class TrackWatch:
    track_id: str
    inside: list[str] = field(default_factory=list)
    dark: bool = False
    fired: list[str] = field(default_factory=list)
    frames_seen: int = 0


@dataclass
class Watchtower:
    label: str = "monitor"
    watches: list[TrackWatch] = field(default_factory=list, repr=False)


def severity_rank(sev: str) -> int:
    return SEV_RANK.get(sev, 0)


def enter_severity(kind: str) -> str:
    return ENTER_SEV.get(kind, "warning")


def dark_inside_severity(kind: str) -> str:
    return DARK_INSIDE_SEV.get(kind, "critical")


def fence_kind(fid: str) -> str:
    if not INDEX:
        return "mpa"
    fence = INDEX[0].get(fid)
    return "mpa" if fence is None else fence.kind


def fence_label(fid: str) -> str:
    if not INDEX:
        return fid
    fence = INDEX[0].get(fid)
    return fid if fence is None else fence.label


def emit(
    kind_of_event: str, actor: str, fid: str, sev: str, detail: str
) -> dict:
    event = {
        "type": kind_of_event,
        "actor": actor,
        "geofence_id": fid,
        "severity": sev,
        "severity_rank": severity_rank(sev),
        "detail": detail,
        "frame": FRAME_NO[0],
    }
    EVENTS.append(event)
    return event


def fmt(ev: dict) -> str:
    return (
        f"[geofence] f{ev['frame']:02d} {ev['severity'].upper():<9} "
        f"{ev['type']:<26} {ev['actor']:<6} {ev['detail']}"
    )


def hub() -> Watchtower:
    if not HUB:
        HUB.append(Watchtower())
    return HUB[0]


def watch_for(track_id: str) -> TrackWatch:
    current = hub()
    for watch in current.watches:
        if watch.track_id == track_id:
            return watch
    watch = TrackWatch(track_id=track_id)
    current.watches.append(watch)
    return watch


def _log(
    emitted: list[dict],
    kind_of_event: str,
    actor: str,
    fid: str,
    sev: str,
    detail: str,
) -> None:
    emitted.append(emit(kind_of_event, actor, fid, sev, detail))


def _check(watch: TrackWatch, observation: list, emitted: list[dict]) -> None:
    now_ids = sorted(observation[0])
    now_dark = observation[1]
    was_ids = watch.inside
    was_dark = watch.dark
    watch.frames_seen += 1

    for fid in was_ids:
        if fid not in now_ids:
            kind = fence_kind(fid)
            _log(
                emitted,
                "geofence_exit",
                watch.track_id,
                fid,
                EXIT_SEV,
                f"{watch.track_id} left {fence_label(fid)} ({kind})",
            )
            if fid in watch.fired:
                watch.fired.remove(fid)

    for fid in now_ids:
        if fid not in was_ids:
            kind = fence_kind(fid)
            _log(
                emitted,
                "geofence_enter",
                watch.track_id,
                fid,
                enter_severity(kind),
                f"{watch.track_id} entered {fence_label(fid)} ({kind})",
            )

    watch.inside = now_ids

    if now_dark and not was_dark and not now_ids:
        _log(
            emitted,
            "ais_dark",
            watch.track_id,
            NO_FENCE,
            DARK_OUTSIDE_SEV,
            f"{watch.track_id} stopped transmitting AIS in open water",
        )
    if was_dark and not now_dark:
        _log(
            emitted,
            "ais_resume",
            watch.track_id,
            NO_FENCE,
            "info",
            f"{watch.track_id} resumed AIS",
        )
        watch.fired = []
    watch.dark = now_dark

    if now_dark:
        for fid in now_ids:
            if fid not in watch.fired:
                watch.fired.append(fid)
                kind = fence_kind(fid)
                _log(
                    emitted,
                    "dark_inside_protected_area",
                    watch.track_id,
                    fid,
                    dark_inside_severity(kind),
                    f"{watch.track_id} went dark INSIDE "
                    f"{fence_label(fid)} ({kind})",
                )


def set_index(idx: GeofenceIndex) -> None:
    INDEX.clear()
    INDEX.append(idx)


def reset() -> None:
    EVENTS.clear()
    HUB.clear()
    FRAME_NO[0] = 0
    HUB.append(Watchtower())


def ingest_frame(
    track_ids: list, points: list, dark_ids: list = []
) -> list[dict]:
    FRAME_NO[0] += 1
    if INDEX and points:
        per_point = INDEX[0].query_many(points)
    else:
        per_point = [[] for _ in points]

    frame = {}
    for i, track_id in enumerate(track_ids):
        fence_ids = list(per_point[i]) if i < len(per_point) else []
        frame[track_id] = [fence_ids, track_id in dark_ids]
        watch_for(track_id)

    emitted: list[dict] = []
    for watch in hub().watches:
        if watch.track_id in frame:
            _check(watch, frame[watch.track_id], emitted)
    return emitted


def events() -> list[dict]:
    return list(EVENTS)


def event_count() -> int:
    return len(EVENTS)


def event_types() -> list[str]:
    return [event["type"] for event in EVENTS]


def events_of_type(t: str) -> list[dict]:
    return [event for event in EVENTS if event["type"] == t]


def events_for(actor: str) -> list[dict]:
    return [event for event in EVENTS if event["actor"] == actor]


def count_of_type(t: str) -> int:
    return len(events_of_type(t))


def inside_of(track_id: str) -> list[str]:
    for watch in hub().watches:
        if watch.track_id == track_id:
            return list(watch.inside)
    return []


def is_dark(track_id: str) -> bool:
    for watch in hub().watches:
        if watch.track_id == track_id:
            return watch.dark
    return False


def watch_count() -> int:
    return len(hub().watches)


def event_lines() -> list[str]:
    return [fmt(event) for event in EVENTS]

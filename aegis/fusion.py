"""Two-stage measurement fusion and track lifecycle orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from data.geometry import innovation_stats, reachable
from tracker.assoc import (
    AssocParams,
    Track as RuntimeTrack,
    associate_global,
    associate_greedy,
    max_assignable_sigma,
)
from tracker.kalman import CVKalman

from .alerts import raise_alert
from .driver import fmt_t
from .graph import (
    DARK_MIN_HITS,
    REACQ_MIN_COAST,
    STATUS_COASTING,
    STATUS_CONFIRMED,
    STATUS_DEAD,
    STATUS_TENTATIVE,
    InZone,
    Mission,
    Track,
    mint_track_id,
    tracks_of,
)

MISSION_PARAMS = AssocParams(delete_after_misses=60)
DARK_PARAMS = AssocParams(p_detect=0.05, delete_after_misses=60)
REACQ_PARAMS = AssocParams(
    p_detect=0.5, beta_fa=1e-11, delete_after_misses=60
)
REACQ_MAX_SPEED_MPS = 20.0
REACQ_MAX_GAP_S = 600.0
BIRTH_GATE_MPS = 15.0


def assoc_provenance(mode: str = "global") -> str:
    if mode == "global":
        stage1 = "tracker.assoc (global, padded Hungarian)"
    else:
        stage1 = "tracker.assoc (greedy nearest-neighbor -- naive baseline)"
    return (
        f"{stage1}; reacq ceiling "
        f"{max_assignable_sigma(REACQ_PARAMS) / 1000.0:.0f} km, "
        f"reach gate {REACQ_MAX_SPEED_MPS:.0f} m/s"
    )


@dataclass
class Fusion:
    t: float = 0.0
    dt: float = 30.0
    meas: list[Any] = field(default_factory=list)
    assoc_mode: str = "global"
    new_alerts: list[Any] = field(default_factory=list)
    born: list[str] = field(default_factory=list)
    dropped_ids: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    n_matched: int = 0
    n_reacquired: int = 0
    assigned: dict[str, tuple[Any, float, float, AssocParams]] = field(
        default_factory=dict
    )
    mission: Mission | None = None

    def run(self, mission: Mission) -> "Fusion":
        if self.assoc_mode not in ("global", "greedy"):
            raise ValueError(f"unknown association mode {self.assoc_mode!r}")
        self.mission = mission
        live = tracks_of(mission)
        for track in live:
            step = self.t - track._rt.kf.t
            if step > 0:
                track._rt.kf.predict(step)

        primary = [track for track in live if track.status != STATUS_COASTING]
        leftovers = self.solve(primary, self.meas, MISSION_PARAMS)
        dark = [track for track in live if track.status == STATUS_COASTING]
        leftovers = self.solve_reacquire(dark, leftovers)
        self.birth_tracks(mission, leftovers)

        for track in list(mission.tracks):
            if track.status == STATUS_DEAD:
                continue
            if track.born_t == self.t and track.track_id not in self.assigned:
                track.sync_from_tracker()
                continue
            if track.track_id in self.assigned:
                self.fuse_measurement(track, self.assigned[track.track_id])
            else:
                self.miss(track)
            track.sync_from_tracker()
            if track.status == STATUS_DEAD:
                self.dropped_ids.append(track.track_id)
                self.log.append(
                    f"[{fmt_t(self.t)}] TRACK {track.track_id} deleted "
                    f"(misses={track.misses}, score={track.llr:.1f})"
                )
            else:
                self.check_zones(track)
        return self

    def solve(
        self, nodes: list[Track], measurements: list[Any], params: AssocParams
    ) -> list[Any]:
        if not nodes or not measurements:
            return list(measurements)
        runtime_tracks = [node._rt for node in nodes]
        zs = [[measurement.x, measurement.y] for measurement in measurements]
        covariances = [measurement.R() for measurement in measurements]
        if self.assoc_mode == "greedy" and params is MISSION_PARAMS:
            result = associate_greedy(
                runtime_tracks, zs, covariances, params=params
            )
        else:
            result = associate_global(
                runtime_tracks, zs, covariances, params=params
            )
        for track_index, measurement_index in result.assignments:
            track = nodes[track_index]
            measurement = measurements[measurement_index]
            d2, logdet = innovation_stats(
                track._rt.kf,
                [measurement.x, measurement.y],
                measurement.R(),
            )
            self.assigned[track.track_id] = (
                measurement,
                d2,
                logdet,
                params,
            )
            self.n_matched += 1
        return [measurements[index] for index in result.unassigned_meas]

    def solve_reacquire(
        self, dark: list[Track], measurements: list[Any]
    ) -> list[Any]:
        if not dark or not measurements:
            return list(measurements)
        fresh = [
            track
            for track in dark
            if self.t - track.last_t <= REACQ_MAX_GAP_S
        ]
        if not fresh:
            return list(measurements)
        candidates = [
            measurement
            for measurement in measurements
            if any(
                reachable(
                    track.last_fix_x,
                    track.last_fix_y,
                    measurement.x,
                    measurement.y,
                    self.t - track.last_t,
                    REACQ_MAX_SPEED_MPS,
                )
                for track in fresh
            )
        ]
        if not candidates:
            return list(measurements)
        unclaimed_ids = {
            measurement.meas_id
            for measurement in self.solve(fresh, candidates, REACQ_PARAMS)
        }
        candidate_ids = {measurement.meas_id for measurement in candidates}
        return [
            measurement
            for measurement in measurements
            if measurement.meas_id not in candidate_ids
            or measurement.meas_id in unclaimed_ids
        ]

    def fuse_measurement(
        self,
        track: Track,
        assignment: tuple[Any, float, float, AssocParams],
    ) -> None:
        measurement, d2, logdet, params = assignment
        was_coasting = track.status == STATUS_COASTING
        was_tentative = track.status == STATUS_TENTATIVE
        dark_frames = track.misses
        track._rt.register_hit(d2, logdet, params=params)
        track._rt.kf.update(measurement.z(), measurement.R())
        track.last_t = self.t
        track.last_source = measurement.source
        track.last_meas_id = measurement.meas_id
        track.last_fix_x = measurement.x
        track.last_fix_y = measurement.y
        if was_coasting:
            self.n_reacquired += 1
            if dark_frames >= REACQ_MIN_COAST:
                alert = raise_alert(
                    self.mission,
                    track,
                    "reacquired",
                    None,
                    self.t,
                    [measurement.meas_id],
                )
                self.new_alerts.append(alert)
                self.log.append(
                    f"[{fmt_t(self.t)}] TRACK {track.track_id} reacquired via "
                    f"{measurement.source} detection after {dark_frames} frames "
                    f"dark (d2={d2:.1f})"
                )
        elif was_tentative and track._rt.status == STATUS_CONFIRMED:
            self.log.append(
                f"[{fmt_t(self.t)}] TRACK {track.track_id} confirmed "
                f"({track._rt.hits} fixes, score {track._rt.llr:.1f})"
            )

    def miss(self, track: Track) -> None:
        was_live = track.status != STATUS_COASTING
        track._rt.register_miss(
            params=MISSION_PARAMS if was_live else DARK_PARAMS
        )
        if was_live and track._rt.status == STATUS_COASTING:
            track.was_dark = True
            if track._rt.hits >= DARK_MIN_HITS:
                alert = raise_alert(
                    self.mission, track, "went_dark", None, self.t, []
                )
                self.new_alerts.append(alert)
                self.log.append(
                    f"[{fmt_t(self.t)}] TRACK {track.track_id} went dark; "
                    "coasting on prediction"
                )

    def check_zones(self, track: Track) -> None:
        for zone in self.mission.zones:
            inside = zone.contains(track.x, track.y)
            was_inside = zone.zone_id in track.zones_in
            if inside and not was_inside:
                track.zones_in.append(zone.zone_id)
                track.in_zone_edges.append(InZone(zone=zone, since_t=self.t))
                evidence = [track.last_meas_id] if track.last_meas_id else []
                alert = raise_alert(
                    self.mission,
                    track,
                    "intrusion",
                    zone,
                    self.t,
                    evidence,
                )
                self.new_alerts.append(alert)
                self.log.append(
                    f"[{fmt_t(self.t)}] ZONE {track.track_id} entered "
                    f"{zone.zone_id} ({track.status})"
                )
            elif was_inside and not inside:
                track.zones_in.remove(zone.zone_id)
                for edge in track.in_zone_edges:
                    if edge.zone is zone:
                        edge.active = False
                self.log.append(
                    f"[{fmt_t(self.t)}] ZONE {track.track_id} exited {zone.zone_id}"
                )

    def birth_tracks(self, mission: Mission, unmatched: list[Any]) -> None:
        pending = mission.meta["pending_births"]
        gate = BIRTH_GATE_MPS * self.dt + 100.0
        still_pending: list[Any] = []
        used: set[str] = set()
        for previous in pending:
            best = None
            best_distance = gate
            for current in unmatched:
                if current.meas_id in used:
                    continue
                distance = (
                    (current.x - previous.x) ** 2
                    + (current.y - previous.y) ** 2
                ) ** 0.5
                if distance < best_distance:
                    best = current
                    best_distance = distance
            if best is None:
                continue
            used.add(best.meas_id)
            kf = CVKalman.init_two_point(
                previous.z(), previous.t, best.z(), best.t, best.R()
            )
            track_id = mint_track_id(mission)
            runtime = RuntimeTrack(
                id=mission.meta["next_track"], kf=kf, params=MISSION_PARAMS
            )
            runtime.hits = 2
            runtime.hit_history.extend((True, True))
            track = Track(
                track_id=track_id,
                _rt=runtime,
                born_t=self.t,
                last_t=self.t,
                last_source=best.source,
                last_meas_id=best.meas_id,
                last_fix_x=best.x,
                last_fix_y=best.y,
            )
            track.sync_from_tracker()
            mission.tracks.append(track)
            self.born.append(track_id)
        still_pending.extend(
            current for current in unmatched if current.meas_id not in used
        )
        mission.meta["pending_births"] = still_pending

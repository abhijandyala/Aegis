"""Scenario event dispatch for the pure-Python pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from data.contracts import Measurement, RADAR_SIGMA_M
from data.scenario import true_position_at

from .graph import Contact, Mission, actor_state


def fmt_t(t: float) -> str:
    seconds = int(t)
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


@dataclass
class ScenarioDriver:
    scenario: Any
    frame: Any
    effective: list[Measurement] = field(default_factory=list)
    injected: list[Measurement] = field(default_factory=list)
    suppressed_count: int = 0
    log: list[str] = field(default_factory=list)

    def run(self, mission: Mission) -> "ScenarioDriver":
        mission.frame_idx = self.frame.idx
        mission.t = self.frame.t
        for event in self.frame.events:
            self.apply_event(mission, event)
        for measurement in self.frame.measurements:
            actor = self.scenario.ground_truth[measurement.meas_id]
            if actor_state(mission, actor).ais_suppressed:
                self.suppressed_count += 1
            else:
                self.effective.append(measurement)
        self.effective.extend(self.injected)
        return self

    def apply_event(self, mission: Mission, event: Any) -> None:
        state = actor_state(mission, event.actor)
        if event.kind == "ais_off":
            state.ais_suppressed = True
            state.dark_since = event.t
            self.log.append(
                f"[{fmt_t(event.t)}] EVENT ais_off: a vessel stopped transmitting"
            )
        elif event.kind == "ais_on":
            state.ais_suppressed = False
            state.dark_since = -1.0
            self.log.append(
                f"[{fmt_t(event.t)}] EVENT ais_on: transmission resumed"
            )
        elif event.kind == "radar_contact":
            self.inject_radar(mission, event)
        elif event.kind == "identity_change":
            old = state.display_id
            state.display_id = str(event.params["new_display_id"])
            self.log.append(
                f"[{fmt_t(event.t)}] EVENT identity_change: display identity "
                f"{old} -> {state.display_id}"
            )
        else:
            raise ValueError(f"unknown scenario event kind {event.kind}")

    def inject_radar(self, mission: Mission, event: Any) -> None:
        if "x" in event.params and "y" in event.params:
            px = float(event.params["x"])
            py = float(event.params["y"])
        else:
            position = true_position_at(
                self.scenario.true_tracks[event.actor], event.t
            )
            if position is None:
                raise ValueError(
                    f"radar_contact {event.event_id}: {event.actor} has no true "
                    f"position at t={event.t}"
                )
            px, py = position
        sigma = float(event.params.get("sigma", RADAR_SIGMA_M))
        noise = mission.meta["rng"].normal(0.0, sigma, 2)
        meas_id = self.scenario.mint_meas_id(event.actor)
        measurement = Measurement(
            meas_id=meas_id,
            t=self.frame.t,
            x=px + float(noise[0]),
            y=py + float(noise[1]),
            source="radar",
            sigma=sigma,
        )
        mission.contacts.append(
            Contact(
                meas_id=meas_id,
                t=self.frame.t,
                x=measurement.x,
                y=measurement.y,
                source="radar",
                sigma=sigma,
            )
        )
        self.injected.append(measurement)
        self.log.append(
            f"[{fmt_t(self.frame.t)}] EVENT radar_contact: unlabelled detection "
            f"at ({measurement.x:.0f} E, {measurement.y:.0f} N) sigma {sigma:.0f} m"
        )

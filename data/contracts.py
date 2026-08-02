"""Frozen interface contract between the data layer, the tracker, and the frontend.

Ownership
---------
- ``Measurement``, ``Frame``, ``Geofence``, ``ScenarioEvent`` are produced by the
  data layer (this package) and consumed by the tracker and Aegis pipeline.
- ``TrackState`` is produced by the tracker layer.
- ``FrameDelta`` is the *only* thing the frontend ever receives. The frontend is
  dumb: it renders FrameDeltas and computes nothing.

Identity rule
-------------
``Measurement`` carries **no identity of any kind** -- no MMSI, no vessel name,
no callsign. Truth lives exclusively in ``Scenario.ground_truth[meas_id]`` and
is used only for scoring and for the demo reveal. Recovering identity from
kinematics alone is the project; leaking it here would make the demo a lie.

All coordinates are local ENU metres relative to the scenario origin. No lat/lon
crosses this boundary except inside ``Geofence.geojson`` (kept verbatim so the
frontend can draw layers on a lat/lon map without doing math).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "AIS_SIGMA_M",
    "RADAR_SIGMA_M",
    "Measurement",
    "Frame",
    "Geofence",
    "ScenarioEvent",
    "TrackState",
    "FrameDelta",
]

# Per-sensor 1-sigma position noise, metres. The tracker receives these as a
# per-call R = diag(sigma^2, sigma^2); it never caches them.
AIS_SIGMA_M: float = 5.0
RADAR_SIGMA_M: float = 50.0


@dataclass(frozen=True, slots=True)
class Measurement:
    """One unlabelled position detection. This is all the tracker ever sees.

    Deliberately has no identity field of any kind. ``source`` distinguishes
    sensor noise characteristics only, never identity.
    """

    meas_id: str          # unique within a scenario, e.g. "m000123"
    t: float              # seconds since scenario start
    x: float              # ENU east, metres
    y: float              # ENU north, metres
    source: str           # "ais" | "radar"
    sigma: float          # 1-sigma position noise, metres (feeds R = diag(sigma^2))

    def z(self) -> Tuple[float, float]:
        """Position vector as the tracker's update() expects it."""
        return (self.x, self.y)

    def R(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Per-call measurement noise covariance, metres^2."""
        v = self.sigma * self.sigma
        return ((v, 0.0), (0.0, v))


@dataclass(frozen=True, slots=True)
class Frame:
    """All measurements landing in one fixed replay tick."""

    idx: int                       # 0-based frame index
    t: float                       # seconds since scenario start (idx * interval)
    measurements: Tuple[Measurement, ...]
    events: Tuple["ScenarioEvent", ...] = ()   # scripted events firing this tick


@dataclass(frozen=True, slots=True)
class Geofence:
    """One protected/restricted polygon, in ENU metres at the scenario origin.

    ``ring`` is the exterior ring as (x, y) tuples (closed not required).
    ``geojson`` is the original lat/lon feature, verbatim, for the map layer.
    """

    fence_id: str
    name: str
    kind: str                      # "sanctuary" | "restricted" | "anchorage" | ...
    ring: Tuple[Tuple[float, float], ...]
    geojson: Dict[str, Any]

    def shapely(self):
        """Shapely polygon in ENU metres (built on demand; not cached on the
        frozen dataclass so instances stay hashable and picklable)."""
        from shapely.geometry import Polygon
        return Polygon(self.ring)

    def contains(self, x: float, y: float) -> bool:
        from shapely.geometry import Point
        return self.shapely().contains(Point(x, y))


@dataclass(frozen=True, slots=True)
class ScenarioEvent:
    """A scripted event applied by the replay layer at its timestamp.

    kinds:
      - "ais_off":        suppress the actor's AIS measurements from t onward
      - "ais_on":         resume the actor's AIS measurements
      - "radar_contact":  inject one unlabelled noisy radar detection at the
                          actor's true position (or params["x"], params["y"])
      - "identity_change": swap the actor's *display* MMSI (ground-truth map
                          only; measurements are untouched because they carry
                          no identity to begin with)
    """

    event_id: str
    t: float                       # seconds since scenario start
    kind: str
    actor: str                     # actor key (synthetic id or "mmsi:123456789")
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrackState:
    """Tracker output for one hypothesised vessel at one frame.

    Produced by the tracker layer (Raj); defined here so both sides compile
    against the same shape.
    """

    track_id: str
    t: float
    x: float
    y: float
    vx: float
    vy: float
    # Position covariance [[pxx, pxy], [pyx, pyy]] for the uncertainty ellipse.
    p_pos: Tuple[Tuple[float, float], Tuple[float, float]]
    # Lifecycle vocabulary is tracker.assoc.Track's, projected verbatim.
    status: str = "confirmed"      # "tentative" | "confirmed" | "coasting" | "dead"
    display_id: Optional[str] = None   # identity claim for the UI, never for association
    score: float = 0.0             # hypothesis log-likelihood / confidence


@dataclass(frozen=True, slots=True)
class FrameDelta:
    """The single message type streamed to the frontend per frame.

    The frontend renders this verbatim. Anything it would need to compute
    belongs in the backend and therefore in this object.
    """

    frame_idx: int
    t: float
    tracks: Tuple[TrackState, ...] = ()
    measurements: Tuple[Measurement, ...] = ()
    alerts: Tuple[Dict[str, Any], ...] = ()    # e.g. {"kind": "intrusion", "track_id", "fence_id"}
    events: Tuple[ScenarioEvent, ...] = ()
    removed_track_ids: Tuple[str, ...] = ()

"""Identity-switch scoring for the Aegis maritime tracker.

The tracker itself never sees vessel identity (MMSI). Identity is held out and
used only here, on the scoring side, so that the numbers this module produces
are an honest, out-of-band measure of association quality. The same functions
are applied unchanged to every algorithm we compare (global/Hungarian
association vs. greedy nearest-neighbour), on the exact same detections, so the
headline demo number is a like-for-like comparison.

The definition of an ID switch (CLEAR-MOT), verbatim
-----------------------------------------------------
  For each ground-truth object, remember the track_id it was MOST RECENTLY
  matched to. An ID switch is counted when that ground-truth object is matched
  to a DIFFERENT track_id than its last recorded one.

Why this is the right metric for this project
---------------------------------------------
The demo claims exactly one thing: that identity survives (a) vessels crossing
one another and (b) vessels going dark and being re-acquired. An ID switch is
the direct count of the times that claim fails. Position error says nothing
about identity; a tracker can hug every trajectory and still swap the labels of
two vessels at a crossing. Counting switches measures the failure we care about
and nothing else.

Consequences of the definition, all implemented here
-----------------------------------------------------
* The FIRST time a truth_id is ever seen is NOT a switch -- there is nothing to
  switch from. It only seeds the remembered track_id.
* Switches are counted ACROSS GAPS. If truth "A" matches track 1, goes
  unmatched for 30 frames, then matches track 2, that IS one switch. The memory
  of the last track_id is never cleared by a gap. This is deliberate: a vessel
  going dark and being re-acquired under a fresh track id is precisely the
  failure mode we are measuring, and forgiving it would flatter the tracker.
* Re-matching the SAME track_id after a gap is NOT a switch.
* Switching back counts again: A->1, A->2, A->1 is TWO switches.
* The mapping is keyed by TRUTH id, never by track id. Two different truths
  matched to the same track_id within one frame is a *merge*, a different
  pathology; it is reported separately as ``merges`` and never inflates
  ``id_switches``.

Determinism
-----------
The count depends only on the order of frames and, within a frame, on the order
of the pairs list as given. No set or dict iteration order ever influences a
count (sets are used only for cardinality). Repeated runs on the same input
give the same answer.

Pure stdlib. numpy is not needed here and is therefore not imported.
"""

from __future__ import annotations

from typing import Any, Hashable

__all__ = ["count_id_switches", "TrackingMetrics"]

Pair = tuple[int, str]


class TrackingMetrics:
    """Accumulates CLEAR-MOT identity statistics over a sequence of frames.

    Feed one frame at a time with :meth:`update`. The definition applied is:

      For each ground-truth object, remember the track_id it was MOST RECENTLY
      matched to. An ID switch is counted when that ground-truth object is
      matched to a DIFFERENT track_id than its last recorded one.

    The remembered mapping is keyed by truth id and is never cleared by a gap,
    so a vessel that goes dark and returns under a new track id is scored as a
    switch. Only :meth:`reset` clears it.
    """

    def __init__(self) -> None:
        self._last_track: dict[Hashable, int] = {}
        self._id_switches: int = 0
        self._merges: int = 0
        self._frames: int = 0
        self._total_matches: int = 0
        self._truths_seen: set[Hashable] = set()
        self._tracks_seen: set[int] = set()

    def update(self, pairs: list[Pair]) -> None:
        """Record one frame's matched ``(track_id, truth_id)`` pairs.

        A frame with no matches is legal and still counts as a frame.

        Pairs are processed sequentially in the order given. If a truth_id
        appears twice in the SAME frame (which a correct associator will never
        emit, but a buggy one might) the occurrences are simply treated as
        consecutive observations: ``[(1, "A"), (2, "A")]`` in one frame records
        one switch and leaves "A" remembered as track 2. This is deterministic
        because it depends only on list order, never on hash order.
        """
        self._frames += 1

        truths_per_track: dict[int, set[Hashable]] = {}

        for track_id, truth_id in pairs:
            self._total_matches += 1
            self._truths_seen.add(truth_id)
            self._tracks_seen.add(track_id)

            previous = self._last_track.get(truth_id)
            if previous is not None and previous != track_id:
                self._id_switches += 1
            self._last_track[truth_id] = track_id

            truths_per_track.setdefault(track_id, set()).add(truth_id)

        # A merge is a track_id carrying more than one distinct truth in the
        # same frame. Counted separately; it never touches _id_switches.
        for truths in truths_per_track.values():
            if len(truths) > 1:
                self._merges += len(truths) - 1

    @property
    def id_switches(self) -> int:
        """Total CLEAR-MOT identity switches observed so far."""
        return self._id_switches

    @property
    def merges(self) -> int:
        """Extra truths sharing one track_id within a frame, summed per frame."""
        return self._merges

    @property
    def frames(self) -> int:
        """Number of :meth:`update` calls, including empty ones."""
        return self._frames

    @property
    def total_matches(self) -> int:
        """Total number of pairs recorded across all frames."""
        return self._total_matches

    @property
    def truths_seen(self) -> int:
        """Number of distinct truth_ids ever matched."""
        return len(self._truths_seen)

    @property
    def tracks_seen(self) -> int:
        """Number of distinct track_ids ever matched."""
        return len(self._tracks_seen)

    def as_dict(self) -> dict[str, Any]:
        """All counters as a flat, JSON-serialisable dict of ints, for the UI."""
        return {
            "id_switches": self.id_switches,
            "merges": self.merges,
            "frames": self.frames,
            "total_matches": self.total_matches,
            "truths_seen": self.truths_seen,
            "tracks_seen": self.tracks_seen,
        }

    def reset(self) -> None:
        """Clear every counter and the remembered truth -> track mapping."""
        self._last_track.clear()
        self._id_switches = 0
        self._merges = 0
        self._frames = 0
        self._total_matches = 0
        self._truths_seen.clear()
        self._tracks_seen.clear()

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience only
        return f"TrackingMetrics({self.as_dict()})"


def count_id_switches(frames: list[list[Pair]]) -> int:
    """Count identity switches over a whole sequence.

    ``frames[k]`` is the list of ``(track_id, truth_id)`` pairs matched at
    frame ``k``.

    The definition applied, verbatim:

      For each ground-truth object, remember the track_id it was MOST RECENTLY
      matched to. An ID switch is counted when that ground-truth object is
      matched to a DIFFERENT track_id than its last recorded one.

    The first sighting of a truth id is not a switch; switches are counted
    across gaps; re-matching the same track after a gap is not a switch.

    This is implemented directly on top of :class:`TrackingMetrics`, so the two
    can never disagree.
    """
    metrics = TrackingMetrics()
    for frame in frames:
        metrics.update(frame)
    return metrics.id_switches

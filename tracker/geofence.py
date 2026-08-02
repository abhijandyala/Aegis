"""Geofence containment geometry for maritime intrusion detection.

Scope
-----
Phase 3.5 asks a single question: *which named areas is this point inside,
right now?* Areas are marine protected areas, buffered submarine-cable
corridors, port approaches and danger zones. This module answers exactly that
and nothing more -- it is stateless, pure geometry.

**Transition tracking is deliberately not here.** "Was this track inside last
frame?", "emit an INTRUSION event", "a dark vessel inside an MPA is severity
CRITICAL" -- all of that is per-track state and lives in the Aegis orchestration
layer where the rest of the track lifecycle already lives. Putting a
`previous_inside` flag in this file would fork the tracker's state across two
modules. If you are about to add a member that remembers a track, stop: it
belongs in the orchestration layer.

Coordinates
-----------
Every coordinate crossing this API is **local ENU metres**, never lon/lat
degrees. The scenario loader projects GeoJSON to ENU once at load time, so
buffers are plain metres, containment is Euclidean, and no code downstream has
to reason about degrees shrinking with latitude. The single exception is
:func:`lonlat_to_enu`, which *is* that converter.

Why an in-memory STRtree and not PostGIS
----------------------------------------
A spatial database earns its keep when you query a small slice of a large
corpus. Here the opposite holds: there are a couple of dozen fences, they are
static for the whole replay, and every frame touches *all* live tracks anyway.
There is no selective query to optimise and no persistence requirement -- only
raw per-frame throughput, at 60x replay speed. A `shapely.STRtree` built once in
:meth:`GeofenceIndex.__init__` and reused gives that with zero I/O, and its
vectorized bulk `query` resolves a whole frame of points in one call into C
rather than a Python loop per track.

Why cable corridors are pre-buffered
------------------------------------
A submarine cable is a LineString; the protected corridor is that line dilated
by a few hundred metres. `LineString.buffer()` is expensive -- it is a real
offsetting operation, orders of magnitude dearer than a containment test. So
:func:`corridor_fence` buffers **once**, at construction, and stores a polygon.
The hot path never buffers, never sees a line, and treats every fence
identically.

CONTRACT DIVERGENCE (read before merging)
-----------------------------------------
Three implementations of "a geofence" exist. This module keeps its own shape and
*adapts* the other two at the boundary; nothing downstream has to care.

===============  ==================  ==========================  ===================
this module      build-plan dict     Abhi (data/contracts.py)    resolved by
===============  ==================  ==========================  ===================
``id``           ``id``              ``fence_id``                ``geofence_from_mapping``
``label``        ``label``           ``name``                    ``geofence_from_mapping``
``kind``         ``type``            ``kind``                    ``normalize_kind``
``geom``         ``polygon``         ``ring`` (+ ``geojson``)     ``geofence_from_mapping``
``buffer_m``     ``buffer_m``        *absent*                    ``contracts.default_buffer_m``
===============  ==================  ==========================  ===================

Field-name note: ``kind`` stays ``kind`` here. It already carries the build
plan's *vocabulary* (mpa/cable/port/danger/land), ``type`` shadows a builtin,
and Abhi chose ``kind`` independently -- two of three spellings agree.

Kind vocabulary AND field-name reconciliation now live in
:mod:`tracker.contracts` -- :data:`CANONICAL_KINDS`, :data:`KIND_ALIASES` and
:func:`normalize_kind` in this module are thin re-exports/wrappers over it, not
a second copy. Abhi's values ("sanctuary", "restricted", "anchorage", ...) are
a different vocabulary for the same concepts; see
:func:`tracker.contracts.normalize_kind` for the unknown-input policy (it
raises -- it never guesses).

**buffer_m is absent from Abhi's dataclass, so** :func:`geofence_from_mapping`
**defaults it to** :func:`tracker.contracts.default_buffer_m` **for the
resolved kind** (500 m for cable, 0.0 otherwise) rather than a hardcoded 0.0.
A hardcoded zero is *wrong for a cable corridor*: a zero-buffered route has no
area and can never contain anything, so a cable layer imported through Abhi's
shape with no buffer field would silently never fire. Pass ``buffer_m=``
explicitly to override the default in either direction. A raw, already-built
shapely LineString handed to :func:`geofence_from_mapping` is the one case
where the default is deliberately NOT applied -- an explicit buffer is
required there too, and :func:`load_geojson_fences` likewise refuses to guess
a buffer for a line geometry (see both functions' docstrings).

Omar's scenario packs (``origin/omar/scenario-packs``) are a *fourth* shape, and
not a Geofence at all. ``pack.json`` keys the layer list as ``geofence_layers``
(build plan says ``layers``), and each entry is a file reference, verified as::

    {"file": "layers/monterey_bay_nms.geojson", "kind": "sanctuary"}

No id, no label, no geometry, no ``buffer_m`` -- so :func:`geofence_from_mapping`
cannot and must not consume it. The correct consumer is::

    load_geojson_fences(pack_dir / entry["file"], origin, kind=entry["kind"])

Two consequences for the pack loader (which is not this module): the pack's
``origin`` is ``{"lat": ..., "lon": ...}``, not the ``[lon, lat]`` sequence this
module's converters take, and a pack layer entry has **nowhere to declare a
buffer**, so a cable layer referenced from a pack cannot be loaded at all until
the entry grows a ``buffer_m`` (this module raises rather than load a
zero-buffered line). Omar's event objects also use ``kind`` where the build plan
says ``type``; that is the event contract, not this one.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from tracker import contracts

__all__ = [
    "Geofence",
    "GeofenceIndex",
    "GeofenceContractError",
    "polygon_fence",
    "corridor_fence",
    "lonlat_to_enu",
    "normalize_kind",
    "geofence_from_ring",
    "geofence_from_mapping",
    "load_geojson_fences",
    "build_index",
    "CANONICAL_KINDS",
    "KIND_ALIASES",
    "EARTH_RADIUS_M",
]

# WGS84 semi-major axis. Used as a sphere radius by the equirectangular
# projection below -- the flattening correction is far smaller than the
# projection's own distortion over a scenario box, so it would be false rigour.
EARTH_RADIUS_M: float = 6378137.0


@dataclass(frozen=True)
class Geofence:
    """One named area, ready to test against.

    ``geom`` is a shapely Polygon/MultiPolygon in ENU metres that is **already
    buffered** -- ``buffer_m`` is retained for provenance and reporting (an
    operator wants to see "500 m cable corridor"), never re-applied at query
    time. Frozen so a fence cannot be mutated out from under the STRtree, whose
    bounding boxes are snapshotted at build time and would silently go stale.
    """

    id: str
    kind: str  # "mpa" | "cable" | "port" | "danger" | "land"
    label: str
    geom: object  # shapely Polygon/MultiPolygon, ENU metres, already buffered
    buffer_m: float = 0.0


def polygon_fence(
    id: str,
    kind: str,
    label: str,
    coords: Sequence[Sequence[float]],
    buffer_m: float = 0.0,
) -> Geofence:
    """Build a fence from a polygon ring.

    ``coords`` is ``[[x, y], ...]`` in ENU metres. ``buffer_m > 0`` dilates the
    polygon outward, which is how an MPA gets an advisory margin.
    """
    geom = Polygon([(float(x), float(y)) for x, y in coords])
    if buffer_m > 0.0:
        # Only when asked: buffer(0) is not a no-op -- it re-noders the ring and
        # can quietly repair or collapse geometry we were handed deliberately.
        geom = geom.buffer(float(buffer_m))
    return Geofence(id=id, kind=kind, label=label, geom=geom, buffer_m=float(buffer_m))


def corridor_fence(
    id: str,
    label: str,
    coords: Sequence[Sequence[float]],
    buffer_m: float,
    kind: str = "cable",
) -> Geofence:
    """Build a corridor fence: a route polyline dilated into a polygon.

    ``coords`` is the cable route ``[[x, y], ...]`` in ENU metres. Buffering
    happens here, once, so the hot path only ever tests point-in-polygon.
    """
    geom = LineString([(float(x), float(y)) for x, y in coords]).buffer(float(buffer_m))
    return Geofence(id=id, kind=kind, label=label, geom=geom, buffer_m=float(buffer_m))


class GeofenceIndex:
    """Immutable spatial index over a fence set; the only stateful thing here
    is the tree itself, which is a pure function of the fences."""

    __slots__ = ("_fences", "_ids", "_by_id", "_tree")

    def __init__(self, fences: Sequence[Geofence]) -> None:
        self._fences: tuple[Geofence, ...] = tuple(fences)
        # Index-aligned with the tree: STRtree returns positional indices, so
        # id lookup is a tuple index rather than a dict hash on the hot path.
        self._ids: tuple[str, ...] = tuple(f.id for f in self._fences)
        self._by_id: dict[str, Geofence] = {f.id: f for f in self._fences}
        self._tree = shapely.STRtree([f.geom for f in self._fences])

    def __len__(self) -> int:
        return len(self._fences)

    @property
    def fences(self) -> tuple[Geofence, ...]:
        """The indexed fences, in build order."""
        return self._fences

    def query(self, xy: Sequence[float]) -> list[str]:
        """Ids of every fence containing the single point ``(x, y)``, sorted.

        Delegates to :meth:`query_many` on purpose: one predicate call exists in
        this module, so the fence/point argument order can only be right or
        wrong in one place.
        """
        return self.query_many([xy])[0]

    def query_many(self, pts) -> list[list[str]]:
        """One sorted id-list per input point. ``pts`` is (N, 2) ENU metres.

        Hot path: a single bulk STRtree call for the whole frame.
        """
        arr = np.asarray(pts, dtype=float).reshape(-1, 2)
        n = arr.shape[0]
        out: list[list[str]] = [[] for _ in range(n)]
        if n == 0 or not self._fences:
            # Also dodges the shape ambiguity of an empty tree/point result.
            return out

        # PREDICATE ORDER, verified empirically against shapely 2.1.2 and NOT
        # what you would guess: STRtree.query evaluates
        # ``predicate(input_geometry, tree_geometry)`` -- the *input* is on the
        # left. Our inputs are points and the tree holds fences, so asking for
        # "contains" asks whether a point contains a polygon, which is never
        # true and returns an empty result set with no error. "within" is the
        # correct spelling of "fence contains point". See test_geofence.py,
        # which pins this with nested fences in both directions.
        idx = self._tree.query(shapely.points(arr), predicate="within")

        # idx is (2, K): row 0 = input point index, row 1 = tree/fence index.
        ids = self._ids
        for pi, fi in zip(idx[0].tolist(), idx[1].tolist()):
            out[pi].append(ids[fi])
        # Sort by id, not by tree order: tree order depends on STRtree's
        # internal packing and is not a stable contract to hand downstream.
        for hits in out:
            hits.sort()
        return out

    def get(self, fence_id: str) -> Geofence:
        """The fence with this id. Raises ``KeyError`` if there is none."""
        return self._by_id[fence_id]


def lonlat_to_enu(
    lonlat: Iterable[Sequence[float]],
    origin_lonlat: Sequence[float],
) -> np.ndarray:
    """Project lon/lat degrees to local ENU metres about ``origin_lonlat``.

    Equirectangular about the origin latitude::

        x = R * (lon - lon0) * cos(lat0)
        y = R * (lat - lat0)

    Accurate to well under a metre across a ~100 km scenario box -- far below
    even AIS position noise -- and a couple of orders of magnitude cheaper than
    a pyproj transform, which we would otherwise pay for on every scenario load.
    Returns an (N, 2) array; the origin maps to exactly (0, 0).
    """
    arr = np.asarray(lonlat, dtype=float).reshape(-1, 2)
    lon0, lat0 = float(origin_lonlat[0]), float(origin_lonlat[1])
    # cos of the *origin* latitude, not per-point: that is what makes the
    # projection an affine map, so ENU metres stay a consistent linear frame.
    cos_lat0 = np.cos(np.radians(lat0))
    out = np.empty_like(arr)
    out[:, 0] = EARTH_RADIUS_M * np.radians(arr[:, 0] - lon0) * cos_lat0
    out[:, 1] = EARTH_RADIUS_M * np.radians(arr[:, 1] - lat0)
    return out


# ==========================================================================
# INTEROP: kind vocabulary, foreign-shape adapters, GeoJSON loading
# ==========================================================================


class GeofenceContractError(contracts.ContractError):
    """A foreign fence shape could not be adapted, and guessing would be worse.

    Subclasses :class:`tracker.contracts.ContractError` (itself a ``ValueError``
    subclass), so a bare ``except contracts.ContractError`` catches both this
    module's failures and contracts.py's own, and existing ``except
    GeofenceContractError`` / ``except ValueError`` callers keep working
    unchanged -- this is still a ``ValueError`` through the chain.
    """


#: The only kind values that may reach :class:`Geofence`. Downstream severity
#: rules switch on these strings; anything else is a merge bug.
#:
#: Sourced from :mod:`tracker.contracts` -- the single vocabulary shared across
#: the tracker, the scenario loader and the orchestration layer. Do NOT reintroduce a
#: private copy here: that is exactly the drift this module used to have (a
#: 4-kind table missing "land", the coastline/grounding-hazard kind).
CANONICAL_KINDS: tuple[str, ...] = contracts.CANONICAL_KINDS

#: Foreign vocabulary -> canonical kind, straight from contracts.py. See
#: :func:`tracker.contracts.normalize_kind` for the unknown-input policy (it
#: raises -- it never guesses).
KIND_ALIASES: dict[str, str] = contracts.KIND_ALIASES

# Marker distinguishing "caller passed default=None" from "caller passed no
# default at all". None is a legitimate thing to want back.
_NO_DEFAULT = object()

# Property keys a GeoJSON feature may use to declare its kind. An allowlist, on
# purpose: scanning every property value for something alias-shaped is how a
# fence silently acquires a kind from an unrelated attribute.
_KIND_PROPERTY_KEYS = ("kind", "type", "category", "class", "fence_type", "layer")
_LABEL_PROPERTY_KEYS = ("label", "name", "title", "sanctuary")
_ID_PROPERTY_KEYS = ("id", "fence_id", "geofence_id", "poly_id")

# id / label / kind / buffer_m are resolved via contracts.resolve_field and
# contracts.FIELD_ALIASES (see geofence_from_mapping) rather than a private
# copy of those tables -- that private-copy drift is exactly what let a
# mapping spelled {"buffer": 500} silently resolve to buffer_m=0.0 before.
#
# Geometry stays geofence-specific: contracts' "geometry" alias list is
# generic (it does not know this module's shapely handling), so the local
# table below is authoritative for geometry, but it is a strict superset of
# contracts.FIELD_ALIASES["geometry"] -- nothing contracts.py already agreed
# to accept ("geometry", "rings") is rejected here.
_GEOM_FIELDS = ("geom", "polygon", "ring", "coordinates", "geometry", "rings")


def _norm_token(raw: object) -> str:
    """Lowercase, trim, and fold spaces/hyphens/dots to single underscores."""
    s = str(raw).strip().lower()
    for ch in (" ", "-", ".", "/"):
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def normalize_kind(raw: str, *, default: Any = _NO_DEFAULT) -> str:
    """Map any known fence vocabulary onto one of :data:`CANONICAL_KINDS`.

    Case-insensitive and tolerant of spaces, hyphens and underscores, so
    ``"Marine Sanctuary"``, ``"marine-sanctuary"`` and ``"MARINE_SANCTUARY"``
    all resolve to ``"mpa"``.

    UNKNOWN INPUT RAISES :class:`GeofenceContractError`. It is not mapped to
    anything. The reasoning:

    * Falling back to ``"mpa"`` invents a marine protected area that does not
      exist. Downstream, "dark vessel inside an MPA" is the CRITICAL rule -- a
      typo in a properties field would manufacture the demo's top alert.
    * Falling back to ``"danger"`` is the *conservative* choice for alerting,
      and it is what a hot path with no operator present should do -- but this
      is not a hot path. Kinds are resolved once, at scenario load, from a
      handful of static files, by a human who can fix the typo in seconds.
      Silently laundering "sancturay" into "danger" trades a loud failure at
      load time for a quiet wrong severity for the whole replay.
    * The two fallbacks disagree about which way to be wrong, which is itself
      evidence that neither is a safe default.

    Callers who genuinely want a fallback must say so explicitly, per call, via
    the keyword-only ``default`` -- which is then returned verbatim and *not*
    validated, so ``default="danger"`` is available to anyone who wants the
    conservative behaviour and has decided it is right for their layer.
    """
    # Delegates to contracts.normalize_kind -- the single source of truth for
    # the kind vocabulary -- and re-wraps its ContractError as
    # GeofenceContractError so existing callers here keep catching the type
    # they already expect. contracts.py's own CANONICAL_KINDS/KIND_ALIASES are
    # used directly (see above), so this never drifts from them again.
    try:
        if default is _NO_DEFAULT:
            return contracts.normalize_kind(raw)
        return contracts.normalize_kind(raw, default=default)
    except contracts.ContractError as exc:
        raise GeofenceContractError(str(exc)) from exc


def _resolve_field(m: object, names: Sequence[str]) -> tuple[str, Any] | None:
    """First present, non-None ``(name, value)`` from a dict OR an object.

    Duck-typed so a plain dict, a ``dataclass`` instance and an ad-hoc namespace
    all work without importing anything from a teammate's branch.
    """
    for name in names:
        if isinstance(m, Mapping):
            if name in m and m[name] is not None:
                return name, m[name]
        else:
            val = getattr(m, name, None)
            if val is not None:
                return name, val
    return None


def _coord_depth(obj: object) -> int:
    """Nesting depth of a coordinate structure: 1 for ``[x, y]``, 2 for a ring,
    3 for a list of rings. Works on lists, tuples and numpy arrays."""
    if isinstance(obj, np.ndarray):
        return int(obj.ndim)
    depth = 0
    cur = obj
    while isinstance(cur, (list, tuple)) and len(cur) > 0:
        depth += 1
        cur = cur[0]
    return depth


def _as_ring(coords: object) -> list[tuple[float, float]]:
    """Coerce anything ring-shaped into a list of ``(x, y)`` float tuples.

    Accepts lists of lists, lists of tuples and ``(N, 2)`` numpy arrays. Closed
    rings (first == last) and unclosed ones both work: shapely closes a ring
    itself, and a duplicated final vertex is harmless, so we do not touch it.
    """
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise GeofenceContractError(
            f"expected a ring of [x, y] pairs, got array of shape {arr.shape}"
        )
    if arr.shape[0] < 3:
        raise GeofenceContractError(
            f"a polygon ring needs at least 3 distinct vertices, got {arr.shape[0]}"
        )
    return [(float(x), float(y)) for x, y in arr[:, :2]]


def _polygon_from_rings(rings: Sequence[object]) -> Polygon:
    """``Polygon(shell, holes)`` from ``[shell, hole, hole, ...]``.

    HOLES MATTER: ``rings[1:]`` are interior boundaries, not extra polygons. A
    sanctuary with a hole punched in it (an excluded harbour, say) must not
    contain points in that hole, so the holes go into the Polygon constructor
    rather than being dropped or unioned.
    """
    shell = _as_ring(rings[0])
    holes = [_as_ring(r) for r in rings[1:]]
    return Polygon(shell, holes)


def _geom_from_coords(coords: object) -> Polygon:
    """Build a polygon from bare coordinates: a ring, or a list of rings."""
    depth = _coord_depth(coords)
    if depth == 2:
        return Polygon(_as_ring(coords))
    if depth == 3:
        return _polygon_from_rings(list(coords))
    raise GeofenceContractError(
        f"cannot read geometry from coordinates of nesting depth {depth}; "
        f"expected a ring [[x, y], ...] (depth 2) or a list of rings "
        f"[shell, hole, ...] (depth 3)"
    )


def geofence_from_ring(
    id: str,
    kind: str,
    label: str,
    ring: Sequence[object],
    buffer_m: float = 0.0,
) -> Geofence:
    """Build a fence from an exterior ring of ``(x, y)`` ENU tuples.

    This is Abhi's ``Geofence.ring`` shape (``Tuple[Tuple[float, float], ...]``,
    closure not required). ``kind`` goes through :func:`normalize_kind`, so his
    vocabulary is accepted here and only canonical kinds leave.
    """
    return polygon_fence(
        id=id,
        kind=normalize_kind(kind),
        label=label,
        coords=_as_ring(ring),
        buffer_m=float(buffer_m),
    )


def geofence_from_mapping(m: object, *, buffer_m: float | None = None) -> Geofence:
    """Build a :class:`Geofence` from any of the three contract shapes.

    Duck-typed: ``m`` may be a ``dict`` (the build plan's shape, or a scenario
    pack's layer entry) or any object with attributes (Abhi's frozen dataclass,
    a namespace, an ORM row). ``id``, ``label``, ``kind`` and ``buffer_m`` are
    resolved via :func:`tracker.contracts.resolve_field` against
    :data:`tracker.contracts.FIELD_ALIASES` -- the single spelling table shared
    with the rest of the project, not a private copy of it. Resolution order,
    first match wins:

    ==========  =====================================================
    ``id``      ``tracker.contracts.FIELD_ALIASES["id"]``
                (``id`` / ``fence_id`` / ``geofence_id`` / ``fenceId``)
    ``label``   ``tracker.contracts.FIELD_ALIASES["label"]``
                (``label`` / ``name`` / ``title``, falls back to ``id``)
    ``kind``    ``tracker.contracts.FIELD_ALIASES["kind"]``
                (``kind`` / ``type`` / ``category``), then :func:`normalize_kind`
    geometry    ``geom`` / ``polygon`` / ``ring`` / ``coordinates`` /
                ``geometry`` / ``rings`` (geofence-specific: this module's
                shapely handling is a superset of contracts' generic list)
    ``buffer_m``  ``tracker.contracts.FIELD_ALIASES["buffer_m"]``
                (``buffer_m`` / ``buffer`` / ``bufferMeters``), else the
                ``buffer_m=`` argument, else
                :func:`tracker.contracts.default_buffer_m` for ``kind``
    ==========  =====================================================

    ``buffer_m`` falls back to :func:`tracker.contracts.default_buffer_m`
    (500 m for ``cable``, 0.0 for everything else) rather than a hardcoded
    ``0.0`` when no field and no argument supplies one -- Abhi's dataclass has
    no ``buffer_m`` member, and a cable layer imported through it with a
    hardcoded zero fallback is a zero-area corridor that can never contain
    anything. Pass ``buffer_m=`` explicitly to override the default in either
    direction.

    RE-BUFFER RULE. ``Geofence.geom`` is *already buffered* and ``buffer_m`` is
    provenance only, so this adapter must not double-buffer a round trip:

    * geometry arrives as an areal shapely geometry (Polygon/MultiPolygon)
      -> taken verbatim, ``buffer_m`` recorded as provenance, nothing applied;
    * geometry arrives as raw coordinates -> polygonized, then buffered by the
      resolved ``buffer_m`` (field, argument, or the per-kind default) if it
      is greater than zero;
    * geometry arrives as a shapely line -> buffered, and an EXPLICIT
      ``buffer_m > 0`` (from the field or the argument) is REQUIRED -- the
      per-kind default is deliberately NOT applied here. A caller who hands
      this adapter a raw, zero-area shapely LineString has made a specific
      choice about geometry and must make an equally specific choice about
      the buffer; silently buffering it by a guessed default would launder a
      missing-field bug into a working-looking corridor of the wrong width.

    Raises :class:`GeofenceContractError` naming the field that was missing.
    A geometry is never guessed or synthesised.
    """
    found_id = contracts.resolve_field(m, "id", default=None)
    if found_id is None:
        raise GeofenceContractError(
            f"geofence is missing an id: none of {contracts.FIELD_ALIASES['id']} "
            f"present on {type(m).__name__}"
        )
    fence_id = str(found_id)

    found_geom = _resolve_field(m, _GEOM_FIELDS)
    if found_geom is None:
        raise GeofenceContractError(
            f"geofence {fence_id!r} is missing a geometry: none of {_GEOM_FIELDS} "
            f"present on {type(m).__name__}. Refusing to guess a geometry."
        )

    found_kind = contracts.resolve_field(m, "kind", default=None)
    if found_kind is None:
        raise GeofenceContractError(
            f"geofence {fence_id!r} is missing a kind: none of "
            f"{contracts.FIELD_ALIASES['kind']} present on {type(m).__name__}; "
            f"expected one of {CANONICAL_KINDS} or a known alias"
        )
    kind = normalize_kind(found_kind)

    found_label = contracts.resolve_field(m, "label", default=None)
    label = str(found_label) if found_label is not None else fence_id

    # `buf` is the EXPLICIT value only (field or argument), never defaulted --
    # it is what gates the raw-shapely-line REQUIRED check below. `effective_buf`
    # is `buf` with the per-kind default filled in, and is what actually gets
    # applied to coordinate-built polygons and recorded as provenance.
    if buffer_m is not None:
        buf: float | None = float(buffer_m)
    else:
        found_buf = contracts.resolve_field(m, "buffer_m", default=None)
        buf = float(found_buf) if found_buf is not None else None
    effective_buf = buf if buf is not None else contracts.default_buffer_m(kind)

    raw_geom = found_geom[1]
    # isinstance FIRST: a shapely geometry is iterable-ish enough that coercing
    # it as a coordinate sequence fails in a confusing way instead of a clear
    # one, and the asdict round trip hands us exactly that.
    if isinstance(raw_geom, BaseGeometry):
        if raw_geom.is_empty:
            raise GeofenceContractError(
                f"geofence {fence_id!r} has an empty geometry ({raw_geom.geom_type})"
            )
        if raw_geom.area > 0.0:
            geom: BaseGeometry = raw_geom          # already buffered; verbatim
            final_buf = effective_buf
        else:
            if buf is None or buf <= 0.0:
                raise GeofenceContractError(
                    f"geofence {fence_id!r} has a zero-area "
                    f"{raw_geom.geom_type} geometry and no explicit buffer_m; a "
                    f"zero-buffered line can never contain a point. Pass "
                    f"buffer_m > 0 (as a field or the buffer_m= argument) for "
                    f"a corridor."
                )
            geom = raw_geom.buffer(buf)
            final_buf = buf
    else:
        geom = _geom_from_coords(raw_geom)
        if effective_buf > 0.0:
            geom = geom.buffer(effective_buf)
        final_buf = effective_buf

    return Geofence(id=fence_id, kind=kind, label=label, geom=geom, buffer_m=final_buf)


# --------------------------------------------------------------------------
# GeoJSON loading (WGS84 lon/lat in, ENU-metre fences out)
# --------------------------------------------------------------------------

_POLYGONAL = ("Polygon", "MultiPolygon")
_LINEAL = ("LineString", "MultiLineString")


def _features_of(doc: Mapping[str, Any], path: str) -> list[Mapping[str, Any]]:
    """Normalize FeatureCollection / Feature / bare geometry into features."""
    doc_type = doc.get("type")
    if doc_type == "FeatureCollection":
        feats = doc.get("features")
        if feats is None:
            raise GeofenceContractError(
                f"{path}: FeatureCollection has no 'features' member"
            )
        return list(feats)
    if doc_type == "Feature":
        return [doc]
    if doc_type in _POLYGONAL + _LINEAL:
        # Bare geometry: wrap it so one code path handles everything.
        return [{"type": "Feature", "geometry": doc, "properties": {}}]
    raise GeofenceContractError(
        f"{path}: unsupported GeoJSON type {doc_type!r}; expected "
        f"FeatureCollection, Feature, or one of {_POLYGONAL + _LINEAL}"
    )


def _prop_lookup(props: Mapping[str, Any], keys: Sequence[str]) -> Any:
    """Case-insensitive first hit over ``keys`` in a properties dict."""
    lowered = {_norm_token(k): v for k, v in props.items()}
    for key in keys:
        val = lowered.get(key)
        if val is not None and val != "":
            return val
    return None


def _project(coords: Sequence[Sequence[float]], origin) -> np.ndarray:
    """Project one GeoJSON position list to ENU, dropping any elevation.

    GeoJSON positions are legally ``[lon, lat]`` OR ``[lon, lat, alt]``, and
    ``lonlat_to_enu`` reshapes to (-1, 2) -- which for an (N, 3) input with even
    N does not raise, it reinterprets the buffer into garbage coordinates that
    still pass "area > 0". So the third ordinate is sliced off *here*, before the
    converter sees it. Not fixed inside ``lonlat_to_enu``: other modules import
    it and its (-1, 2) contract is theirs too.
    """
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise GeofenceContractError(
            f"expected a list of [lon, lat] positions, got shape {arr.shape}"
        )
    return lonlat_to_enu(arr[:, :2], origin)


def _rings_to_enu(rings: Sequence[Sequence[Sequence[float]]], origin) -> Polygon:
    """One GeoJSON polygon (shell + holes), each ring projected separately."""
    enu = [_project(r, origin) for r in rings]
    return _polygon_from_rings(enu)


def load_geojson_fences(
    path: str | os.PathLike[str],
    origin_lonlat: Sequence[float],
    *,
    default_buffer_m: float = 0.0,
    kind: str | None = None,
    label: str | None = None,
    layer: str | None = None,
) -> list[Geofence]:
    """Load a WGS84 lon/lat GeoJSON file into ENU-metre :class:`Geofence` objects.

    Every coordinate is projected through :func:`lonlat_to_enu` about
    ``origin_lonlat`` at load time, so nothing downstream ever sees degrees.

    Accepts a ``FeatureCollection``, a single ``Feature``, or a bare geometry.

    ONE GEOFENCE PER FEATURE, including MultiPolygon. A multi-part feature keeps
    its parts in a single MultiPolygon geometry under one id rather than being
    split into N fences: a feature is one *named* area, and splitting it would
    make a vessel crossing between parts of the same sanctuary look like it left
    one fence and entered another, breaking alert de-duplication downstream.

    Line geometries (a cable route) are buffered into a corridor here, once.
    **A non-zero buffer is required for lines** -- a zero-buffered line has no
    area, can never contain a point, and would silently never fire. Enforced
    per *geometry*, so a mixed collection only demands a buffer for its line
    features.

    ``kind`` resolution: the explicit ``kind=`` argument, else an allowlisted
    feature property (``kind`` / ``type`` / ``category`` / ``class`` /
    ``fence_type`` / ``layer``) through :func:`normalize_kind`, else ``"cable"``
    for a line geometry (geometry is dispositive: a buffered line *is* a
    corridor), else raise. Kind is never inferred from the filename -- that is
    the same guessing :func:`normalize_kind` refuses to do.

    ``label`` resolution: the explicit ``label=`` argument, else a feature
    property (``label`` / ``name`` / ``title`` / ``sanctuary``), else the
    filename stem. Ids come from ``id`` / ``fence_id`` / ``geofence_id`` /
    ``poly_id`` properties, else ``<stem>-<index>``; duplicates get a ``#n``
    suffix so :class:`GeofenceIndex` lookups stay unambiguous.

    ``layer`` NAMESPACING (optional, additive). Two independently loaded files
    can each produce id ``"1"`` (Monterey's fence resolves to ``"1"`` from its
    ``POLY_ID`` property, and so can any other layer that numbers its polygons
    from one); combined into one :class:`GeofenceIndex`, that collision is a
    silent last-wins fence swap. Passing ``layer=`` runs every id produced from
    this file through :func:`tracker.contracts.namespaced_id` (e.g.
    ``"monterey:1"``), after the ``#n`` de-duplication above. The default,
    ``layer=None``, preserves the exact old behaviour -- bare, unprefixed ids --
    so existing callers are unaffected. See also :func:`build_index`, which
    namespaces and combines fences from several layers in one call.
    """
    path_s = os.fspath(path)
    stem = os.path.splitext(os.path.basename(path_s))[0]
    with open(path_s, "r", encoding="utf-8") as fh:
        doc = json.load(fh)

    buf = float(default_buffer_m)
    out: list[Geofence] = []
    seen: dict[str, int] = {}

    for i, feat in enumerate(_features_of(doc, path_s)):
        geometry = feat.get("geometry") if isinstance(feat, Mapping) else None
        if not geometry:
            raise GeofenceContractError(
                f"{path_s}: feature {i} has no geometry (null-geometry features "
                f"are not fences)"
            )
        gtype = geometry.get("type")
        coords = geometry.get("coordinates")
        if coords is None:
            raise GeofenceContractError(
                f"{path_s}: feature {i} geometry {gtype!r} has no 'coordinates'"
            )
        props = feat.get("properties") or {}

        # --- geometry ----------------------------------------------------
        if gtype == "Polygon":
            geom: BaseGeometry = _rings_to_enu(coords, origin_lonlat)
            if buf > 0.0:
                geom = geom.buffer(buf)
        elif gtype == "MultiPolygon":
            parts = [_rings_to_enu(rings, origin_lonlat) for rings in coords]
            geom = MultiPolygon(parts)
            if buf > 0.0:
                geom = geom.buffer(buf)
        elif gtype in _LINEAL:
            if buf <= 0.0:
                raise GeofenceContractError(
                    f"{path_s}: feature {i} is a {gtype} and default_buffer_m="
                    f"{buf}. A zero-buffered line has no area and can never "
                    f"contain anything, so this fence would silently never "
                    f"fire. Pass default_buffer_m > 0 for a cable/route layer."
                )
            if gtype == "LineString":
                line: BaseGeometry = LineString(_project(coords, origin_lonlat))
            else:
                line = MultiLineString(
                    [_project(part, origin_lonlat) for part in coords]
                )
            geom = line.buffer(buf)
        else:
            raise GeofenceContractError(
                f"{path_s}: feature {i} has unsupported geometry type {gtype!r}; "
                f"supported: {_POLYGONAL + _LINEAL}"
            )

        # --- kind --------------------------------------------------------
        if kind is not None:
            fkind = normalize_kind(kind)
        else:
            raw_kind = _prop_lookup(props, _KIND_PROPERTY_KEYS)
            if raw_kind is not None:
                fkind = normalize_kind(raw_kind)
            elif gtype in _LINEAL:
                fkind = "cable"
            else:
                raise GeofenceContractError(
                    f"{path_s}: feature {i} declares no kind (looked for "
                    f"properties {_KIND_PROPERTY_KEYS}) and its geometry "
                    f"({gtype}) does not imply one. Pass kind= explicitly."
                )

        # --- label and id ------------------------------------------------
        if label is not None:
            flabel = label
        else:
            raw_label = _prop_lookup(props, _LABEL_PROPERTY_KEYS)
            flabel = str(raw_label) if raw_label is not None else stem

        raw_id = _prop_lookup(props, _ID_PROPERTY_KEYS)
        fid = str(raw_id) if raw_id is not None else f"{stem}-{i}"
        if fid in seen:
            seen[fid] += 1
            fid = f"{fid}#{seen[fid]}"
        else:
            seen[fid] = 0

        if layer is not None:
            try:
                fid = contracts.namespaced_id(layer, fid)
            except contracts.ContractError as exc:
                raise GeofenceContractError(str(exc)) from exc

        out.append(
            Geofence(id=fid, kind=fkind, label=flabel, geom=geom, buffer_m=buf)
        )

    return out


def build_index(fences_by_layer: Mapping[str, Sequence[Geofence]]) -> GeofenceIndex:
    """Combine fences from several named layers into one collision-safe index.

    The real fix for "two layers both produce fence id '1'": every fence's id
    is run through :func:`tracker.contracts.namespaced_id` for the layer it
    came from, UNLESS it is already namespaced (its id already contains
    ``":"`` -- e.g. it came from :func:`load_geojson_fences` called with
    ``layer=`` already), in which case it is taken verbatim rather than
    double-namespaced. Uniqueness is then asserted across the *combined* id
    set via :func:`tracker.contracts.assert_unique_ids` before the
    :class:`GeofenceIndex` is built, so two fences that still collide --
    typically because they arrived pre-namespaced with the same layer:id pair
    -- raise :class:`GeofenceContractError` instead of one silently replacing
    the other in a last-wins id map.

    ``fences_by_layer`` is ``{layer_name: [Geofence, ...]}``. Layer names are
    only used for namespacing ids that need it; the fences themselves are not
    otherwise modified (kind/label/geom/buffer_m are preserved verbatim).
    """
    namespaced: list[Geofence] = []
    for layer_name, fences in fences_by_layer.items():
        for f in fences:
            if ":" in f.id:
                namespaced.append(f)
            else:
                try:
                    new_id = contracts.namespaced_id(layer_name, f.id)
                except contracts.ContractError as exc:
                    raise GeofenceContractError(str(exc)) from exc
                namespaced.append(replace(f, id=new_id))

    try:
        contracts.assert_unique_ids(f.id for f in namespaced)
    except contracts.ContractError as exc:
        raise GeofenceContractError(str(exc)) from exc

    return GeofenceIndex(namespaced)

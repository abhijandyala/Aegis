"""Shared vocabulary and coercions for Aegis data contracts.

Why this module exists
----------------------
Three people implemented the same written contract three different ways and the
branches have not merged. Every downstream consumer -- the tracker, the scenario
loader, and policy layer otherwise re-invent "what is a geofence, really?" at
its own boundary, which means three places to fix when a spelling changes and
three chances to disagree. This module is the ONE place the vocabulary and the
coercions live.

It is deliberately **dependency-light: stdlib + numpy only.** No shapely, no
project imports, no I/O, no network. That is the whole point -- orchestration and
the data loader must be able to import the vocabulary without pulling in the
geometry stack. A consequence: :func:`_norm_token` is *duplicated* from
``tracker.geofence`` rather than imported. That duplication is intentional. Do
not "fix" it by importing from ``tracker.geofence``; that would reintroduce the
shapely dependency this module exists to avoid.

Field mapping table -- the three real shapes side by side
---------------------------------------------------------
==============  ==================  ==========================  ============================
canonical here  build-plan dict     Abhi (data/contracts.py)     Omar (scenario packs)
==============  ==================  ==========================  ============================
``id``          ``id``              ``fence_id``                 *absent* (layer entry only)
``label``       ``label``           ``name``                     *absent*
``kind``        ``type``            ``kind``                     ``kind``
``geometry``    ``polygon``         ``ring`` (+ ``geojson``)     ``file`` (a path, not geometry)
``buffer_m``    ``buffer_m``        *absent*                     *absent*
==============  ==================  ==========================  ============================

The tracker's own shape (``tracker.geofence.Geofence``) is
``(id, kind, label, geom, buffer_m)`` -- it agrees with this module's canonical
spellings except that its geometry member is ``geom``.

Two of three spellings agree on ``kind`` over ``type`` (and ``type`` shadows a
builtin), so ``kind`` is canonical here. :data:`FIELD_ALIASES` accepts all of
them; :func:`resolve_field` is the reader.

Kind vocabulary, and why there are FIVE kinds
---------------------------------------------
The written contract lists four: ``mpa``, ``cable``, ``port``, ``danger``.
:data:`CANONICAL_KINDS` adds a fifth, ``land``.

**"land" is not a loader convenience -- it is a fix for a gap in the contract.**
The coastline layer (``geo/ca_coastline_10m.geojson``) is loaded as an area and
serves two jobs no other kind covers: it is a grounding hazard, and it is used to
clip dark-vessel uncertainty ellipses so a lost track's probability mass is not
spread across dry land. A coastline is not a marine protected area, not a cable
corridor, not a port and not a danger zone. Forcing it into any of those four
either fabricates an alert class (an "MPA intrusion" every time a vessel moors)
or suppresses a real one. Four kinds is a genuine hole in the written contract,
so the vocabulary grows by one and says why here.

The two silent-failure traps this file exists to prevent
--------------------------------------------------------
1. **A swapped origin is a ~1000 km offset with no error anywhere.** Omar's packs
   carry ``{"lat": 36.75, "lon": -121.95}``; the geometry code takes ``(lon, lat)``
   pairs. Nothing downstream can detect the swap -- the tracks simply run off the
   coast of West Africa and every containment test quietly returns nothing.
   :func:`parse_origin` returns ``(lon, lat)`` always, and *refuses to guess*
   when a bare pair is only self-consistent the other way round.
2. **A zero-buffered cable corridor can never contain anything.** A submarine
   cable is a LineString; the protected corridor is that line dilated. Omar's
   pack layer entries have no ``buffer_m`` field and Abhi's dataclass has no such
   member, so a cable layer imported through either shape gets ``buffer_m=0.0``
   -- a zero-area line that no vessel can ever be inside. The layer loads, the
   tests pass, and the alert never fires. :data:`DEFAULT_BUFFER_M_BY_KIND` gives
   cables a documented non-zero default so the layer is usable at all.

A third trap, narrower but just as quiet, is handled in section 5: Monterey's
fence id resolves to ``"1"`` from its ``POLY_ID`` property, and two separately
loaded layers can each produce ``"1"``. A last-wins id map then silently swaps
one fence for the other. See :func:`namespaced_id`, :func:`dedupe_ids` and
:func:`assert_unique_ids`.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "ContractError",
    "CANONICAL_KINDS",
    "KIND_ALIASES",
    "normalize_kind",
    "FIELD_ALIASES",
    "resolve_field",
    "parse_origin",
    "parse_bbox",
    "DEFAULT_BUFFER_M_BY_KIND",
    "default_buffer_m",
    "namespaced_id",
    "dedupe_ids",
    "assert_unique_ids",
    "parse_epoch",
    "parse_time_window",
]


class ContractError(ValueError):
    """A value could not be reconciled across the three contracts, and guessing
    would be worse than failing.

    A ``ValueError`` subclass so existing ``except ValueError`` handlers still
    catch it, but nameable on its own so a caller can report "your scenario pack
    is wrong" distinctly from arithmetic failures.

    NOTE FOR THE MERGE: this is *not* ``tracker.geofence.GeofenceContractError``.
    The two classes are unrelated -- neither is a subclass of the other -- so
    ``except ContractError`` will not catch geofence.py's failures and vice
    versa. Both subclass ``ValueError``, which is therefore the only handler
    that spans both modules.
    """


# Marker distinguishing "caller passed default=None" from "caller passed no
# default at all". None is a legitimate thing to want back.
_NO_DEFAULT: Any = object()


# ==========================================================================
# 1. KIND VOCABULARY
# ==========================================================================

#: The only kind values that may cross a module boundary. Downstream severity
#: rules switch on these strings; anything else is a merge bug.
#:
#: ``"land"`` is the fifth kind the written contract omits -- see the module
#: docstring for why the coastline layer needs its own kind rather than being
#: filed under one of the other four.
CANONICAL_KINDS: tuple[str, ...] = ("mpa", "cable", "port", "danger", "land")

#: Foreign vocabulary -> canonical kind. Keys are already normalized (lowercase,
#: underscore-separated); :func:`normalize_kind` folds its input the same way, so
#: ``"Marine Sanctuary"``, ``"marine-sanctuary"`` and ``"MARINE_SANCTUARY"`` all
#: hit one entry. Covers Abhi's vocabulary ("sanctuary", "restricted",
#: "anchorage"), Omar's pack ``kind`` values, and the property values that turn
#: up in NOAA / Natural Earth / NGA source layers.
KIND_ALIASES: dict[str, str] = {
    # -> mpa
    "mpa": "mpa",
    "sanctuary": "mpa",
    "marine_sanctuary": "mpa",
    "national_marine_sanctuary": "mpa",
    "nms": "mpa",
    "protected": "mpa",
    "protected_area": "mpa",
    "marine_protected_area": "mpa",
    "reserve": "mpa",
    "preserve": "mpa",
    # -> danger
    "danger": "danger",
    "danger_zone": "danger",
    "restricted": "danger",
    "restricted_area": "danger",
    "military": "danger",
    "military_area": "danger",
    "exclusion": "danger",
    "exclusion_zone": "danger",
    "hazard": "danger",
    # -> port
    "port": "port",
    "anchorage": "port",
    "harbor": "port",
    "harbour": "port",
    "terminal": "port",
    "berth": "port",
    # -> cable
    "cable": "cable",
    "cable_corridor": "cable",
    "corridor": "cable",
    "submarine_cable": "cable",
    "subsea_cable": "cable",
    "pipeline": "cable",
    # -> land (the fifth kind; see module docstring)
    "land": "land",
    "coastline": "land",
    "coast": "land",
    "shore": "land",
    "shoreline": "land",
    "landmask": "land",
    "terrain": "land",
}


def _norm_token(raw: object) -> str:
    """Lowercase, trim, and fold spaces/hyphens/dots/slashes to one underscore.

    Duplicated from ``tracker.geofence`` on purpose -- see the module docstring.
    Applies to kind *values* only. Field *names* are matched verbatim (see
    :func:`resolve_field`): folding them would make camelCase spellings such as
    ``fenceId`` and ``bufferMeters`` unreachable.
    """
    s = str(raw).strip().lower()
    for ch in (" ", "-", ".", "/"):
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def normalize_kind(raw: str, *, default: Any = _NO_DEFAULT) -> str:
    """Map any known fence vocabulary onto one of :data:`CANONICAL_KINDS`.

    Case-insensitive and tolerant of spaces, hyphens, dots and slashes, so
    ``"Marine Sanctuary"``, ``"marine-sanctuary"`` and ``"MARINE_SANCTUARY"``
    all resolve to ``"mpa"``.

    UNKNOWN INPUT RAISES :class:`ContractError`. It is not mapped to anything:

    * Falling back to ``"mpa"`` invents a marine protected area that does not
      exist. "Dark vessel inside an MPA" is the top-severity rule, so a typo in
      one properties field would manufacture the demo's headline alert out of
      nothing.
    * Falling back to ``"danger"`` is the conservative choice for *alerting*,
      but this is not a hot path with no operator present. Kinds are resolved
      once, at scenario load, from a handful of static files, by a human who can
      fix the typo in seconds. Laundering ``"sancturay"`` into ``"danger"``
      trades a loud failure at load time for a quiet wrong severity for the
      whole replay.
    * The two candidate fallbacks disagree about which way to be wrong, which is
      itself evidence that neither is safe as an implicit default.

    A caller who genuinely wants a fallback says so per call via the
    keyword-only ``default``, which is returned **verbatim and unvalidated** --
    so ``default="danger"`` is available to anyone who has decided the
    conservative behaviour is right for their layer.
    """
    token = _norm_token(raw)
    hit = KIND_ALIASES.get(token)
    if hit is not None:
        return hit
    if default is not _NO_DEFAULT:
        return default
    raise ContractError(
        f"unknown geofence kind {raw!r} (normalized to {token!r}); canonical "
        f"kinds are {CANONICAL_KINDS}. Add an entry to "
        f"tracker.contracts.KIND_ALIASES, or pass "
        f"normalize_kind(..., default=...) if a fallback is genuinely correct. "
        f"Refusing to guess: an invented kind changes alert severity."
    )


# ==========================================================================
# 2. FIELD-NAME RECONCILIATION
# ==========================================================================

#: canonical field -> accepted spellings, in priority order (first match wins).
#:
#: Spellings are matched **verbatim**, not normalized, because camelCase entries
#: (``fenceId``, ``bufferMeters``) must stay reachable.
#:
#: ``geometry`` leads with ``"geometry"`` itself even though none of the three
#: contracts spells it that way: a module whose job is to be the single source of
#: truth must not reject its own canonical name. The rest are the real spellings
#: -- ``geom`` (tracker), ``polygon`` (build plan), ``ring`` (Abhi), and the
#: GeoJSON-ish ``coordinates`` / ``rings``.
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id", "fence_id", "geofence_id", "fenceId"),
    "label": ("label", "name", "title"),
    "kind": ("kind", "type", "category"),
    "geometry": ("geometry", "geom", "polygon", "ring", "coordinates", "rings"),
    "buffer_m": ("buffer_m", "buffer", "bufferMeters"),
}


def resolve_field(obj: object, canonical: str, *, default: Any = _NO_DEFAULT) -> Any:
    """Read one canonical field from any of the three shapes.

    Duck-typed over a ``dict``/``Mapping`` **or** an object with attributes, so a
    build-plan dict, Abhi's frozen dataclass, an ad-hoc namespace and an ORM row
    all work without importing anything from a teammate's branch. Each accepted
    spelling in ``FIELD_ALIASES[canonical]`` is tried in order and the first hit
    wins.

    ABSENCE IS ``None``, AND NOTHING ELSE. ``0``, ``0.0``, ``""`` and ``[]`` are
    all *present* values and are returned as-is. This matters most for
    ``buffer_m``: ``0.0`` is a meaningful, deliberate buffer for a polygon fence,
    and treating it as missing would silently substitute a default and change the
    fence's geometry.

    Raises :class:`ContractError` naming the canonical field and every spelling
    tried when nothing is found and no ``default`` was given.
    """
    names = FIELD_ALIASES.get(canonical)
    if names is None:
        raise ContractError(
            f"{canonical!r} is not a canonical field; known fields are "
            f"{tuple(FIELD_ALIASES)}"
        )
    for name in names:
        if isinstance(obj, Mapping):
            if name in obj and obj[name] is not None:
                return obj[name]
        else:
            val = getattr(obj, name, None)
            if val is not None:
                return val
    if default is not _NO_DEFAULT:
        return default
    raise ContractError(
        f"cannot resolve canonical field {canonical!r} on "
        f"{type(obj).__name__}: tried spellings {names} and found none "
        f"(a present-but-None value counts as absent). Pass an explicit "
        f"default= if absence is acceptable here."
    )


# ==========================================================================
# 3. ORIGIN AND BBOX PARSING  (a swapped pair is a silent ~1000 km bug)
# ==========================================================================

_LAT_KEYS = ("lat", "latitude")
_LON_KEYS = ("lon", "longitude", "lng", "long")


def _finite(value: object, what: str) -> float:
    """``float(value)``, refusing NaN/inf -- both propagate silently through the
    projection and produce empty geometry with no error."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{what}: {value!r} is not a number") from exc
    if not np.isfinite(out):
        raise ContractError(f"{what}: {value!r} is not finite")
    return out


def _lookup(m: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in m and m[key] is not None:
            return m[key]
    return None


def parse_origin(origin: object) -> tuple[float, float]:
    """Coerce a scenario origin into ``(lon, lat)`` -- **always that order**.

    Omar's packs carry ``{"lat": 36.75, "lon": -121.95}``; the geometry code
    (``tracker.geofence.lonlat_to_enu``) takes ``(lon, lat)``. A swapped pair is a
    ~1000 km offset that raises nothing anywhere: every track simply lands in the
    wrong ocean and every containment test returns nothing. So this is the one
    conversion in the project that is allowed to be loud.

    Accepted:

    * ``{"lat": ..., "lon": ...}`` or ``{"latitude": ..., "longitude": ...}`` --
      **unambiguous**, because the keys say which is which. Always preferred.
    * a bare 2-sequence -- **ambiguous**. Read as GeoJSON order ``(lon, lat)``,
      which is the standard and what every ``.geojson`` file in ``geo/`` uses,
      then validated: ``|lon| <= 180`` and ``|lat| <= 90``.

    The bare-sequence decision table, given ``(a, b)``::

        ok_direct  = |a| <= 180 and |b| <= 90      # (a, b) works as (lon, lat)
        ok_swapped = |b| <= 180 and |a| <= 90      # (b, a) works as (lon, lat)

        ok_direct                -> return (a, b).  GeoJSON order wins, including
                                    when both readings are valid.
        not ok_direct, ok_swapped-> RAISE. The values are self-consistent ONLY the
                                    other way round, so the caller almost
                                    certainly has a (lat, lon) pair -- but
                                    "almost certainly" is not good enough to
                                    silently reorder someone's scenario origin.
                                    Pass a dict and say which is which.
        neither                  -> RAISE, out of range.

    Worked cases:

    * ``(-121.95, 36.75)`` -> ``(-121.95, 36.75)``. Only ``ok_direct`` holds.
    * ``(36.75, -121.95)`` -> RAISES. ``-121.95`` is not a latitude, so the pair
      only parses swapped. This is exactly Omar's Monterey origin written the
      wrong way round, and it is the case that must never pass quietly.
    * ``(10.0, 20.0)`` -> ``(10.0, 20.0)``. Genuinely ambiguous, both readings
      valid; GeoJSON order applies with no error, as the standard dictates.
    * ``(200.0, 100.0)`` -> RAISES, neither reading is in range.

    NOT silently swapping is the point. Guessing here is precisely how the
    1000 km bug ships.
    """
    if isinstance(origin, Mapping):
        raw_lat = _lookup(origin, _LAT_KEYS)
        raw_lon = _lookup(origin, _LON_KEYS)
        if raw_lat is None or raw_lon is None:
            raise ContractError(
                f"origin mapping must carry both a latitude and a longitude: "
                f"looked for {_LAT_KEYS} and {_LON_KEYS}, got keys "
                f"{sorted(map(str, origin))}"
            )
        lat = _finite(raw_lat, "origin latitude")
        lon = _finite(raw_lon, "origin longitude")
        if abs(lat) > 90.0:
            raise ContractError(f"origin latitude {lat} is outside [-90, 90]")
        if abs(lon) > 180.0:
            raise ContractError(f"origin longitude {lon} is outside [-180, 180]")
        return (lon, lat)

    if isinstance(origin, (list, tuple, np.ndarray)):
        seq = list(np.asarray(origin, dtype=object).ravel())
        if len(seq) != 2:
            raise ContractError(
                f"a bare origin sequence must have exactly 2 values "
                f"(lon, lat); got {len(seq)}"
            )
        a = _finite(seq[0], "origin[0]")
        b = _finite(seq[1], "origin[1]")
        ok_direct = abs(a) <= 180.0 and abs(b) <= 90.0
        ok_swapped = abs(b) <= 180.0 and abs(a) <= 90.0
        if ok_direct:
            # GeoJSON order wins, including when both readings are valid.
            return (a, b)
        if ok_swapped:
            raise ContractError(
                f"ambiguous origin {(a, b)}: read as GeoJSON (lon, lat) the "
                f"latitude {b} is outside [-90, 90], but the pair is valid "
                f"reversed -- it looks like (lat, lon). Refusing to reorder it "
                f"for you: a silently swapped origin is a ~1000 km position "
                f'error with no other symptom. Pass {{"lat": {a}, "lon": {b}}} '
                f"to say which is which."
            )
        raise ContractError(
            f"origin {(a, b)} is out of range under either reading: need "
            f"|lon| <= 180 and |lat| <= 90"
        )

    raise ContractError(
        f"cannot read an origin from {type(origin).__name__}; expected "
        f'{{"lat": .., "lon": ..}} or a 2-sequence (lon, lat)'
    )


def parse_bbox(bbox: object) -> tuple[float, float, float, float]:
    """Coerce a scenario bbox into ``(lon_min, lat_min, lon_max, lat_max)``.

    Omar uses ``{"lon_min": .., "lat_min": .., "lon_max": .., "lat_max": ..}``;
    the build plan uses the array ``[lon_min, lat_min, lon_max, lat_max]``, which
    is also the GeoJSON ``bbox`` member order. Both are accepted.

    Unlike :func:`parse_origin` there is **no swap heuristic here**, and that is
    deliberate: a 4-element bbox is positional and its ordering is fixed by both
    the GeoJSON spec and Omar's explicit keys, so each ordinate is range-checked
    in place. Applying the origin's ambiguity test to bbox corners would reject
    legitimate boxes (any box straddling low latitudes has corners that are
    valid both ways). Also requires strict ``min < max`` -- a degenerate box has
    zero area and silently selects nothing.
    """
    if isinstance(bbox, Mapping):
        vals = []
        for key in ("lon_min", "lat_min", "lon_max", "lat_max"):
            if bbox.get(key) is None:
                raise ContractError(
                    f"bbox mapping is missing {key!r}; need lon_min, lat_min, "
                    f"lon_max, lat_max (got keys {sorted(map(str, bbox))})"
                )
            vals.append(_finite(bbox[key], f"bbox {key}"))
        lon_min, lat_min, lon_max, lat_max = vals
    elif isinstance(bbox, (list, tuple, np.ndarray)):
        seq = list(np.asarray(bbox, dtype=object).ravel())
        if len(seq) != 4:
            raise ContractError(
                f"a bbox sequence must have exactly 4 values "
                f"(lon_min, lat_min, lon_max, lat_max); got {len(seq)}"
            )
        lon_min = _finite(seq[0], "bbox lon_min")
        lat_min = _finite(seq[1], "bbox lat_min")
        lon_max = _finite(seq[2], "bbox lon_max")
        lat_max = _finite(seq[3], "bbox lat_max")
    else:
        raise ContractError(
            f"cannot read a bbox from {type(bbox).__name__}; expected a mapping "
            f"with lon_min/lat_min/lon_max/lat_max or a 4-sequence"
        )

    for name, value in (("lon_min", lon_min), ("lon_max", lon_max)):
        if abs(value) > 180.0:
            raise ContractError(f"bbox {name} {value} is outside [-180, 180]")
    for name, value in (("lat_min", lat_min), ("lat_max", lat_max)):
        if abs(value) > 90.0:
            raise ContractError(f"bbox {name} {value} is outside [-90, 90]")
    if not lon_min < lon_max:
        raise ContractError(
            f"bbox lon_min ({lon_min}) must be strictly less than lon_max "
            f"({lon_max}); a degenerate box has no area and selects nothing"
        )
    if not lat_min < lat_max:
        raise ContractError(
            f"bbox lat_min ({lat_min}) must be strictly less than lat_max "
            f"({lat_max}); a degenerate box has no area and selects nothing"
        )
    return (lon_min, lat_min, lon_max, lat_max)


# ==========================================================================
# 4. BUFFER DEFAULTS BY KIND
# ==========================================================================

#: canonical kind -> the buffer, in metres, to use when no ``buffer_m`` is given.
#:
#: ``cable`` is the only non-zero entry, and it is the reason this table exists.
#: A cable corridor's source geometry is a LineString; the protected area is that
#: line dilated. With ``buffer_m = 0.0`` the fence is a zero-area line that can
#: NEVER contain a point -- it loads clean, indexes clean, and never fires. Omar's
#: pack layer entries have nowhere to declare a buffer and Abhi's dataclass has no
#: such member, so without a default a cable layer referenced from a pack cannot
#: produce a usable fence at all. 500 m is the order of magnitude used for
#: submarine-cable protection zones and is a documented starting point, not a
#: measured value: override it per layer when the real corridor width is known.
#:
#: Every areal kind defaults to 0.0 -- a polygon fence already has area, and an
#: unrequested buffer would quietly enlarge a sanctuary and over-report
#: intrusions.
DEFAULT_BUFFER_M_BY_KIND: dict[str, float] = {
    "mpa": 0.0,
    "cable": 500.0,
    "port": 0.0,
    "danger": 0.0,
    "land": 0.0,
}


def default_buffer_m(kind: str) -> float:
    """The default buffer in metres for ``kind``, which is normalized first.

    Normalizing first is load-bearing: Omar's layer entries say
    ``{"kind": "cable_corridor"}``, and a raw dict lookup on that would raise
    ``KeyError`` -- which a caller wrapping this in a ``try`` would most likely
    turn back into 0.0, re-creating the exact zero-buffered-cable trap this table
    exists to close. An unrecognised kind raises :class:`ContractError`.
    """
    return DEFAULT_BUFFER_M_BY_KIND[normalize_kind(kind)]


# ==========================================================================
# 5. ID NAMESPACING  (fixes a silent fence swap)
# ==========================================================================

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _sanitise_layer(layer: str) -> str:
    """Lowercase, fold runs of non-alphanumerics to a single underscore, trim.

    Sanitising keeps namespaced ids **stable** across the ways a layer gets
    named: a file stem, a display name and a path fragment for the same layer
    must not produce three different id namespaces, or a fence's id changes
    depending on which loader reached it first and alert de-duplication breaks.
    """
    out = _NON_ALNUM.sub("_", str(layer).strip().lower()).strip("_")
    if not out:
        raise ContractError(
            f"layer name {layer!r} sanitises to the empty string; it cannot "
            f"namespace an id"
        )
    return out


def namespaced_id(layer: str, raw_id: object) -> str:
    """``"<sanitised layer>:<raw id>"``, e.g. ``"monterey_bay_nms:1"``.

    Monterey's fence id resolves to ``"1"`` from its ``POLY_ID`` property. So does
    a feature in any other layer that numbers its polygons from one. Two layers
    loaded into one id map then collide, and last-wins means one fence silently
    *replaces* the other: the containment tests still pass, the index still has
    the right length, and one protected area has quietly vanished from the
    scenario. Namespacing by layer makes the collision impossible instead of
    merely unlikely.

    The raw id is kept verbatim (only stringified) so it still matches the source
    file's property and stays greppable.
    """
    text = str(raw_id).strip()
    if not text:
        raise ContractError(f"cannot namespace an empty raw id (layer {layer!r})")
    return f"{_sanitise_layer(layer)}:{text}"


def dedupe_ids(ids: Sequence[str]) -> list[str]:
    """Make ``ids`` unique in place-order by suffixing collisions ``#1``, ``#2``.

    Order-preserving and deterministic: the first occurrence keeps its id, later
    ones get the lowest free suffix. The ``#n`` spelling matches
    ``tracker.geofence.load_geojson_fences`` so the two modules read consistently
    at merge time.

    The candidate suffix is re-checked against everything already emitted, which
    is not paranoia: given ``["1", "1", "1#1"]`` a naive counter emits ``"1#1"``
    twice and reintroduces exactly the silent fence swap this section exists to
    prevent.
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw in ids:
        base = str(raw)
        candidate = base
        n = 0
        while candidate in seen:
            n += 1
            candidate = f"{base}#{n}"
        seen.add(candidate)
        out.append(candidate)
    return out


def assert_unique_ids(ids: Iterable[str]) -> None:
    """Raise :class:`ContractError` listing every duplicated id, or return None.

    The checker exists alongside :func:`dedupe_ids` because the two answer
    different questions. ``dedupe_ids`` is right when you are *building* an id
    space and a suffix is an acceptable outcome. This is right when duplicates
    mean something upstream is already wrong -- two layers claiming the same
    fence -- and renaming would paper over it. A loader that dedupes silently can
    never tell you your layer list is broken.

    Duplicates are reported sorted, so the message is deterministic and testable.
    """
    counts: dict[str, int] = {}
    for raw in ids:
        key = str(raw)
        counts[key] = counts.get(key, 0) + 1
    dupes = sorted(k for k, n in counts.items() if n > 1)
    if dupes:
        detail = ", ".join(f"{k!r} x{counts[k]}" for k in dupes)
        raise ContractError(
            f"duplicate ids: {detail}. A last-wins id map would silently drop "
            f"one of each pair. Namespace them with namespaced_id(layer, id), "
            f"or call dedupe_ids() if a #n suffix is acceptable here."
        )


# ==========================================================================
# 6. TIME PARSING
# ==========================================================================


def parse_epoch(value: object) -> float:
    """Coerce a timestamp to **UTC epoch seconds**.

    Omar's ``time_window`` is ISO-8601 strings
    (``"2024-06-15T06:00:00+00:00"``); the build plan says epoch seconds
    (``start_t`` / ``end_t``). Accepted here: an ``int``/``float`` epoch (passed
    through), an ISO-8601 string (with a ``Z`` suffix, an explicit offset, or
    none), or a ``datetime``.

    **A NAIVE datetime or ISO string is treated as UTC.** This is stated
    explicitly because Python's default is the opposite and the default is
    wrong for us: ``datetime.timestamp()`` on a naive datetime applies the
    *machine's* local zone, so the same scenario pack would replay at a
    different wall-clock time on every developer's laptop and shift every
    timestamp by whole hours -- silently, since the numbers stay plausible.
    Scenario data is authored in UTC; a missing offset means UTC, not "wherever
    this happens to run".
    """
    if isinstance(value, bool):
        # bool is an int subclass; a True timestamp is a bug, not epoch 1.
        raise ContractError(f"cannot read a timestamp from {value!r}")
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float, np.integer, np.floating)):
        return _finite(value, "epoch seconds")
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ContractError("cannot read a timestamp from an empty string")
        try:
            # Python 3.12's fromisoformat handles a trailing 'Z' and explicit
            # offsets natively, so no manual suffix surgery is needed.
            dt = datetime.fromisoformat(text)
        except ValueError:
            # A bare epoch that arrived as a string ("1718431200") is a common
            # JSON-authoring accident and unambiguous, so accept it.
            try:
                numeric = float(text)
            except ValueError:
                raise ContractError(
                    f"cannot parse timestamp {value!r}: expected ISO-8601 "
                    f"(e.g. '2024-06-15T06:00:00+00:00') or epoch seconds"
                ) from None
            return _finite(numeric, "epoch seconds")
    else:
        raise ContractError(
            f"cannot read a timestamp from {type(value).__name__}; expected "
            f"epoch seconds, an ISO-8601 string, or a datetime"
        )

    if dt.tzinfo is None:
        # THE naive branch. Do not delete: without it, .timestamp() silently
        # applies the local zone.
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


_START_KEYS = ("start", "start_t", "t0", "begin")
_END_KEYS = ("end", "end_t", "t1", "finish")


def parse_time_window(tw: object) -> tuple[float, float]:
    """Coerce a time window to ``(start_epoch, end_epoch)`` in UTC seconds.

    Accepts Omar's ``{"start": iso, "end": iso}`` and the build plan's
    ``{"start_t": epoch, "end_t": epoch}`` -- mixed forms too, since each bound
    goes through :func:`parse_epoch` independently. A 2-sequence
    ``[start, end]`` is also accepted; it is positional and unambiguous.

    Requires ``start < end``: a zero-length or reversed window replays nothing
    and would otherwise look like a scenario with no detections.
    """
    if isinstance(tw, Mapping):
        raw_start = _lookup(tw, _START_KEYS)
        raw_end = _lookup(tw, _END_KEYS)
        if raw_start is None or raw_end is None:
            raise ContractError(
                f"time window must carry both bounds: looked for {_START_KEYS} "
                f"and {_END_KEYS}, got keys {sorted(map(str, tw))}"
            )
    elif isinstance(tw, (list, tuple, np.ndarray)):
        if len(tw) != 2:
            raise ContractError(
                f"a bare time window sequence must have exactly 2 values "
                f"(start, end); got {len(tw)}"
            )
        raw_start, raw_end = tw[0], tw[1]
    else:
        raise ContractError(
            f"cannot read a time window from {type(tw).__name__}; expected "
            f'{{"start": .., "end": ..}}, {{"start_t": .., "end_t": ..}} or a '
            f"2-sequence"
        )

    start = parse_epoch(raw_start)
    end = parse_epoch(raw_end)
    if not start < end:
        raise ContractError(
            f"time window start ({start}) must be strictly before end ({end}); "
            f"an empty window replays nothing and looks like a scenario with no "
            f"detections"
        )
    return (start, end)

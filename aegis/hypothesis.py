"""Deterministic multi-hypothesis lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field

NSCAN_DEPTH = 3
WEIGHT_FLOOR = 0.01
MAX_LIVE = 8

LOG: list[str] = []
ROOTS: list["Hypothesis"] = []


@dataclass
class Hypothesis:
    hyp_id: str
    weight: float = 1.0
    depth: int = 0
    alive: bool = True
    parent_id: str = ""
    superseded: bool = False
    prune_reason: str = ""
    _parent: "Hypothesis | None" = field(default=None, repr=False)
    _children: list["Hypothesis"] = field(default_factory=list, repr=False)


def fmt2(w: float) -> str:
    return format(w, ".2f")


def emit(line: str) -> None:
    LOG.append(line)
    print(line)


def _walk(node: Hypothesis):
    yield node
    for child in node._children:
        yield from _walk(child)


def live_leaves() -> list[Hypothesis]:
    return [node for root in ROOTS for node in _walk(root) if node.alive]


def branch_root_id(h: Hypothesis) -> str:
    while h._parent is not None:
        h = h._parent
    return h.hyp_id


def heaviest(hs: list[Hypothesis]) -> Hypothesis:
    best = hs[0]
    for h in hs:
        if h.weight > best.weight or (
            h.weight == best.weight and h.hyp_id < best.hyp_id
        ):
            best = h
    return best


def lightest(hs: list[Hypothesis]) -> Hypothesis:
    worst = hs[0]
    for h in hs:
        if h.weight < worst.weight or (
            h.weight == worst.weight and h.hyp_id > worst.hyp_id
        ):
            worst = h
    return worst


def normalize() -> None:
    leaves = live_leaves()
    if not leaves:
        return
    total = sum(h.weight for h in leaves)
    if total <= 0.0:
        uniform = 1.0 / len(leaves)
        for h in leaves:
            h.weight = uniform
        return
    for h in leaves:
        h.weight /= total


def prune(h: Hypothesis, reason: str) -> None:
    if not h.alive:
        return
    h.alive = False
    h.prune_reason = reason
    emit("[Hypothesis] pruned " + h.hyp_id)


def split(h: Hypothesis, w1: float, w2: float) -> list[Hypothesis]:
    a = Hypothesis(
        hyp_id=h.hyp_id + "a",
        weight=h.weight * w1,
        depth=h.depth + 1,
        parent_id=h.hyp_id,
        _parent=h,
    )
    b = Hypothesis(
        hyp_id=h.hyp_id + "b",
        weight=h.weight * w2,
        depth=h.depth + 1,
        parent_id=h.hyp_id,
        _parent=h,
    )
    h._children.extend((a, b))
    h.alive = False
    h.superseded = True
    emit(
        "[Hypothesis] split "
        + h.hyp_id
        + " -> "
        + a.hyp_id
        + " ("
        + fmt2(a.weight)
        + ") / "
        + b.hyp_id
        + " ("
        + fmt2(b.weight)
        + ")"
    )
    return [a, b]


def nscan_collapse() -> None:
    groups: dict[str, list[Hypothesis]] = {}
    for h in live_leaves():
        groups.setdefault(branch_root_id(h), []).append(h)
    for members in groups.values():
        if len(members) < 2 or not any(h.depth >= NSCAN_DEPTH for h in members):
            continue
        keep = heaviest(members)
        for h in members:
            if h.hyp_id != keep.hyp_id:
                prune(h, "nscan")


def weight_kill() -> None:
    leaves = live_leaves()
    if len(leaves) < 2:
        return
    keep = heaviest(leaves)
    for h in leaves:
        if h.hyp_id != keep.hyp_id and h.weight < WEIGHT_FLOOR:
            prune(h, "weight")


def cap_enforce() -> None:
    leaves = live_leaves()
    while len(leaves) > MAX_LIVE:
        prune(lightest(leaves), "cap")
        leaves = live_leaves()


def maintain() -> None:
    normalize()
    nscan_collapse()
    weight_kill()
    cap_enforce()
    normalize()


def pad2(i: int) -> str:
    text = str(i)
    return "0" + text if len(text) < 2 else text


def reset(n_roots: int = 1) -> None:
    ROOTS.clear()
    LOG.clear()
    for i in range(n_roots):
        ROOTS.append(Hypothesis(hyp_id="h_" + pad2(i)))
    normalize()


def run_frame(
    ambiguous: bool,
    w1: float = 0.5,
    w2: float = 0.5,
    fork_all: bool = False,
) -> None:
    leaves = live_leaves()
    if not leaves:
        return
    if ambiguous:
        targets = leaves if fork_all else [heaviest(leaves)]
        for h in targets:
            split(h, w1, w2)
    else:
        for h in leaves:
            h.depth += 1
    maintain()


def live_ids() -> list[str]:
    return sorted(h.hyp_id for h in live_leaves())


def live_count() -> int:
    return len(live_leaves())


def live_weight_sum() -> float:
    return sum(h.weight for h in live_leaves())


def live_state() -> list[tuple[str, float, int]]:
    return sorted((h.hyp_id, h.weight, h.depth) for h in live_leaves())


def max_live_depth() -> int:
    return max((h.depth for h in live_leaves()), default=0)


def get_log() -> list[str]:
    return list(LOG)

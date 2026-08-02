"""Deterministic justification-based truth maintenance system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAX_PASSES = 64


@dataclass
class Fact:
    fact_id: str
    label: str
    believed: bool = True


@dataclass
class Conclusion:
    concl_id: str
    label: str
    status: str = "IN"


FACTS: dict[str, Fact] = {}
CONCLUSIONS: dict[str, Conclusion] = {}
JUSTS: dict[str, dict[str, Any]] = {}
JUST_ORDER: dict[str, list[str]] = {}
PREV_STATUS: dict[str, str] = {}


def add_fact(fact_id: str, label: str, believed: bool = True) -> Fact:
    if fact_id in FACTS:
        raise ValueError(f"duplicate fact_id: {fact_id}")
    fact = Fact(fact_id=fact_id, label=label, believed=believed)
    FACTS[fact_id] = fact
    return fact


def add_conclusion(concl_id: str, label: str) -> Conclusion:
    if concl_id in CONCLUSIONS:
        raise ValueError(f"duplicate concl_id: {concl_id}")
    conclusion = Conclusion(concl_id=concl_id, label=label)
    CONCLUSIONS[concl_id] = conclusion
    JUST_ORDER[concl_id] = []
    return conclusion


def _node_for(node_id: str) -> Fact | Conclusion:
    if node_id in FACTS:
        return FACTS[node_id]
    if node_id in CONCLUSIONS:
        return CONCLUSIONS[node_id]
    raise ValueError(f"unknown fact/conclusion id: {node_id}")


def justify(
    concl_id: str,
    justification_id: str,
    supports_ids: list[str] | None = None,
    refutes_ids: list[str] | None = None,
) -> None:
    if concl_id not in CONCLUSIONS:
        raise ValueError(f"unknown concl_id: {concl_id}")
    if justification_id in JUSTS:
        raise ValueError(f"duplicate justification_id: {justification_id}")

    supports = list(supports_ids or [])
    refutes = list(refutes_ids or [])
    for node_id in supports:
        _node_for(node_id)
    for node_id in refutes:
        _node_for(node_id)

    JUSTS[justification_id] = {
        "concl_id": concl_id,
        "supports": supports,
        "refutes": refutes,
    }
    JUST_ORDER[concl_id].append(justification_id)


def _antecedent_believed(node_id: str) -> bool:
    if node_id in FACTS:
        return FACTS[node_id].believed
    if node_id in CONCLUSIONS:
        return CONCLUSIONS[node_id].status == "IN"
    raise ValueError(f"unknown antecedent/inhibitor id: {node_id}")


def _justification_satisfied(justification_id: str) -> bool:
    justification = JUSTS[justification_id]
    return all(_antecedent_believed(node_id) for node_id in justification["supports"]) and all(
        not _antecedent_believed(node_id) for node_id in justification["refutes"]
    )


def _conclusion_should_be_in(concl_id: str) -> bool:
    return any(_justification_satisfied(jid) for jid in JUST_ORDER[concl_id])


def _recompute_all_once() -> list[str]:
    wanted = {
        concl_id: "IN" if _conclusion_should_be_in(concl_id) else "OUT"
        for concl_id in CONCLUSIONS
    }
    changed: list[str] = []
    for concl_id, conclusion in CONCLUSIONS.items():
        if conclusion.status != wanted[concl_id]:
            conclusion.status = wanted[concl_id]
            changed.append(concl_id)
    return changed


def propagate() -> list[str]:
    before = {concl_id: conclusion.status for concl_id, conclusion in CONCLUSIONS.items()}
    for _ in range(MAX_PASSES):
        if not _recompute_all_once():
            break
    else:
        raise RuntimeError(
            "JTMS propagation did not reach a fixpoint within "
            f"{MAX_PASSES} passes -- likely a cyclic justification graph"
        )

    PREV_STATUS.clear()
    PREV_STATUS.update(before)
    return sorted(
        concl_id
        for concl_id, previous in before.items()
        if previous != CONCLUSIONS[concl_id].status
    )


def retract(fact_id: str) -> list[str]:
    if fact_id not in FACTS:
        raise ValueError(f"unknown fact_id: {fact_id}")
    FACTS[fact_id].believed = False
    return propagate()


def reinstate(fact_id: str) -> list[str]:
    if fact_id not in FACTS:
        raise ValueError(f"unknown fact_id: {fact_id}")
    FACTS[fact_id].believed = True
    return propagate()


def _label_of(node_id: str) -> str:
    if node_id in FACTS:
        return FACTS[node_id].label
    if node_id in CONCLUSIONS:
        return CONCLUSIONS[node_id].label
    return node_id


def explain(concl_id: str) -> dict[str, Any]:
    if concl_id not in CONCLUSIONS:
        raise ValueError(f"unknown concl_id: {concl_id}")

    justifications: list[dict[str, Any]] = []
    for jid in JUST_ORDER[concl_id]:
        raw = JUSTS[jid]
        antecedents = [
            {
                "id": node_id,
                "label": _label_of(node_id),
                "in_or_believed": _antecedent_believed(node_id),
                "satisfied": _antecedent_believed(node_id),
            }
            for node_id in raw["supports"]
        ]
        inhibitors = [
            {
                "id": node_id,
                "label": _label_of(node_id),
                "in_or_believed": _antecedent_believed(node_id),
                "satisfied": not _antecedent_believed(node_id),
            }
            for node_id in raw["refutes"]
        ]
        justifications.append(
            {
                "justification_id": jid,
                "satisfied": all(item["satisfied"] for item in antecedents + inhibitors),
                "antecedents": antecedents,
                "inhibitors": inhibitors,
            }
        )

    conclusion = CONCLUSIONS[concl_id]
    return {
        "concl_id": concl_id,
        "label": conclusion.label,
        "status": conclusion.status,
        "justifications": justifications,
    }


def _antecedent_clause(justification: dict[str, Any]) -> str:
    parts = [f"{item['label']} is believed" for item in justification["antecedents"]]
    parts.extend(f"{item['label']} is not believed" for item in justification["inhibitors"])
    return " and ".join(parts) if parts else "trivially (no antecedents or inhibitors)"


def _unsatisfied_clause(justification: dict[str, Any]) -> str:
    bad = [
        f"{item['label']} is no longer believed"
        for item in justification["antecedents"]
        if not item["satisfied"]
    ]
    bad.extend(
        f"{item['label']} is now believed"
        for item in justification["inhibitors"]
        if not item["satisfied"]
    )
    return " and ".join(bad) if bad else "unknown reason"


def _reason_sentence(explanation: dict[str, Any]) -> str:
    justifications = explanation["justifications"]
    if not justifications:
        return "no justification exists"
    if explanation["status"] == "IN":
        for justification in justifications:
            if justification["satisfied"]:
                return (
                    f"justification {justification['justification_id']} satisfied: "
                    + _antecedent_clause(justification)
                )
        return "no satisfied justification found"

    primary = justifications[0]
    extra = ""
    if len(justifications) > 1:
        extra = (
            f"; no alternate justification holds "
            f"({len(justifications) - 1} other(s) also unsatisfied)"
        )
    return (
        f"justification {primary['justification_id']} not satisfied: "
        f"{_unsatisfied_clause(primary)}{extra}"
    )


def brief_for(concl_id: str) -> str:
    explanation = explain(concl_id)
    status = explanation["status"]
    previous = PREV_STATUS.get(concl_id, status)
    transition = f"{previous} -> {status}" if previous != status else status
    return f"{concl_id}: {transition} ({_reason_sentence(explanation)})"


def fact_believed(fact_id: str) -> bool:
    return FACTS[fact_id].believed


def conclusion_status(concl_id: str) -> str:
    return CONCLUSIONS[concl_id].status


def fact_ids() -> list[str]:
    return sorted(FACTS)


def conclusion_ids() -> list[str]:
    return sorted(CONCLUSIONS)


def justification_ids_for(concl_id: str) -> list[str]:
    return list(JUST_ORDER[concl_id])


def reset() -> None:
    FACTS.clear()
    CONCLUSIONS.clear()
    JUSTS.clear()
    JUST_ORDER.clear()
    PREV_STATUS.clear()


def build_mmsi_spoof_demo() -> None:
    add_fact("broadcast_a_mmsi", "vessel_a broadcasts MMSI 412345678")
    add_fact("broadcast_b_mmsi", "vessel_b broadcasts MMSI 412345678")
    add_fact(
        "physically_impossible",
        "vessel_a and vessel_b broadcasting MMSI 412345678 simultaneously, "
        "40 km apart, is physically impossible",
    )
    add_conclusion("identity_a_confirmed", "vessel_a's identity is confirmed")
    add_conclusion("identity_b_confirmed", "vessel_b's identity is confirmed")
    add_conclusion("spoof_detected", "an MMSI spoof is in progress")
    justify("identity_a_confirmed", "j_a1", supports_ids=["broadcast_a_mmsi"])
    justify(
        "identity_b_confirmed",
        "j_b1",
        supports_ids=["broadcast_b_mmsi"],
        refutes_ids=["spoof_detected"],
    )
    justify(
        "spoof_detected",
        "j_spoof_a",
        supports_ids=["physically_impossible", "broadcast_a_mmsi"],
    )
    justify(
        "spoof_detected",
        "j_spoof_b",
        supports_ids=["physically_impossible", "broadcast_b_mmsi"],
    )

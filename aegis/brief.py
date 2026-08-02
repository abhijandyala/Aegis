"""JSON-ready brief-panel data built from :mod:`aegis.jtms`."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from aegis import jtms

LLM_MODEL_NAME = os.environ.get("AEGIS_LLM", "mockllm")
LLM_MODEL_NAME_BOX = [LLM_MODEL_NAME]

_MOCK_OUTPUTS = [
    "Belief status update: this conclusion's support has just been "
    "recomputed from the justification graph.",
    "The evidence behind this conclusion has shifted, and its status now reflects that.",
    "This conclusion's justification structure was re-evaluated against current beliefs.",
]
MODEL_BOX: list[dict[str, Any]] = []


@dataclass
class Brief:
    concl_id: str
    deterministic_text: str
    polished_text: str = ""
    model_used: str = ""
    cost_usd: float = 0.0
    source_ids: list[str] = field(default_factory=list)


@dataclass
class CostLedger:
    total_calls: int = 0
    total_usd: float = 0.0


LEDGER: list[CostLedger] = []


def set_llm_model(name: str) -> None:
    """Configure the model used for subsequent brief composition."""
    MODEL_BOX.clear()
    MODEL_BOX.append({"name": name, "outputs": _MOCK_OUTPUTS * 20})
    LLM_MODEL_NAME_BOX[0] = name


def reset_llm_model() -> None:
    """Restore the environment-selected model, or the offline mock."""
    set_llm_model(os.environ.get("AEGIS_LLM", "mockllm"))


def polish_text(deterministic_text: str) -> str:
    """Rewrite via the deterministic offline model.

    Non-mock models are handled by the direct LiteLLM tier, which retains
    provider configuration and response cost information.
    """
    model = MODEL_BOX[0]
    if model["name"] != "mockllm":
        raise RuntimeError("configured model is not the offline mock")
    try:
        return model["outputs"].pop(0)
    except IndexError as exc:
        raise RuntimeError("mockllm outputs exhausted") from exc


def _ledger() -> CostLedger:
    if not LEDGER:
        LEDGER.append(CostLedger())
    return LEDGER[0]


def record_call(model_used: str, cost_usd: float) -> None:
    del model_used
    ledger = _ledger()
    ledger.total_calls += 1
    ledger.total_usd += cost_usd


def ledger_summary() -> dict[str, int | float]:
    ledger = _ledger()
    return {"calls": ledger.total_calls, "usd": ledger.total_usd}


def reset_ledger() -> None:
    LEDGER.clear()


def source_ids_for(concl_id: str) -> list[str]:
    explanation = jtms.explain(concl_id)
    seen: set[str] = set()
    source_ids: list[str] = []
    for justification in explanation["justifications"]:
        for item in justification["antecedents"] + justification["inhibitors"]:
            if item["id"] not in seen:
                seen.add(item["id"])
                source_ids.append(item["id"])
    return source_ids


def _litellm_polish(text: str) -> dict[str, str | float]:
    import litellm

    litellm.suppress_debug_info = True
    response = litellm.completion(
        model=LLM_MODEL_NAME_BOX[0],
        messages=[
            {
                "role": "user",
                "content": (
                    "Rewrite this one sentence with clearer prose. Preserve every "
                    f"fact exactly; add nothing, remove nothing: {text}"
                ),
            }
        ],
        max_tokens=60,
        timeout=10,
    )
    output = response.choices[0].message.content.strip()
    try:
        cost = float(litellm.completion_cost(completion_response=response))
    except Exception:
        cost = 0.0
    return {"text": output, "cost": cost}


def compose_brief(concl_id: str, use_llm: bool = True) -> Brief:
    deterministic_text = jtms.brief_for(concl_id)
    brief = Brief(
        concl_id=concl_id,
        deterministic_text=deterministic_text,
        polished_text=deterministic_text,
        model_used="template",
        cost_usd=0.0,
        source_ids=source_ids_for(concl_id),
    )
    if not use_llm:
        return brief

    try:
        brief.polished_text = polish_text(deterministic_text)
        brief.model_used = LLM_MODEL_NAME_BOX[0]
        record_call(brief.model_used, brief.cost_usd)
        return brief
    except Exception:
        pass

    try:
        result = _litellm_polish(deterministic_text)
        brief.polished_text = str(result["text"])
        brief.model_used = "litellm-fallback"
        brief.cost_usd = float(result["cost"])
        record_call(brief.model_used, brief.cost_usd)
    except Exception:
        brief.polished_text = deterministic_text
        brief.model_used = "template"
        brief.cost_usd = 0.0
    return brief


def panel_payload(concl_ids: list[str], use_llm: bool = True) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for concl_id in concl_ids:
        brief = compose_brief(concl_id, use_llm=use_llm)
        payload.append(
            {
                "concl_id": brief.concl_id,
                "text": brief.polished_text,
                "model_used": brief.model_used,
                "cost_usd": float(brief.cost_usd),
                "source_ids": list(brief.source_ids),
                "status": jtms.conclusion_status(concl_id),
            }
        )
    return payload


reset_llm_model()

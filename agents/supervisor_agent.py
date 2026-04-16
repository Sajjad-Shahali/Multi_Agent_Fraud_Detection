"""Node 4 — Supervisor Agent.

Receives the outputs of both specialist agents plus the full FraudState, applies
asymmetric cost-aware thresholds, and issues the final verdict: FRAUD, REVIEW, or CLEAN.
Both FRAUD and REVIEW are included in the output file.

# REFACTOR: uses STRONG_MODEL env var (default openai/gpt-4o) — evidence weighting
# and conflict resolution require the strongest available reasoning.
"""

import json
import os
import re
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from prompts.supervisor_prompt import SUPERVISOR_SYSTEM_PROMPT
from state import FraudState

_VALID_VERDICTS = {"FRAUD", "REVIEW", "CLEAN"}
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _build_llm() -> ChatOpenAI:
    """Instantiate the LLM from environment variables."""
    # REFACTOR: STRONG_MODEL for supervisor — evidence arbitration and asymmetric
    # cost-aware decision requires the best available reasoning model
    return ChatOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        model=os.getenv("STRONG_MODEL", "openai/gpt-4o"),
        temperature=0.0,
        max_tokens=2048,
    )


def _format_user_message(state: FraudState) -> str:
    """Build the supervisor's user message from both specialist outputs."""
    tx_signal = state["transaction_fraud_signal"]
    comms_signal = state["comms_fraud_signal"]
    avg_signal = (tx_signal + comms_signal) / 2.0

    # Key deterministic evidence flags
    det_flags = []
    if state["geo_travel_anomaly"]:
        det_flags.append("impossible travel detected")
    if state["balance_integrity_flag"]:
        det_flags.append("balance integrity violated")
    if state["iban_risk_tier"] in ("high", "medium"):
        det_flags.append(f"IBAN risk tier: {state['iban_risk_tier']}")
    if state["velocity_score"] > 5:
        det_flags.append(f"high velocity: {state['velocity_score']:.0f} txs in window")
    if state["amount_zscore"] > 3.0:
        det_flags.append(f"extreme amount z-score: {state['amount_zscore']:.1f}")

    det_str = "; ".join(det_flags) if det_flags else "none"
    cross_flags = "; ".join(state["cross_source_flags"][:3]) or "none"
    comms_mismatches = "; ".join(state["cross_reference_mismatches"][:3]) or "none"

    # REFACTOR: include combined_risk_score so supervisor can anchor to the
    # calibrated deterministic signal and apply per-level thresholds from config
    return (
        f"Transaction ID: {state['transaction_id']}\n\n"
        f"Specialist signals:\n"
        f"  transaction_fraud_signal: {tx_signal:.3f} "
        f"  (pattern: {state['transaction_pattern_label']})\n"
        f"  transaction_reasoning: {state['transaction_reasoning']}\n\n"
        f"  comms_fraud_signal: {comms_signal:.3f}\n"
        f"  comms_reasoning: {state['comms_reasoning']}\n"
        f"  comms_flagged_phrases: {state['flagged_phrases'][:5]}\n"
        f"  comms_mismatches: {comms_mismatches}\n\n"
        f"Combined average signal: {avg_signal:.3f}\n"
        f"Deterministic combined_risk_score: {state['combined_risk_score']:.4f}\n\n"
        f"Deterministic evidence flags: {det_str}\n"
        f"Cross-source flags (featurizer): {cross_flags}\n\n"
        f"Now follow STATE 1 → STATE 4 and return the final verdict JSON."
    )


def _parse_json(content: str) -> Dict[str, Any]:
    """Extract and validate the supervisor's JSON output."""
    try:
        data = json.loads(content.strip())
        if "verdict" in data:
            return data
    except json.JSONDecodeError:
        pass

    matches = _JSON_RE.findall(content)
    for m in reversed(matches):
        try:
            data = json.loads(m)
            if "verdict" in data:
                return data
        except json.JSONDecodeError:
            continue

    raise ValueError(f"No valid supervisor JSON found in:\n{content[:500]}")


def run_supervisor_agent(
    state: FraudState,
    llm: ChatOpenAI = None,
    langchain_cfg: Dict = None,
) -> FraudState:
    """Run the Supervisor Agent and update FraudState with the final verdict.

    Args:
        state: FraudState with all specialist agent fields populated.
        llm: Optional pre-built LLM (used for testing with mocks).
        langchain_cfg: Optional LangChain config dict (for Langfuse session ID).

    Returns:
        Updated FraudState with verdict, confidence, primary_evidence,
        and explanation populated.
    """
    if llm is None:
        llm = _build_llm()

    user_msg = _format_user_message(state)
    messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT), HumanMessage(content=user_msg)]

    response = llm.invoke(messages)
    final_content = response.content

    try:
        parsed = _parse_json(final_content)
    except ValueError:
        correction = (
            "Reply with ONLY a JSON object — no markdown, no explanation:\n"
            '{"verdict": "<FRAUD|REVIEW|CLEAN>", "confidence": <0.0-1.0>, '
            '"primary_evidence": "<one line>", "explanation": "<one sentence>"}'
        )
        messages.append(response)
        messages.append(HumanMessage(content=correction))
        retry_response = llm.invoke(messages)
        final_content = retry_response.content
        parsed = _parse_json(final_content)

    verdict = str(parsed.get("verdict", "REVIEW")).upper().strip()
    if verdict not in _VALID_VERDICTS:
        # Default to REVIEW rather than CLEAN — err on the side of caution
        verdict = "REVIEW"

    confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
    primary_evidence = str(parsed.get("primary_evidence", ""))
    explanation = str(parsed.get("explanation", ""))

    return {
        **state,
        "verdict": verdict,
        "confidence": confidence,
        "primary_evidence": primary_evidence,
        "explanation": explanation,
    }

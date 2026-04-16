"""Node 2 — Transaction Reasoning Agent.

Uses create_react_agent (LangGraph prebuilt) with an SCoT system prompt.
Accepts a populated FraudState (all deterministic features already computed by
the featurizer) and returns it with transaction_fraud_signal,
transaction_pattern_label, and transaction_reasoning filled in.

# REFACTOR: removed redundant tool calls (check_transaction_velocity,
# compute_balance_integrity, score_amount_anomaly) — these are already computed
# by the featurizer and present in FraudState. Passing them as agent tools caused
# 1–3 extra LLM turns per transaction with no accuracy benefit.
# REFACTOR: uses FAST_MODEL env var (default openai/gpt-4o-mini) — transaction
# reasoning is structured feature synthesis that does not need a strong model.
"""

import json
import os
import re
from typing import Any, Dict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from prompts.transaction_agent_prompt import TRANSACTION_AGENT_SYSTEM_PROMPT
from state import FraudState

_VALID_PATTERNS = {
    "velocity_fraud", "account_takeover", "mule_activity",
    "identity_mismatch", "synthetic_identity", "unclear",
}

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _build_llm() -> ChatOpenAI:
    # REFACTOR: FAST_MODEL for transaction agent — structured feature reasoning
    # is rule-like and does not require the stronger model used by comms + supervisor
    return ChatOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        model=os.getenv("FAST_MODEL", "openai/gpt-4o-mini"),
        temperature=0.0,
        max_tokens=2048,
    )


def _format_user_message(state: FraudState) -> str:
    """Build a compact user message from the deterministic features in FraudState."""
    tx = state["raw_transaction"]
    user = state["raw_user"]
    return (
        f"Transaction ID: {state['transaction_id']}\n"
        f"Type: {tx.get('transaction_type', 'unknown')} | "
        f"Amount: {tx.get('amount', 0):.2f} | "
        f"Timestamp: {tx.get('timestamp', 'unknown')}\n\n"
        f"User profile: job={user.get('job', 'unknown')}, "
        f"salary={user.get('salary', 'unknown')}, "
        f"birth_year={user.get('birth_year', 'unknown')}\n\n"
        f"Deterministic features (pre-computed — do NOT recalculate):\n"
        f"  velocity_score: {state['velocity_score']}\n"
        f"  amount_zscore: {state['amount_zscore']}\n"
        f"  balance_integrity_flag: {state['balance_integrity_flag']}\n"
        f"  iban_risk_tier: {state['iban_risk_tier']}\n"
        f"  geo_travel_anomaly: {state['geo_travel_anomaly']}\n"
        f"  demographic_deviation_pct: {state['demographic_deviation_pct']}\n"
        f"  drift_psi: {state['drift_psi']}\n"
        f"  combined_risk_score: {state['combined_risk_score']:.4f}\n"
        f"  cross_source_flags: {state['cross_source_flags']}\n\n"
        f"Now follow STATE 1 → STATE 4 and return the JSON output."
    )


def _parse_json(content: str) -> Dict[str, Any]:
    """Extract and validate the JSON output from the agent's final message."""
    try:
        data = json.loads(content.strip())
        if "transaction_fraud_signal" in data:
            return data
    except json.JSONDecodeError:
        pass

    matches = _JSON_RE.findall(content)
    for m in reversed(matches):
        try:
            data = json.loads(m)
            if "transaction_fraud_signal" in data:
                return data
        except json.JSONDecodeError:
            continue

    raise ValueError(f"No valid transaction agent JSON found in:\n{content[:500]}")


def run_transaction_agent(
    state: FraudState,
    llm: ChatOpenAI = None,
    langchain_cfg: Dict = None,
) -> FraudState:
    """Run the Transaction Reasoning Agent and update FraudState.

    Args:
        state: FraudState with all deterministic fields populated.
        llm: Optional pre-built LLM (used for testing with mocks).
        langchain_cfg: Optional LangChain config dict (for Langfuse session ID).

    Returns:
        Updated FraudState with transaction_fraud_signal, transaction_pattern_label,
        and transaction_reasoning populated.
    """
    if llm is None:
        llm = _build_llm()

    # REFACTOR: no tools passed — all features already in FraudState from featurizer.
    # Passing tools caused the agent to re-call them, wasting tokens and adding latency.
    agent = create_react_agent(llm, tools=[], prompt=TRANSACTION_AGENT_SYSTEM_PROMPT)

    user_msg = _format_user_message(state)
    cfg = langchain_cfg or {}

    result = agent.invoke({"messages": [HumanMessage(content=user_msg)]}, config=cfg)

    final_content = ""
    for msg in reversed(result.get("messages", [])):
        if hasattr(msg, "content") and msg.content:
            final_content = msg.content
            break

    try:
        parsed = _parse_json(final_content)
    except ValueError:
        correction = (
            "Your previous response did not contain valid JSON. "
            "Reply with ONLY a JSON object matching this exact schema — "
            "no markdown, no explanation:\n"
            '{"transaction_fraud_signal": <0.0-1.0>, '
            '"pattern_label": "<label>", "reasoning": "<one sentence>"}'
        )
        retry_result = agent.invoke(
            {"messages": [
                HumanMessage(content=user_msg),
                HumanMessage(content=correction),
            ]},
            config=cfg,
        )
        final_content = ""
        for msg in reversed(retry_result.get("messages", [])):
            if hasattr(msg, "content") and msg.content:
                final_content = msg.content
                break
        parsed = _parse_json(final_content)

    signal = float(parsed.get("transaction_fraud_signal", 0.0))
    label = str(parsed.get("pattern_label", "unclear"))
    reasoning = str(parsed.get("reasoning", ""))

    signal = max(0.0, min(1.0, signal))
    if label not in _VALID_PATTERNS:
        label = "unclear"

    return {
        **state,
        "transaction_fraud_signal": signal,
        "transaction_pattern_label": label,
        "transaction_reasoning": reasoning,
    }

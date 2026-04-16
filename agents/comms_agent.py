"""Node 3 — Communications Reasoning Agent.

Uses create_react_agent (LangGraph prebuilt) with the communications-domain tools
and an SCoT system prompt. Accepts a populated FraudState and returns it with
comms_fraud_signal, flagged_phrases, cross_reference_mismatches, and comms_reasoning
filled in.

# REFACTOR: uses STRONG_MODEL env var (default openai/gpt-4o) — free-text tone,
# intent, and deception analysis requires stronger reasoning than structured features.
"""

import json
import os
import re
from typing import Any, Dict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from prompts.comms_agent_prompt import COMMS_AGENT_SYSTEM_PROMPT
from state import FraudState
from tools.comms_tools import extract_comms_entities, find_amount_iban_mismatch

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# Token budget: truncate SMS/mail text to stay within context limits
_MAX_COMMS_CHARS = 2000


def _build_llm() -> ChatOpenAI:
    """Instantiate the LLM from environment variables."""
    # REFACTOR: STRONG_MODEL for comms agent — nuanced language understanding
    return ChatOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        model=os.getenv("STRONG_MODEL", "openai/gpt-4o"),
        temperature=0.0,
        max_tokens=2048,
    )


def _format_user_message(state: FraudState) -> str:
    """Build the user message for the comms agent from FraudState."""
    tx = state["raw_transaction"]
    entities = state["extracted_entities"]

    # Compact representation of entities
    iban_str = ", ".join(entities.get("ibans", [])[:5]) or "none found"
    urgency_str = ", ".join(entities.get("urgency_phrases", [])[:5]) or "none found"
    url_str = ", ".join(entities.get("urls", [])[:3]) or "none found"
    flags_str = "; ".join(state["cross_source_flags"][:5]) or "none"

    return (
        f"Transaction ID: {state['transaction_id']}\n"
        f"Type: {tx.get('transaction_type', 'unknown')} | "
        f"Amount: {tx.get('amount', 0):.2f}\n"
        f"Sender IBAN: {tx.get('sender_iban', 'N/A')} | "
        f"Recipient IBAN: {tx.get('recipient_iban', 'N/A')}\n\n"
        f"Extracted communication entities:\n"
        f"  IBANs in comms: {iban_str}\n"
        f"  Urgency phrases: {urgency_str}\n"
        f"  URLs: {url_str}\n"
        f"  Amounts in comms: {', '.join(entities.get('amounts', [])[:3]) or 'none'}\n\n"
        f"Cross-source flags already detected: {flags_str}\n\n"
        f"Now follow STATE 1 → STATE 4 and return the JSON output."
    )


def _parse_json(content: str) -> Dict[str, Any]:
    """Extract and validate the JSON output from the agent's final message."""
    try:
        data = json.loads(content.strip())
        if "comms_fraud_signal" in data:
            return data
    except json.JSONDecodeError:
        pass

    matches = _JSON_RE.findall(content)
    for m in reversed(matches):
        try:
            data = json.loads(m)
            if "comms_fraud_signal" in data:
                return data
        except json.JSONDecodeError:
            continue

    raise ValueError(f"No valid comms agent JSON found in:\n{content[:500]}")


def run_comms_agent(
    state: FraudState,
    llm: ChatOpenAI = None,
    langchain_cfg: Dict = None,
) -> FraudState:
    """Run the Communications Reasoning Agent and update FraudState.

    Args:
        state: FraudState with all deterministic and entity fields populated.
        llm: Optional pre-built LLM (used for testing with mocks).
        langchain_cfg: Optional LangChain config dict (for Langfuse session ID).

    Returns:
        Updated FraudState with comms_fraud_signal, flagged_phrases,
        cross_reference_mismatches, and comms_reasoning populated.
    """
    if llm is None:
        llm = _build_llm()

    tools = [extract_comms_entities, find_amount_iban_mismatch]
    agent = create_react_agent(llm, tools=tools, prompt=COMMS_AGENT_SYSTEM_PROMPT)

    user_msg = _format_user_message(state)
    cfg = langchain_cfg or {}

    result = agent.invoke({"messages": [HumanMessage(content=user_msg)]}, config=cfg)

    final_content = ""
    for msg in reversed(result.get("messages", [])):
        if hasattr(msg, "content") and msg.content:
            final_content = msg.content
            break

    # Parse JSON; retry once on failure
    try:
        parsed = _parse_json(final_content)
    except ValueError:
        correction = (
            "Your previous response did not contain valid JSON. "
            "Reply with ONLY a JSON object matching this exact schema:\n"
            '{"comms_fraud_signal": <0.0-1.0>, '
            '"flagged_phrases": [...], '
            '"cross_reference_mismatches": [...], '
            '"reasoning": "<one sentence>"}'
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

    signal = max(0.0, min(1.0, float(parsed.get("comms_fraud_signal", 0.0))))
    flagged = [str(p) for p in parsed.get("flagged_phrases", []) if p]
    mismatches = [str(m) for m in parsed.get("cross_reference_mismatches", []) if m]
    reasoning = str(parsed.get("reasoning", ""))

    return {
        **state,
        "comms_fraud_signal": signal,
        "flagged_phrases": flagged,
        "cross_reference_mismatches": mismatches,
        "comms_reasoning": reasoning,
    }

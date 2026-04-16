"""LangGraph computation graph for the Reply Mirror fraud detection pipeline.

Topology (per design brief):
  Node 1 (featurizer)  →  route_after_featurizer
                             ├── (combined_risk_score >= SHORTCIRCUIT_THRESHOLD) → supervisor_shortcircuit → END
                             └── (below threshold) → dispatch_node → [tx_agent ‖ comms_agent] → supervisor → END

# REFACTOR: added conditional short-circuit edge after featurizer.
# When combined_risk_score >= SHORTCIRCUIT_THRESHOLD (default 0.90), both specialist
# agents are skipped and the supervisor runs directly on deterministic evidence.
# Expose threshold via env var so it can be tuned at competition time without code changes.
# Reference: Langgraph_AML_Detection uses conditional routing to skip EDD for low-risk txs.

The two specialist agents run in parallel using LangGraph's Send API (fan-out) in
the normal (non-short-circuit) path.
"""

import os
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langgraph.types import Send
from langgraph.graph import END, START, StateGraph

from agents.comms_agent import run_comms_agent
from agents.supervisor_agent import run_supervisor_agent
from agents.transaction_agent import run_transaction_agent
from featurizer import make_featurizer_node
from state import FraudState, default_state

# REFACTOR: SHORTCIRCUIT_THRESHOLD — if combined_risk_score from featurizer exceeds
# this value, skip both specialist agents and route directly to supervisor.
# Default 0.90 means two strong independent deterministic signals trigger short-circuit.
_SHORTCIRCUIT_THRESHOLD = float(os.getenv("SHORTCIRCUIT_THRESHOLD", "0.90"))


# ── Short-circuit routing ─────────────────────────────────────────────────────

def _route_after_featurizer(state: FraudState):
    """Conditional edge: short-circuit to supervisor or fan-out to both specialists."""
    if state.get("combined_risk_score", 0.0) >= _SHORTCIRCUIT_THRESHOLD:
        return "supervisor_shortcircuit"
    return [
        Send("transaction_agent", state),
        Send("comms_agent", state),
    ]


def supervisor_shortcircuit_node(state: FraudState, llm=None, cfg=None) -> FraudState:
    """Supervisor runs directly on deterministic evidence when short-circuit triggers.

    # REFACTOR: sets placeholder agent outputs so the supervisor prompt is coherent
    # and the field contract of FraudState is satisfied, then calls the supervisor with
    # a high-confidence prior derived from the combined_risk_score.
    """
    merged = {
        **state,
        "transaction_fraud_signal": state["combined_risk_score"],
        "transaction_pattern_label": "account_takeover",
        "transaction_reasoning": (
            f"Short-circuit: combined_risk_score={state['combined_risk_score']:.4f} "
            f">= threshold={_SHORTCIRCUIT_THRESHOLD}. "
            f"geo_travel_anomaly={state['geo_travel_anomaly']}, "
            f"balance_integrity_flag={state['balance_integrity_flag']}."
        ),
        "comms_fraud_signal": state["combined_risk_score"],
        "comms_reasoning": (
            f"Short-circuit: high combined_risk_score bypassed comms agent. "
            f"cross_source_flags={state['cross_source_flags'][:3]}."
        ),
        "flagged_phrases": [f for f in state["cross_source_flags"][:5] if f],
        "cross_reference_mismatches": [],
    }
    return run_supervisor_agent(merged, llm=llm, langchain_cfg=cfg)



def transaction_agent_node(state: FraudState, llm=None, cfg=None) -> Dict[str, Any]:
    """Wrapper: run transaction agent, return only its output fields."""
    updated = run_transaction_agent(state, llm=llm, langchain_cfg=cfg)
    return {
        "transaction_fraud_signal": updated["transaction_fraud_signal"],
        "transaction_pattern_label": updated["transaction_pattern_label"],
        "transaction_reasoning": updated["transaction_reasoning"],
    }


def comms_agent_node(state: FraudState, llm=None, cfg=None) -> Dict[str, Any]:
    """Wrapper: run comms agent, return only its output fields."""
    updated = run_comms_agent(state, llm=llm, langchain_cfg=cfg)
    return {
        "comms_fraud_signal": updated["comms_fraud_signal"],
        "flagged_phrases": updated["flagged_phrases"],
        "cross_reference_mismatches": updated["cross_reference_mismatches"],
        "comms_reasoning": updated["comms_reasoning"],
    }


def merge_and_supervise(state: FraudState, llm=None, cfg=None) -> FraudState:
    """Fan-in: both specialist fields are already merged by LangGraph; run supervisor."""
    return run_supervisor_agent(state, llm=llm, langchain_cfg=cfg)


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(dataset: Dict[str, Any], llm: ChatOpenAI = None) -> StateGraph:
    """Assemble and compile the full LangGraph computation graph.

    Args:
        dataset: Dict from data_loader.load_dataset().
        llm: Optional shared LLM instance (useful for testing with mocks).

    Returns:
        A compiled LangGraph graph ready for .invoke().
    """
    featurizer_node = make_featurizer_node(dataset)

    def _tx_node(state: FraudState) -> FraudState:
        return transaction_agent_node(state, llm=llm)

    def _comms_node(state: FraudState) -> FraudState:
        return comms_agent_node(state, llm=llm)

    def _supervisor_node(state: FraudState) -> FraudState:
        return merge_and_supervise(state, llm=llm)

    # REFACTOR: short-circuit supervisor node — runs when combined_risk_score is high
    def _shortcircuit_supervisor_node(state: FraudState) -> FraudState:
        return supervisor_shortcircuit_node(state, llm=llm)

    graph = StateGraph(FraudState)

    graph.add_node("featurizer", featurizer_node)
    graph.add_node("transaction_agent", _tx_node)
    graph.add_node("comms_agent", _comms_node)
    graph.add_node("supervisor", _supervisor_node)
    graph.add_node("supervisor_shortcircuit", _shortcircuit_supervisor_node)

    graph.add_edge(START, "featurizer")

    # Conditional edge: returns "supervisor_shortcircuit" string OR list of Send for fan-out
    graph.add_conditional_edges("featurizer", _route_after_featurizer)

    # Fan-in: both specialist nodes → supervisor (normal path)
    graph.add_edge("transaction_agent", "supervisor")
    graph.add_edge("comms_agent", "supervisor")

    # Both supervisor paths → END
    graph.add_edge("supervisor", END)
    graph.add_edge("supervisor_shortcircuit", END)

    return graph.compile()


# ── Output writer ─────────────────────────────────────────────────────────────

def write_output(results: List[FraudState], output_path: str) -> int:
    """Write flagged transaction IDs to an ASCII output file (one ID per line).

    Both FRAUD and REVIEW verdicts are written, as per engineering rule 8.

    Args:
        results: List of fully processed FraudState dicts.
        output_path: Path for the output ASCII file.

    Returns:
        Number of transaction IDs written.
    """
    flagged = sorted(
        r["transaction_id"]
        for r in results
        if r.get("verdict") in ("FRAUD", "REVIEW")
    )
    with open(output_path, "w", encoding="ascii") as f:
        f.write("\n".join(flagged))
        if flagged:
            f.write("\n")

    return len(flagged)

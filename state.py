"""Shared state schema for the Reply Mirror fraud detection graph."""

from typing import Any
from typing_extensions import TypedDict


class FraudState(TypedDict):
    """Single-transaction state passed through every node of the LangGraph graph.

    Fields are grouped into four layers: identity, deterministic features computed
    by the featurizer node, outputs from each specialist agent, and the final
    supervisor verdict. Every field must be present; the featurizer initialises
    all downstream fields to safe defaults before the agents run.
    """

    # ── Identity ─────────────────────────────────────────────────────────────
    transaction_id: str
    user_id: str            # sender_id from transactions.csv
    raw_transaction: dict   # original CSV row as a dict (passed to agents as context)
    raw_user: dict          # matching users.json entry (empty dict if not found)

    # ── Deterministic features (Node 1 — Featurizer) ─────────────────────────
    velocity_score: float           # number of txs by this sender in ±24-h window
    amount_zscore: float            # z-score of this amount vs. sender history
    balance_integrity_flag: bool    # True if balance_after is inconsistent with amount
    iban_risk_tier: str             # "low" | "medium" | "high" | "unknown"
    geo_travel_anomaly: bool        # True if GPS pings imply impossible travel
    demographic_deviation_pct: float  # 0–1 atypicality score for this user profile
    drift_psi: float                # population stability index vs. baseline (0 = stable)
    extracted_entities: dict        # {ibans, amounts, urls, urgency_phrases} from comms
    cross_source_flags: list        # mismatches between comms entities and tx record
    # REFACTOR: combined deterministic risk score — weighted sum of all featurizer signals
    # Used by graph.py short-circuit edge and supervisor prompt for calibrated thresholds
    combined_risk_score: float      # 0.0–1.0 pre-computed by featurizer

    # ── Transaction Reasoning Agent output (Node 2) ──────────────────────────
    transaction_fraud_signal: float   # 0.0–1.0
    transaction_pattern_label: str    # velocity_fraud | account_takeover | mule_activity |
                                      # identity_mismatch | synthetic_identity | unclear
    transaction_reasoning: str        # one-sentence rationale from the agent

    # ── Communications Reasoning Agent output (Node 3) ───────────────────────
    comms_fraud_signal: float         # 0.0–1.0
    flagged_phrases: list             # list[str] of suspicious phrases found
    cross_reference_mismatches: list  # list[str] of entity mismatches detected
    comms_reasoning: str              # one-sentence rationale from the agent

    # ── Supervisor output (Node 4) ────────────────────────────────────────────
    verdict: str            # "FRAUD" | "REVIEW" | "CLEAN"
    confidence: float       # 0.0–1.0
    primary_evidence: str   # brief evidence summary driving the verdict
    explanation: str        # one-sentence explanation for judges / audit trail


def default_state(transaction_id: str, user_id: str) -> FraudState:
    """Return a FraudState with all agent/supervisor fields set to safe defaults."""
    return FraudState(
        transaction_id=transaction_id,
        user_id=user_id,
        raw_transaction={},
        raw_user={},
        velocity_score=0.0,
        amount_zscore=0.0,
        balance_integrity_flag=False,
        iban_risk_tier="unknown",
        geo_travel_anomaly=False,
        demographic_deviation_pct=0.0,
        drift_psi=0.0,
        extracted_entities={},
        cross_source_flags=[],
        combined_risk_score=0.0,  # REFACTOR: initialised to 0; featurizer sets this
        transaction_fraud_signal=0.0,
        transaction_pattern_label="pending",
        transaction_reasoning="",
        comms_fraud_signal=0.0,
        flagged_phrases=[],
        cross_reference_mismatches=[],
        comms_reasoning="",
        verdict="pending",
        confidence=0.0,
        primary_evidence="",
        explanation="",
    )

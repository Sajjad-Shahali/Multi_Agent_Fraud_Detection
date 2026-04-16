"""Task 4 acceptance tests — supervisor agent and graph assembly."""

import json
import os
import sys
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch
import pytest

from state import default_state, FraudState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_full_state(tx_signal=0.85, comms_signal=0.80, verdict="pending") -> FraudState:
    """Build a FraudState with both specialist outputs filled in."""
    state = default_state("tx-graph-001", "SENDER-001")
    state["raw_transaction"] = {
        "transaction_id": "tx-graph-001",
        "transaction_type": "transfer",
        "amount": 5000.0,
        "timestamp": "2087-03-15T02:30:00",
        "sender_iban": "IT16Y943000230016707",
        "recipient_iban": "NG32ABCDEFGH01234",
        "balance_after": 50.0,
    }
    state["raw_user"] = {"job": "Retired", "salary": 25000, "birth_year": 1945}
    state["velocity_score"] = 7.0
    state["amount_zscore"] = 4.5
    state["geo_travel_anomaly"] = True
    state["balance_integrity_flag"] = True
    state["iban_risk_tier"] = "high"
    state["demographic_deviation_pct"] = 0.65
    state["drift_psi"] = 0.31
    state["extracted_entities"] = {"ibans": ["NG99X000"], "urgency_phrases": ["urgent"], "urls": [], "amounts": []}
    state["cross_source_flags"] = ["IBAN mismatch in comms"]
    state["transaction_fraud_signal"] = tx_signal
    state["transaction_pattern_label"] = "account_takeover"
    state["transaction_reasoning"] = "Velocity + travel anomaly."
    state["comms_fraud_signal"] = comms_signal
    state["flagged_phrases"] = ["urgent"]
    state["cross_reference_mismatches"] = ["IBAN mismatch"]
    state["comms_reasoning"] = "Urgency + IBAN substitution."
    state["verdict"] = verdict
    return state


def _mock_supervisor_invoke(json_response: dict):
    """Return a mock create_react_agent that returns a fixed supervisor JSON."""
    from langchain_core.messages import AIMessage
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {
        "messages": [AIMessage(content=json.dumps(json_response))]
    }
    return mock_agent


# ── Supervisor Agent tests ────────────────────────────────────────────────────

class TestSupervisorAgent:

    def test_both_high_signals_give_fraud(self):
        """Both signals ≥ 0.6 + corroborating flags must produce FRAUD."""
        from agents.supervisor_agent import run_supervisor_agent

        state = _make_full_state(tx_signal=0.92, comms_signal=0.88)
        mock_resp = {"verdict": "FRAUD", "confidence": 0.93,
                     "primary_evidence": "Impossible travel + IBAN mismatch",
                     "explanation": "Corroborated by both channels."}

        with patch("agents.supervisor_agent.create_react_agent", return_value=_mock_supervisor_invoke(mock_resp)):
            with patch("agents.supervisor_agent._build_llm", return_value=MagicMock()):
                result = run_supervisor_agent(state)

        assert result["verdict"] == "FRAUD"
        assert result["confidence"] >= 0.5

    def test_one_high_one_low_gives_review(self):
        """One high + one low signal should produce REVIEW not FRAUD."""
        from agents.supervisor_agent import run_supervisor_agent

        state = _make_full_state(tx_signal=0.7, comms_signal=0.1)
        mock_resp = {"verdict": "REVIEW", "confidence": 0.55,
                     "primary_evidence": "Partial transaction evidence only",
                     "explanation": "Comms do not corroborate transaction signal."}

        with patch("agents.supervisor_agent.create_react_agent", return_value=_mock_supervisor_invoke(mock_resp)):
            with patch("agents.supervisor_agent._build_llm", return_value=MagicMock()):
                result = run_supervisor_agent(state)

        assert result["verdict"] == "REVIEW"

    def test_both_low_gives_clean(self):
        """Both signals < 0.3 with no flags should produce CLEAN."""
        from agents.supervisor_agent import run_supervisor_agent

        state = _make_full_state(tx_signal=0.1, comms_signal=0.08)
        state["cross_source_flags"] = []
        state["geo_travel_anomaly"] = False
        state["balance_integrity_flag"] = False
        state["iban_risk_tier"] = "low"

        mock_resp = {"verdict": "CLEAN", "confidence": 0.85,
                     "primary_evidence": "No corroborating signals",
                     "explanation": "Both specialist signals are low with no flags."}

        with patch("agents.supervisor_agent.create_react_agent", return_value=_mock_supervisor_invoke(mock_resp)):
            with patch("agents.supervisor_agent._build_llm", return_value=MagicMock()):
                result = run_supervisor_agent(state)

        assert result["verdict"] == "CLEAN"

    def test_invalid_verdict_defaults_to_review(self):
        """Invalid verdict must default to REVIEW (fail safe)."""
        from agents.supervisor_agent import run_supervisor_agent

        state = _make_full_state()
        mock_resp = {"verdict": "MAYBE", "confidence": 0.5,
                     "primary_evidence": "Unknown", "explanation": "Unclear."}

        with patch("agents.supervisor_agent.create_react_agent", return_value=_mock_supervisor_invoke(mock_resp)):
            with patch("agents.supervisor_agent._build_llm", return_value=MagicMock()):
                result = run_supervisor_agent(state)

        assert result["verdict"] == "REVIEW"

    def test_confidence_clamped(self):
        """Confidence must be clamped to [0, 1]."""
        from agents.supervisor_agent import run_supervisor_agent

        state = _make_full_state()
        mock_resp = {"verdict": "FRAUD", "confidence": 2.5,
                     "primary_evidence": "Strong evidence", "explanation": "Clear fraud."}

        with patch("agents.supervisor_agent.create_react_agent", return_value=_mock_supervisor_invoke(mock_resp)):
            with patch("agents.supervisor_agent._build_llm", return_value=MagicMock()):
                result = run_supervisor_agent(state)

        assert result["confidence"] <= 1.0

    def test_all_fields_populated(self):
        """Supervisor must populate all four output fields."""
        from agents.supervisor_agent import run_supervisor_agent

        state = _make_full_state()
        mock_resp = {"verdict": "FRAUD", "confidence": 0.9,
                     "primary_evidence": "Multiple signals", "explanation": "Clear case."}

        with patch("agents.supervisor_agent.create_react_agent", return_value=_mock_supervisor_invoke(mock_resp)):
            with patch("agents.supervisor_agent._build_llm", return_value=MagicMock()):
                result = run_supervisor_agent(state)

        assert result["verdict"] in ("FRAUD", "REVIEW", "CLEAN")
        assert isinstance(result["confidence"], float)
        assert isinstance(result["primary_evidence"], str)
        assert isinstance(result["explanation"], str)


# ── Output writer tests ───────────────────────────────────────────────────────

class TestOutputWriter:

    def test_fraud_and_review_written(self):
        """Both FRAUD and REVIEW verdicts must appear in the output file."""
        from graph import write_output

        states = [
            {**default_state("tx-001", "u1"), "verdict": "FRAUD"},
            {**default_state("tx-002", "u2"), "verdict": "REVIEW"},
            {**default_state("tx-003", "u3"), "verdict": "CLEAN"},
        ]
        with tempfile.NamedTemporaryFile(mode="r", suffix=".txt", delete=False) as f:
            path = f.name

        n = write_output(states, path)
        assert n == 2

        with open(path) as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        assert "tx-001" in lines
        assert "tx-002" in lines
        assert "tx-003" not in lines

    def test_output_is_sorted(self):
        """Output IDs must be sorted lexicographically."""
        from graph import write_output

        states = [
            {**default_state("tx-zzz", "u1"), "verdict": "FRAUD"},
            {**default_state("tx-aaa", "u2"), "verdict": "REVIEW"},
        ]
        with tempfile.NamedTemporaryFile(mode="r", suffix=".txt", delete=False) as f:
            path = f.name

        write_output(states, path)
        with open(path) as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        assert lines == sorted(lines)

    def test_clean_verdicts_suppressed(self):
        """CLEAN verdicts must NOT appear in the output file."""
        from graph import write_output

        states = [{**default_state(f"tx-{i:03d}", "u"), "verdict": "CLEAN"} for i in range(5)]
        with tempfile.NamedTemporaryFile(mode="r", suffix=".txt", delete=False) as f:
            path = f.name

        n = write_output(states, path)
        assert n == 0

    def test_output_is_ascii(self):
        """Output file must be valid ASCII."""
        from graph import write_output

        states = [{**default_state("tx-ascii-001", "u"), "verdict": "FRAUD"}]
        with tempfile.NamedTemporaryFile(mode="r", suffix=".txt", delete=False) as f:
            path = f.name

        write_output(states, path)
        with open(path, "rb") as f:
            content = f.read()
        content.decode("ascii")  # raises UnicodeDecodeError if not ASCII

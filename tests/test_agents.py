"""Task 3 acceptance tests — specialist agents with mocked OpenRouter responses."""

import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch
import pytest

from state import default_state, FraudState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_state(
    tx_signal=0.0,
    comms_signal=0.0,
    velocity=1.0,
    zscore=0.5,
    geo=False,
    balance_flag=False,
    iban_tier="low",
    entities=None,
    flags=None,
) -> FraudState:
    """Build a FraudState fixture with configurable deterministic fields."""
    state = default_state("tx-test-001", "SENDER-001")
    state["raw_transaction"] = {
        "transaction_id": "tx-test-001",
        "transaction_type": "transfer",
        "amount": 1500.0,
        "timestamp": "2087-03-15T02:30:00",
        "sender_iban": "IT16Y9430002300167070752952",
        "recipient_iban": "DE62U9486200442637789135342",
        "balance_after": 500.0,
    }
    state["raw_user"] = {"job": "Software Engineer", "salary": 60000, "birth_year": 1990}
    state["velocity_score"] = velocity
    state["amount_zscore"] = zscore
    state["geo_travel_anomaly"] = geo
    state["balance_integrity_flag"] = balance_flag
    state["iban_risk_tier"] = iban_tier
    state["demographic_deviation_pct"] = 0.1
    state["drift_psi"] = 0.05
    state["extracted_entities"] = entities or {
        "ibans": [], "amounts": [], "urls": [], "urgency_phrases": []
    }
    state["cross_source_flags"] = flags or []
    state["transaction_fraud_signal"] = tx_signal
    state["comms_fraud_signal"] = comms_signal
    return state


def _mock_agent_invoke(json_response: dict):
    """Return a mock create_react_agent that always returns a fixed JSON response."""
    from langchain_core.messages import AIMessage

    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {
        "messages": [AIMessage(content=json.dumps(json_response))]
    }
    return mock_agent


# ── Transaction Agent tests ───────────────────────────────────────────────────

class TestTransactionAgent:

    def test_fraud_signal_populated(self):
        """run_transaction_agent must set transaction_fraud_signal in state."""
        from agents.transaction_agent import run_transaction_agent

        state = _make_state(velocity=8.0, zscore=4.2, geo=True, balance_flag=True)
        mock_response = {
            "transaction_fraud_signal": 0.94,
            "pattern_label": "account_takeover",
            "reasoning": "Multiple corroborating signals of account takeover.",
        }
        with patch("agents.transaction_agent.create_react_agent", return_value=_mock_agent_invoke(mock_response)):
            with patch("agents.transaction_agent._build_llm", return_value=MagicMock()):
                result = run_transaction_agent(state)

        assert result["transaction_fraud_signal"] == pytest.approx(0.94)
        assert result["transaction_pattern_label"] == "account_takeover"
        assert len(result["transaction_reasoning"]) > 0

    def test_clean_signal_preserved(self):
        """Agent must return low signal for a clean-looking transaction."""
        from agents.transaction_agent import run_transaction_agent

        state = _make_state(velocity=2.0, zscore=0.3)
        mock_response = {
            "transaction_fraud_signal": 0.12,
            "pattern_label": "unclear",
            "reasoning": "No corroborating signals — transaction appears legitimate.",
        }
        with patch("agents.transaction_agent.create_react_agent", return_value=_mock_agent_invoke(mock_response)):
            with patch("agents.transaction_agent._build_llm", return_value=MagicMock()):
                result = run_transaction_agent(state)

        assert result["transaction_fraud_signal"] < 0.3
        assert result["transaction_pattern_label"] == "unclear"

    def test_signal_clamped_to_0_1(self):
        """Signal must be clamped to [0, 1] even if the LLM returns out-of-range value."""
        from agents.transaction_agent import run_transaction_agent

        state = _make_state()
        mock_response = {
            "transaction_fraud_signal": 1.5,  # out of range
            "pattern_label": "velocity_fraud",
            "reasoning": "High velocity.",
        }
        with patch("agents.transaction_agent.create_react_agent", return_value=_mock_agent_invoke(mock_response)):
            with patch("agents.transaction_agent._build_llm", return_value=MagicMock()):
                result = run_transaction_agent(state)

        assert result["transaction_fraud_signal"] <= 1.0

    def test_invalid_pattern_label_falls_back_to_unclear(self):
        """Unknown pattern labels must be normalised to 'unclear'."""
        from agents.transaction_agent import run_transaction_agent

        state = _make_state()
        mock_response = {
            "transaction_fraud_signal": 0.5,
            "pattern_label": "some_unknown_pattern",
            "reasoning": "Unusual.",
        }
        with patch("agents.transaction_agent.create_react_agent", return_value=_mock_agent_invoke(mock_response)):
            with patch("agents.transaction_agent._build_llm", return_value=MagicMock()):
                result = run_transaction_agent(state)

        assert result["transaction_pattern_label"] == "unclear"

    def test_other_state_fields_unchanged(self):
        """run_transaction_agent must not modify deterministic or comms fields."""
        from agents.transaction_agent import run_transaction_agent

        state = _make_state(velocity=3.0, zscore=1.5)
        mock_response = {
            "transaction_fraud_signal": 0.4,
            "pattern_label": "unclear",
            "reasoning": "Moderate.",
        }
        with patch("agents.transaction_agent.create_react_agent", return_value=_mock_agent_invoke(mock_response)):
            with patch("agents.transaction_agent._build_llm", return_value=MagicMock()):
                result = run_transaction_agent(state)

        assert result["velocity_score"] == 3.0
        assert result["amount_zscore"] == 1.5
        assert result["comms_fraud_signal"] == 0.0  # untouched


# ── Communications Agent tests ────────────────────────────────────────────────

class TestCommsAgent:

    def test_comms_signal_populated(self):
        """run_comms_agent must set comms_fraud_signal in state."""
        from agents.comms_agent import run_comms_agent

        state = _make_state(
            entities={"ibans": ["IT99X0000000000"], "urgency_phrases": ["urgent", "immediately"], "urls": [], "amounts": []},
            flags=["IBAN in comms (IT99X000...) does not match transaction IBANs"],
        )
        mock_response = {
            "comms_fraud_signal": 0.91,
            "flagged_phrases": ["urgent", "immediately"],
            "cross_reference_mismatches": ["IBAN mismatch"],
            "reasoning": "IBAN substitution + urgency language.",
        }
        with patch("agents.comms_agent.create_react_agent", return_value=_mock_agent_invoke(mock_response)):
            with patch("agents.comms_agent._build_llm", return_value=MagicMock()):
                result = run_comms_agent(state)

        assert result["comms_fraud_signal"] == pytest.approx(0.91)
        assert "urgent" in result["flagged_phrases"]
        assert len(result["cross_reference_mismatches"]) > 0

    def test_clean_comms_returns_low_signal(self):
        """Legitimate communication must yield a low comms_fraud_signal."""
        from agents.comms_agent import run_comms_agent

        state = _make_state(entities={"ibans": [], "urgency_phrases": [], "urls": [], "amounts": []})
        mock_response = {
            "comms_fraud_signal": 0.04,
            "flagged_phrases": [],
            "cross_reference_mismatches": [],
            "reasoning": "Legitimate bank statement notification.",
        }
        with patch("agents.comms_agent.create_react_agent", return_value=_mock_agent_invoke(mock_response)):
            with patch("agents.comms_agent._build_llm", return_value=MagicMock()):
                result = run_comms_agent(state)

        assert result["comms_fraud_signal"] < 0.3
        assert result["flagged_phrases"] == []

    def test_comms_signal_clamped(self):
        """comms_fraud_signal must be clamped to [0, 1]."""
        from agents.comms_agent import run_comms_agent

        state = _make_state()
        mock_response = {
            "comms_fraud_signal": -0.5,  # invalid
            "flagged_phrases": [],
            "cross_reference_mismatches": [],
            "reasoning": "Error case.",
        }
        with patch("agents.comms_agent.create_react_agent", return_value=_mock_agent_invoke(mock_response)):
            with patch("agents.comms_agent._build_llm", return_value=MagicMock()):
                result = run_comms_agent(state)

        assert result["comms_fraud_signal"] >= 0.0

    def test_transaction_fields_unchanged(self):
        """run_comms_agent must not modify transaction agent or featurizer fields."""
        from agents.comms_agent import run_comms_agent

        state = _make_state(velocity=5.0, tx_signal=0.75)
        mock_response = {
            "comms_fraud_signal": 0.5,
            "flagged_phrases": [],
            "cross_reference_mismatches": [],
            "reasoning": "Moderate.",
        }
        with patch("agents.comms_agent.create_react_agent", return_value=_mock_agent_invoke(mock_response)):
            with patch("agents.comms_agent._build_llm", return_value=MagicMock()):
                result = run_comms_agent(state)

        assert result["velocity_score"] == 5.0
        assert result["transaction_fraud_signal"] == pytest.approx(0.75)


# ── JSON parsing tests ────────────────────────────────────────────────────────

class TestJsonParsing:

    def test_transaction_parse_from_clean_json(self):
        """_parse_json must handle a clean JSON string."""
        from agents.transaction_agent import _parse_json
        raw = '{"transaction_fraud_signal": 0.7, "pattern_label": "velocity_fraud", "reasoning": "High velocity."}'
        result = _parse_json(raw)
        assert result["transaction_fraud_signal"] == pytest.approx(0.7)

    def test_transaction_parse_from_markdown_wrapped_json(self):
        """_parse_json must extract JSON even when wrapped in markdown code blocks."""
        from agents.transaction_agent import _parse_json
        raw = 'Here is my analysis:\n```json\n{"transaction_fraud_signal": 0.5, "pattern_label": "unclear", "reasoning": "ok"}\n```'
        result = _parse_json(raw)
        assert result["transaction_fraud_signal"] == pytest.approx(0.5)

    def test_comms_parse_raises_on_missing_key(self):
        """_parse_json for comms must raise ValueError when the key is absent."""
        from agents.comms_agent import _parse_json
        with pytest.raises(ValueError):
            _parse_json('{"wrong_key": 0.5}')

    def test_prompt_constants_are_non_empty(self):
        """All three prompt constants must be non-empty strings."""
        from prompts.transaction_agent_prompt import TRANSACTION_AGENT_SYSTEM_PROMPT
        from prompts.comms_agent_prompt import COMMS_AGENT_SYSTEM_PROMPT
        from prompts.supervisor_prompt import SUPERVISOR_SYSTEM_PROMPT

        assert len(TRANSACTION_AGENT_SYSTEM_PROMPT) > 100
        assert len(COMMS_AGENT_SYSTEM_PROMPT) > 100
        assert len(SUPERVISOR_SYSTEM_PROMPT) > 100

    def test_prompt_contains_required_states(self):
        """Prompts must contain all four STATE headings."""
        from prompts.transaction_agent_prompt import TRANSACTION_AGENT_SYSTEM_PROMPT
        from prompts.comms_agent_prompt import COMMS_AGENT_SYSTEM_PROMPT

        for prompt in [TRANSACTION_AGENT_SYSTEM_PROMPT, COMMS_AGENT_SYSTEM_PROMPT]:
            for state_n in ["STATE 1", "STATE 2", "STATE 3", "STATE 4"]:
                assert state_n in prompt, f"'{state_n}' missing from prompt"

"""Task 1 acceptance tests — state schema and data loader sanity checks."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from state import FraudState, default_state
from data_loader import LEVEL_MAP, SPLIT_MAP, resolve_zip_path


# ── FraudState schema checks ──────────────────────────────────────────────────

REQUIRED_KEYS = [
    # identity
    "transaction_id", "user_id", "raw_transaction", "raw_user",
    # deterministic
    "velocity_score", "amount_zscore", "balance_integrity_flag",
    "iban_risk_tier", "geo_travel_anomaly", "demographic_deviation_pct",
    "drift_psi", "extracted_entities", "cross_source_flags",
    "combined_risk_score",
    # transaction agent
    "transaction_fraud_signal", "transaction_pattern_label", "transaction_reasoning",
    # comms agent
    "comms_fraud_signal", "flagged_phrases", "cross_reference_mismatches", "comms_reasoning",
    # supervisor
    "verdict", "confidence", "primary_evidence", "explanation",
]


def test_fraud_state_is_typed_dict():
    """FraudState must be a TypedDict subclass."""
    from typing_extensions import get_type_hints
    hints = get_type_hints(FraudState)
    assert len(hints) > 0, "FraudState has no type hints"


def test_all_required_keys_annotated():
    """Every required key must appear in FraudState annotations."""
    from typing_extensions import get_type_hints
    hints = get_type_hints(FraudState)
    missing = [k for k in REQUIRED_KEYS if k not in hints]
    assert not missing, f"Missing keys in FraudState: {missing}"


def test_default_state_has_all_keys():
    """default_state() must return a dict with every required key."""
    state = default_state("tx-001", "user-001")
    missing = [k for k in REQUIRED_KEYS if k not in state]
    assert not missing, f"default_state() missing keys: {missing}"


def test_default_state_types():
    """Spot-check that default_state() produces correct types for each layer."""
    state = default_state("tx-001", "user-001")

    assert isinstance(state["transaction_id"], str)
    assert isinstance(state["user_id"], str)
    assert isinstance(state["raw_transaction"], dict)
    assert isinstance(state["raw_user"], dict)

    assert isinstance(state["velocity_score"], float)
    assert isinstance(state["amount_zscore"], float)
    assert isinstance(state["balance_integrity_flag"], bool)
    assert isinstance(state["iban_risk_tier"], str)
    assert isinstance(state["geo_travel_anomaly"], bool)
    assert isinstance(state["demographic_deviation_pct"], float)
    assert isinstance(state["drift_psi"], float)
    assert isinstance(state["extracted_entities"], dict)
    assert isinstance(state["cross_source_flags"], list)
    assert isinstance(state["combined_risk_score"], float)

    assert isinstance(state["transaction_fraud_signal"], float)
    assert isinstance(state["transaction_pattern_label"], str)
    assert isinstance(state["transaction_reasoning"], str)

    assert isinstance(state["comms_fraud_signal"], float)
    assert isinstance(state["flagged_phrases"], list)
    assert isinstance(state["cross_reference_mismatches"], list)
    assert isinstance(state["comms_reasoning"], str)

    assert isinstance(state["verdict"], str)
    assert isinstance(state["confidence"], float)
    assert isinstance(state["primary_evidence"], str)
    assert isinstance(state["explanation"], str)


def test_default_state_safe_defaults():
    """Default state must have no pending signals that could leak into output."""
    state = default_state("tx-abc", "sender-xyz")
    assert state["transaction_fraud_signal"] == 0.0
    assert state["comms_fraud_signal"] == 0.0
    assert state["verdict"] == "pending"
    assert state["confidence"] == 0.0


# ── data_loader schema checks (no I/O — just the mapping tables) ──────────────

def test_level_map_has_expected_levels():
    expected = {"brave-new-world", "deus-ex", "the-truman-show"}
    assert expected.issubset(set(LEVEL_MAP.keys()))


def test_split_map_has_train_and_validation():
    assert "train" in SPLIT_MAP
    assert "validation" in SPLIT_MAP


def test_resolve_zip_path_format():
    """resolve_zip_path must produce a path with the correct filename pattern."""
    path = resolve_zip_path("train-validation", "brave-new-world", "train")
    assert "Brave+New+World" in path.name
    assert "train" in path.name
    assert path.suffix == ".zip"


def test_resolve_zip_path_invalid_level():
    with pytest.raises(ValueError, match="Unknown level"):
        resolve_zip_path("train-validation", "nonexistent-level", "train")


def test_resolve_zip_path_invalid_split():
    with pytest.raises(ValueError, match="Unknown split"):
        resolve_zip_path("train-validation", "brave-new-world", "test")

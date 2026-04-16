"""Task 2 acceptance tests — featurizer against real Brave New World training data."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd

from data_loader import load_dataset
from featurizer import featurize_transaction
from state import FraudState

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "train-validation")
LEVEL = "brave-new-world"
SPLIT = "train"


@pytest.fixture(scope="module")
def dataset():
    """Load the Brave New World training dataset once for all tests."""
    return load_dataset(DATA_DIR, LEVEL, SPLIT)


@pytest.fixture(scope="module")
def first_tx(dataset):
    """Return the first transaction row."""
    return dataset["transactions"].iloc[0]


@pytest.fixture(scope="module")
def first_state(first_tx, dataset):
    """Featurize the first transaction."""
    return featurize_transaction(first_tx, dataset)


# ── Schema completeness ───────────────────────────────────────────────────────

def test_featurized_state_has_all_deterministic_keys(first_state):
    """All deterministic FraudState keys must be populated after featurization."""
    required = [
        "transaction_id", "user_id", "raw_transaction", "raw_user",
        "velocity_score", "amount_zscore", "balance_integrity_flag",
        "iban_risk_tier", "geo_travel_anomaly", "demographic_deviation_pct",
        "drift_psi", "extracted_entities", "cross_source_flags",
        "combined_risk_score",
    ]
    missing = [k for k in required if k not in first_state]
    assert not missing, f"Missing keys after featurization: {missing}"


def test_featurized_state_agent_fields_are_defaults(first_state):
    """Agent and supervisor fields must remain at defaults — featurizer must not touch them."""
    assert first_state["transaction_fraud_signal"] == 0.0
    assert first_state["comms_fraud_signal"] == 0.0
    assert first_state["verdict"] == "pending"


# ── Type correctness ──────────────────────────────────────────────────────────

def test_velocity_score_is_float(first_state):
    assert isinstance(first_state["velocity_score"], float)
    assert first_state["velocity_score"] >= 0.0


def test_amount_zscore_is_non_negative_float(first_state):
    assert isinstance(first_state["amount_zscore"], float)
    assert first_state["amount_zscore"] >= 0.0


def test_balance_integrity_flag_is_bool(first_state):
    assert isinstance(first_state["balance_integrity_flag"], bool)


def test_iban_risk_tier_is_valid(first_state):
    assert first_state["iban_risk_tier"] in ("low", "medium", "high", "unknown")


def test_geo_travel_anomaly_is_bool(first_state):
    assert isinstance(first_state["geo_travel_anomaly"], bool)


def test_demographic_deviation_in_range(first_state):
    score = first_state["demographic_deviation_pct"]
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_drift_psi_is_float(first_state):
    assert isinstance(first_state["drift_psi"], float)
    assert first_state["drift_psi"] >= 0.0


def test_extracted_entities_has_expected_keys(first_state):
    entities = first_state["extracted_entities"]
    assert isinstance(entities, dict)
    for key in ("ibans", "amounts", "urls", "urgency_phrases"):
        assert key in entities, f"Missing entity key: {key}"


def test_cross_source_flags_is_list(first_state):
    assert isinstance(first_state["cross_source_flags"], list)


def test_combined_risk_score_in_range(first_state):
    score = first_state["combined_risk_score"]
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# ── Data integrity ────────────────────────────────────────────────────────────

def test_transaction_id_matches_input(first_tx, first_state):
    assert first_state["transaction_id"] == str(first_tx["transaction_id"])


def test_raw_transaction_is_populated(first_state):
    assert len(first_state["raw_transaction"]) > 0
    assert "transaction_id" in first_state["raw_transaction"]


def test_dataset_loads_all_five_sources(dataset):
    assert "transactions" in dataset
    assert "users" in dataset
    assert "locations" in dataset
    assert "sms" in dataset
    assert "mails" in dataset
    assert len(dataset["transactions"]) > 0
    assert len(dataset["users"]) > 0
    assert len(dataset["locations"]) > 0
    assert len(dataset["sms"]) > 0
    assert len(dataset["mails"]) > 0


# ── Tool-level spot checks ────────────────────────────────────────────────────

def test_validate_iban_risk_high_country():
    from tools.transaction_tools import validate_iban_risk
    result = validate_iban_risk.func("NG32ABCDEFGHIJ1234567890")
    assert result["tier"] == "high"
    assert result["country"] == "NG"


def test_validate_iban_risk_low_country():
    from tools.transaction_tools import validate_iban_risk
    result = validate_iban_risk.func("DE89370400440532013000")
    assert result["tier"] == "low"


def test_extract_comms_entities_finds_urgency():
    from tools.comms_tools import extract_comms_entities_direct
    sms = ["URGENT: your account is suspended. Verify your identity immediately."]
    result = extract_comms_entities_direct(sms, [])
    assert len(result["urgency_phrases"]) > 0


def test_extract_comms_entities_finds_iban():
    from tools.comms_tools import extract_comms_entities_direct
    sms = ["Please transfer to IBAN IT16Y9430002300167070752952 now."]
    result = extract_comms_entities_direct(sms, [])
    assert any("IT16" in i for i in result["ibans"])


def test_balance_integrity_flags_drain_pattern():
    from tools.transaction_tools import compute_balance_integrity_direct
    # A sender who had a large balance (est ~5000) and drains it near-zero in one tx
    df = pd.DataFrame([
        {"sender_id": "A", "timestamp": "2087-01-01", "amount": 200.0, "balance_after": 4800.0},
        {"sender_id": "A", "timestamp": "2087-01-02", "amount": 200.0, "balance_after": 4600.0},
        {"sender_id": "A", "timestamp": "2087-01-03", "amount": 4550.0, "balance_after": 50.0},
    ])
    result = compute_balance_integrity_direct(df, "A")
    # Near-zero drain: balance ended at 50 after max_prev_bal was > 1000
    assert result["flag"] is True
    assert result["n_violations"] >= 1


def test_balance_integrity_clean_account():
    from tools.transaction_tools import compute_balance_integrity_direct
    # Normal account with consistent small transactions — should not flag
    df = pd.DataFrame([
        {"sender_id": "B", "timestamp": "2087-01-01", "amount": 100.0, "balance_after": 900.0},
        {"sender_id": "B", "timestamp": "2087-01-02", "amount": 80.0, "balance_after": 820.0},
    ])
    result = compute_balance_integrity_direct(df, "B")
    assert result["flag"] is False

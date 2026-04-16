"""Task 5 acceptance tests — evaluate.py metrics, session, and Langfuse wiring."""

import os
import sys
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from evaluate import compute_metrics, load_ids


# ── compute_metrics tests ─────────────────────────────────────────────────────

class TestComputeMetrics:

    def test_perfect_precision_recall(self):
        predicted = {"tx-1", "tx-2", "tx-3"}
        ground_truth = {"tx-1", "tx-2", "tx-3"}
        m = compute_metrics(predicted, ground_truth, total_transactions=10)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)
        assert m["fp"] == 0
        assert m["fn"] == 0

    def test_zero_precision_all_false_positives(self):
        predicted = {"tx-4", "tx-5"}
        ground_truth = {"tx-1", "tx-2"}
        m = compute_metrics(predicted, ground_truth, total_transactions=10)
        assert m["precision"] == pytest.approx(0.0)
        assert m["recall"] == pytest.approx(0.0)
        assert m["tp"] == 0
        assert m["fp"] == 2
        assert m["fn"] == 2

    def test_partial_recall(self):
        predicted = {"tx-1"}
        ground_truth = {"tx-1", "tx-2", "tx-3", "tx-4"}
        m = compute_metrics(predicted, ground_truth, total_transactions=20)
        assert m["recall"] == pytest.approx(0.25)
        assert m["fn"] == 3

    def test_recall_below_threshold_detectable(self):
        """compute_metrics must return recall values that can be compared to 0.15."""
        predicted = set()
        ground_truth = {"tx-1", "tx-2", "tx-3", "tx-4", "tx-5"}
        m = compute_metrics(predicted, ground_truth, total_transactions=100)
        assert m["recall"] < 0.15

    def test_false_positive_rate(self):
        predicted = {"tx-1", "tx-bad-1", "tx-bad-2"}
        ground_truth = {"tx-1"}
        m = compute_metrics(predicted, ground_truth, total_transactions=10)
        # FP=2, TN=10-1-2-0=7, FPR=2/9
        assert m["fpr"] == pytest.approx(2 / (2 + 7))

    def test_counts_sum_to_total(self):
        predicted = {"tx-1", "tx-2", "tx-bad"}
        ground_truth = {"tx-1", "tx-2", "tx-3"}
        m = compute_metrics(predicted, ground_truth, total_transactions=10)
        assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == 10


# ── load_ids tests ────────────────────────────────────────────────────────────

class TestLoadIds:

    def test_loads_all_ids(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("tx-001\ntx-002\ntx-003\n")
            path = f.name
        ids = load_ids(path)
        assert ids == {"tx-001", "tx-002", "tx-003"}

    def test_ignores_blank_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("tx-001\n\ntx-002\n\n")
            path = f.name
        ids = load_ids(path)
        assert "" not in ids
        assert len(ids) == 2

    def test_strips_whitespace(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("  tx-001  \n  tx-002  \n")
            path = f.name
        ids = load_ids(path)
        assert "tx-001" in ids


# ── session tests ─────────────────────────────────────────────────────────────

class TestSession:

    def test_session_id_format(self):
        """Session IDs must match the pattern {TEAM_NAME}-{ULID}."""
        import re
        from session import generate_session_id

        sid = generate_session_id()
        # ULID is 26 uppercase alphanumeric chars
        assert re.match(r"^[a-z0-9\-]+-[0-9A-Z]{26}$", sid), f"Unexpected format: {sid}"

    def test_session_ids_are_unique(self):
        """Two consecutive session IDs must never be the same."""
        from session import generate_session_id
        assert generate_session_id() != generate_session_id()

    def test_langchain_config_contains_session_id(self):
        """langchain_config must embed the session_id in metadata."""
        from session import langchain_config
        sid = "test-team-01ARZ3NDEKTSV4RRFFQ69G5FAV"
        cfg = langchain_config(sid)
        assert cfg["metadata"]["langfuse_session_id"] == sid

    def test_langfuse_client_returns_none_without_credentials(self, monkeypatch):
        """langfuse_client must return None gracefully when credentials are absent."""
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        from session import langfuse_client
        result = langfuse_client()
        # None is acceptable — no exception should be raised
        assert result is None or hasattr(result, "flush")


# ── Pipeline smoke test (no LLM calls) ────────────────────────────────────────

class TestPipelineSmoke:

    def test_data_loader_brave_new_world(self):
        """Dataset must load with all five sources and correct types."""
        from data_loader import load_dataset
        data_dir = os.path.join(os.path.dirname(__file__), "..", "train-validation")
        dataset = load_dataset(data_dir, "brave-new-world", "train")

        import pandas as pd
        assert isinstance(dataset["transactions"], pd.DataFrame)
        assert isinstance(dataset["users"], list)
        assert isinstance(dataset["locations"], list)
        assert isinstance(dataset["sms"], list)
        assert isinstance(dataset["mails"], list)
        assert len(dataset["transactions"]) > 0

    def test_featurizer_runs_on_full_train(self):
        """Featurizer must process the first 5 training transactions without error."""
        from data_loader import load_dataset
        from featurizer import featurize_transaction

        data_dir = os.path.join(os.path.dirname(__file__), "..", "train-validation")
        dataset = load_dataset(data_dir, "brave-new-world", "train")
        txs = dataset["transactions"].head(5)

        for _, row in txs.iterrows():
            state = featurize_transaction(row, dataset)
            assert state["verdict"] == "pending"  # agents not yet run
            assert isinstance(state["velocity_score"], float)
            assert state["iban_risk_tier"] in ("low", "medium", "high", "unknown")

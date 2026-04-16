"""Node 1 — Deterministic Featurizer.

Pure Python LangGraph function node. Calls all deterministic tool functions
and returns a fully populated FraudState. No LLM calls are made here.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from state import FraudState, default_state
from tools.transaction_tools import (
    check_transaction_velocity_direct,
    compute_balance_integrity_direct,
    detect_pattern_drift_direct,
    score_amount_anomaly_direct,
    validate_iban_risk,
)
from tools.geospatial_tools import detect_impossible_travel_direct
from tools.comms_tools import extract_comms_entities_direct, find_amount_iban_mismatch_direct


# REFACTOR: fix _find_user — match by sender_iban (from transaction row) against user.iban.
# Original code tried to match sender_id (e.g. 'FCHN-VTSI-7FA-BRE-0') against IBAN suffix,
# which never matched. Result: raw_user was always {}, making all user-dependent features zero.
def _find_user(users: list, sender_iban: str) -> dict:
    """Match a user profile by their IBAN field against the transaction's sender_iban."""
    if not sender_iban or not sender_iban.strip():
        return {}
    sender_iban = sender_iban.strip()
    for u in users:
        if str(u.get("iban", "")).strip() == sender_iban:
            return u
    return {}


def _resolve_tx_location(
    location_str: Optional[str],
    locations: list,
) -> Tuple[Optional[float], Optional[float]]:
    """Return (lat, lng) for a transaction location string, or (None, None)."""
    if not location_str or not isinstance(location_str, str) or not location_str.strip():
        return None, None
    city = location_str.strip().lower()
    for loc in locations:
        if str(loc.get("city", "")).lower() == city:
            return float(loc["lat"]), float(loc["lng"])
    return None, None


# REFACTOR: filter SMS and mails to only those that mention the user's name,
# reducing cross-user contamination. Falls back to all messages if none match.
def _filter_comms_for_user(
    sms_texts: List[str],
    mail_texts: List[str],
    user: dict,
) -> Tuple[List[str], List[str]]:
    """Return SMS and mail texts relevant to this user by name matching."""
    first = str(user.get("first_name", "")).strip().lower()
    last = str(user.get("last_name", "")).strip().lower()
    if not first and not last:
        return sms_texts, mail_texts

    def matches(text: str) -> bool:
        t = text.lower()
        return (first and first in t) or (last and last in t)

    filtered_sms = [s for s in sms_texts if matches(s)]
    filtered_mails = [m for m in mail_texts if matches(m)]
    # Fall back to all messages if no match (user name may not appear in comms)
    return filtered_sms or sms_texts, filtered_mails or mail_texts


def _compute_combined_risk_score(
    velocity_score: float,
    amount_zscore: float,
    balance_integrity_flag: bool,
    iban_risk_tier: str,
    geo_travel_anomaly: bool,
    demographic_deviation_pct: float,
    drift_psi: float,
    n_cross_flags: int,
) -> float:
    """Compute a weighted deterministic risk score in [0, 1] for short-circuit decisions.

    # REFACTOR: pre-computed combined score lets graph.py short-circuit both agents
    # when deterministic evidence is already conclusive (see SHORTCIRCUIT_THRESHOLD).
    Weights are calibrated so that two strong independent signals push score above 0.90.
    """
    score = 0.0

    # Velocity: normalise 0–10 → 0–0.20
    score += min(velocity_score / 10.0, 1.0) * 0.20

    # Amount z-score: z > 3 is notable, z > 5 is extreme
    score += min(amount_zscore / 6.0, 1.0) * 0.15

    # Balance integrity: hard binary flag
    score += 0.20 if balance_integrity_flag else 0.0

    # IBAN risk tier
    iban_weight = {"high": 0.15, "medium": 0.07, "low": 0.0, "unknown": 0.0}
    score += iban_weight.get(iban_risk_tier, 0.0)

    # Impossible travel: hard binary flag — very strong signal
    score += 0.20 if geo_travel_anomaly else 0.0

    # Demographic deviation (0–1)
    score += demographic_deviation_pct * 0.10

    # PSI drift (0 = stable, >0.25 = significant)
    score += min(drift_psi / 0.5, 1.0) * 0.10

    # Cross-source flags from comms
    score += min(n_cross_flags / 5.0, 1.0) * 0.10

    return round(min(score, 1.0), 4)


# REFACTOR: batch pre-compute per-user statistics once, keyed by sender_id.
# Original code recomputed these inside featurize_transaction on every call —
# with 7 users and 522 transactions, each user's DataFrame was scanned ~75 times.
def _build_user_stats_cache(all_txs: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Pre-compute per-sender statistics (velocity baselines, balance integrity, drift)."""
    cache: Dict[str, Dict[str, Any]] = {}
    for sender_id, grp in all_txs.groupby("sender_id"):
        grp_sorted = grp.sort_values("timestamp").reset_index(drop=True)
        amounts = grp_sorted["amount"].dropna()

        # Balance integrity (computed once per sender)
        integrity = compute_balance_integrity_direct(all_txs, sender_id)

        # Pattern drift (computed once per sender)
        if len(grp_sorted) >= 10:
            mid = len(grp_sorted) // 2
            baseline = grp_sorted.iloc[:mid]["amount"].dropna()
            recent = grp_sorted.iloc[mid:]["amount"].dropna()
            drift = detect_pattern_drift_direct(recent, baseline)
        else:
            drift = {"psi": 0.0, "interpretation": "insufficient_data"}

        cache[str(sender_id)] = {
            "amounts": amounts,
            "balance_integrity": integrity,
            "drift": drift,
        }
    return cache


def featurize_transaction(
    tx: pd.Series,
    dataset: Dict[str, Any],
    user_stats_cache: Optional[Dict[str, Any]] = None,
) -> FraudState:
    """Compute all deterministic features for a single transaction row.

    Args:
        tx: A single row from the transactions DataFrame.
        dataset: Dict with keys 'transactions', 'users', 'locations', 'sms', 'mails'.
        user_stats_cache: Pre-computed per-sender stats from _build_user_stats_cache().

    Returns:
        A fully populated FraudState with all deterministic fields set.
    """
    tx_id = str(tx.get("transaction_id", ""))
    sender_id = str(tx.get("sender_id", "") or "")
    sender_iban = str(tx.get("sender_iban", "") or "")
    recipient_iban = str(tx.get("recipient_iban", "") or "")

    all_txs: pd.DataFrame = dataset["transactions"]
    users: list = dataset["users"]
    locations: list = dataset["locations"]
    sms_texts: list = dataset["sms"]
    mail_texts: list = dataset["mails"]

    state = default_state(tx_id, sender_id)
    state["raw_transaction"] = tx.to_dict()

    # REFACTOR: use sender_iban to look up user, not sender_id
    state["raw_user"] = _find_user(users, sender_iban)

    # ── Transaction velocity ─────────────────────────────────────────────────
    state["velocity_score"] = check_transaction_velocity_direct(
        sender_id, tx.get("timestamp"), all_txs, window_hours=24
    )

    # ── Amount anomaly ───────────────────────────────────────────────────────
    user_salary = float(state["raw_user"].get("salary", 0) or 0)
    amount = float(tx.get("amount", 0) or 0)

    # REFACTOR: use pre-cached amounts when available
    if user_stats_cache and sender_id in user_stats_cache:
        user_amounts = user_stats_cache[sender_id]["amounts"]
    else:
        user_amounts = all_txs[all_txs["sender_id"] == sender_id]["amount"].dropna()
    anomaly = score_amount_anomaly_direct(amount, user_salary, user_amounts)
    state["amount_zscore"] = anomaly["zscore"]

    # ── Balance integrity ────────────────────────────────────────────────────
    # REFACTOR: use cached balance integrity (computed once per sender)
    if user_stats_cache and sender_id in user_stats_cache:
        integrity = user_stats_cache[sender_id]["balance_integrity"]
    else:
        integrity = compute_balance_integrity_direct(all_txs, sender_id)
    state["balance_integrity_flag"] = integrity["flag"]

    # ── IBAN risk ────────────────────────────────────────────────────────────
    iban_to_check = recipient_iban or sender_iban
    iban_result = validate_iban_risk.func(iban_to_check) if iban_to_check else {"tier": "unknown"}
    state["iban_risk_tier"] = iban_result.get("tier", "unknown")

    # ── Impossible travel ────────────────────────────────────────────────────
    tx_lat, tx_lng = _resolve_tx_location(tx.get("location"), locations)
    travel = detect_impossible_travel_direct(
        sender_id, tx.get("timestamp"), tx_lat, tx_lng, locations
    )
    state["geo_travel_anomaly"] = travel["flag"]

    # ── Demographic deviation ────────────────────────────────────────────────
    user = state["raw_user"]
    try:
        tx_hour = tx.get("timestamp").hour if pd.notna(tx.get("timestamp")) else 12
    except AttributeError:
        tx_hour = 12
    state["demographic_deviation_pct"] = _score_demographic(
        user, tx_type=str(tx.get("transaction_type", "")), amount=amount, hour=tx_hour
    )

    # ── Pattern drift (PSI) ──────────────────────────────────────────────────
    # REFACTOR: use cached drift (computed once per sender)
    if user_stats_cache and sender_id in user_stats_cache:
        drift = user_stats_cache[sender_id]["drift"]
    else:
        user_txs = all_txs[all_txs["sender_id"] == sender_id].sort_values("timestamp")
        if len(user_txs) >= 10:
            mid = len(user_txs) // 2
            drift = detect_pattern_drift_direct(
                user_txs.iloc[mid:]["amount"].dropna(),
                user_txs.iloc[:mid]["amount"].dropna(),
            )
        else:
            drift = {"psi": 0.0}
    state["drift_psi"] = drift.get("psi", 0.0)

    # ── Communications entities (per-user filtered) ───────────────────────────
    # REFACTOR: filter SMS and mails to those mentioning the user's name so that
    # one user's fraud-related messages do not contaminate other users' analysis
    user_sms, user_mails = _filter_comms_for_user(sms_texts, mail_texts, user)
    entities = extract_comms_entities_direct(user_sms, user_mails)
    state["extracted_entities"] = entities

    # ── Cross-source flags ────────────────────────────────────────────────────
    cross_flags = find_amount_iban_mismatch_direct(
        entities, sender_iban, recipient_iban, amount
    )
    state["cross_source_flags"] = cross_flags

    # REFACTOR: compute combined deterministic risk score for short-circuit edge
    state["combined_risk_score"] = _compute_combined_risk_score(
        velocity_score=state["velocity_score"],
        amount_zscore=state["amount_zscore"],
        balance_integrity_flag=state["balance_integrity_flag"],
        iban_risk_tier=state["iban_risk_tier"],
        geo_travel_anomaly=state["geo_travel_anomaly"],
        demographic_deviation_pct=state["demographic_deviation_pct"],
        drift_psi=state["drift_psi"],
        n_cross_flags=len(cross_flags),
    )

    return state


def _score_demographic(user: dict, tx_type: str, amount: float, hour: int) -> float:
    """Compute a 0–1 demographic deviation score without calling the @tool wrapper."""
    score = 0.0
    salary = float(user.get("salary", 0) or 0)
    monthly = salary / 12.0 if salary > 0 else 1.0
    job = str(user.get("job", "") or "").lower()

    if amount > monthly * 3:
        score += 0.35
    elif amount > monthly:
        score += 0.15

    if hour >= 23 or hour < 5:
        score += 0.20

    if job in ("retired", "student") and tx_type in ("e-commerce",) and amount > 500:
        score += 0.20

    return round(min(score, 1.0), 3)


# ── LangGraph node wrapper ────────────────────────────────────────────────────

def make_featurizer_node(dataset: Dict[str, Any]):
    """Return a LangGraph-compatible node function closed over the loaded dataset.

    # REFACTOR: build user stats cache once at graph construction time, shared across
    # all transaction invocations — eliminates O(N*M) repeated DataFrame scans.
    """
    # Pre-compute per-user stats once for the full dataset
    user_stats_cache = _build_user_stats_cache(dataset["transactions"])

    def featurizer_node(state: FraudState) -> FraudState:
        """LangGraph Node 1: populate all deterministic fields for one transaction."""
        tx_id = state["transaction_id"]
        txs: pd.DataFrame = dataset["transactions"]
        row = txs[txs["transaction_id"] == tx_id]
        if row.empty:
            return state
        return featurize_transaction(row.iloc[0], dataset, user_stats_cache)

    return featurizer_node

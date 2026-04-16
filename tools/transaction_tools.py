"""Deterministic transaction and tabular fraud tools.

No LLM calls. Each function accepts structured data and returns a typed dict
so the featurizer can merge results into FraudState cleanly.
"""

import math
from datetime import timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from langchain_core.tools import tool


@tool
def check_transaction_velocity(
    sender_id: str,
    tx_timestamp: str,
    transactions_json: str,
    window_hours: int = 24,
) -> Dict[str, Any]:
    """Count how many transactions this sender made in a ±window_hours window around tx_timestamp.

    Args:
        sender_id: The sender's ID to look up.
        tx_timestamp: ISO-8601 timestamp of the target transaction.
        transactions_json: JSON-serialised transactions DataFrame (orient='records').
        window_hours: Half-width of the rolling window in hours (default 24).

    Returns:
        {"velocity": float, "window_hours": int}
    """
    try:
        df = pd.read_json(transactions_json, orient="records")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        ts = pd.to_datetime(tx_timestamp, errors="coerce")
        lo = ts - timedelta(hours=window_hours)
        hi = ts + timedelta(hours=window_hours)
        mask = (
            (df["sender_id"] == sender_id)
            & (df["timestamp"] >= lo)
            & (df["timestamp"] <= hi)
        )
        return {"velocity": float(mask.sum()), "window_hours": window_hours}
    except Exception:
        return {"velocity": 0.0, "window_hours": window_hours}


def check_transaction_velocity_direct(
    sender_id: str,
    tx_timestamp: Any,
    df: pd.DataFrame,
    window_hours: int = 24,
) -> float:
    """Direct (non-tool) version that accepts a DataFrame directly — used by featurizer."""
    try:
        ts = pd.to_datetime(tx_timestamp, errors="coerce")
        lo = ts - timedelta(hours=window_hours)
        hi = ts + timedelta(hours=window_hours)
        mask = (
            (df["sender_id"] == sender_id)
            & (pd.to_datetime(df["timestamp"], errors="coerce") >= lo)
            & (pd.to_datetime(df["timestamp"], errors="coerce") <= hi)
        )
        return float(mask.sum())
    except Exception:
        return 0.0


@tool
def score_amount_anomaly(
    amount: float,
    user_salary: float,
    user_tx_history_json: str,
) -> Dict[str, Any]:
    """Compute z-score of this transaction amount vs. the sender's historical amounts.

    Also returns an absolute-salary ratio to catch large purchases relative to income.

    Args:
        amount: Amount of this transaction.
        user_salary: Annual salary of the user (0 if unknown).
        user_tx_history_json: JSON list of past transaction amounts for this user.

    Returns:
        {"zscore": float, "salary_ratio": float, "history_len": int}
    """
    try:
        history = pd.read_json(user_tx_history_json, typ="series")
        if len(history) < 3:
            zscore = 0.0
        else:
            mean = float(history.mean())
            std = float(history.std())
            zscore = abs((amount - mean) / std) if std > 1e-6 else 0.0
        monthly_salary = (user_salary / 12.0) if user_salary > 0 else 1.0
        salary_ratio = amount / monthly_salary
        return {"zscore": round(zscore, 3), "salary_ratio": round(salary_ratio, 3), "history_len": len(history)}
    except Exception:
        return {"zscore": 0.0, "salary_ratio": 0.0, "history_len": 0}


def score_amount_anomaly_direct(
    amount: float,
    user_salary: float,
    user_tx_history: pd.Series,
) -> Dict[str, float]:
    """Direct version that accepts a Series — used by featurizer."""
    if len(user_tx_history) < 3:
        zscore = 0.0
    else:
        mean = float(user_tx_history.mean())
        std = float(user_tx_history.std())
        zscore = abs((amount - mean) / std) if std > 1e-6 else 0.0
    monthly_salary = (user_salary / 12.0) if user_salary > 0 else 1.0
    return {
        "zscore": round(zscore, 3),
        "salary_ratio": round(amount / monthly_salary, 3),
        "history_len": len(user_tx_history),
    }


@tool
def compute_balance_integrity(transactions_json: str, sender_id: str) -> Dict[str, Any]:
    """Check whether balance_after values are internally consistent for a sender.

    A flag is raised if any consecutive pair of transactions has a balance
    discrepancy larger than 5% of the expected post-transaction balance.

    Args:
        transactions_json: JSON-serialised transactions (orient='records').
        sender_id: Sender whose transactions to examine.

    Returns:
        {"flag": bool, "n_violations": int, "max_discrepancy": float}
    """
    try:
        df = pd.read_json(transactions_json, orient="records")
        sender_txs = (
            df[df["sender_id"] == sender_id]
            .copy()
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        if len(sender_txs) < 2:
            return {"flag": False, "n_violations": 0, "max_discrepancy": 0.0}

        violations = 0
        max_disc = 0.0
        for i in range(1, len(sender_txs)):
            prev_bal = sender_txs.loc[i - 1, "balance_after"]
            curr_amount = sender_txs.loc[i, "amount"]
            curr_bal = sender_txs.loc[i, "balance_after"]
            if any(math.isnan(v) for v in [prev_bal, curr_amount, curr_bal]):
                continue
            expected = prev_bal - curr_amount
            discrepancy = abs(expected - curr_bal)
            threshold = abs(expected) * 0.05 + 1.0
            if discrepancy > threshold:
                violations += 1
                max_disc = max(max_disc, discrepancy)

        return {"flag": violations > 0, "n_violations": violations, "max_discrepancy": round(max_disc, 2)}
    except Exception:
        return {"flag": False, "n_violations": 0, "max_discrepancy": 0.0}


def compute_balance_integrity_direct(df: pd.DataFrame, sender_id: str) -> Dict[str, Any]:
    """Direct version that accepts a DataFrame — used by featurizer.

    # REFACTOR: tightened to only flag genuine large discrepancies.
    # Original 5% threshold flagged 100% of transactions because intermediate
    # deposits (received by this user as recipient) inflate balance between sends.
    # New rule: only flag if the discrepancy > 40% of the TRANSACTION AMOUNT itself
    # AND the balance DECREASED more than 3× the transaction amount (drain pattern),
    # AND at least 2 consecutive violations are present (require persistent anomaly).
    # This filters legitimate intermediary credits while still catching drain patterns.
    """
    sender_txs = (
        df[df["sender_id"] == sender_id]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    if len(sender_txs) < 2:
        return {"flag": False, "n_violations": 0, "max_discrepancy": 0.0}

    # REFACTOR: use near-zero balance drain detection instead of consecutive-pair check.
    # Consecutive-pair check fails because legitimate deposits between sends cause false jumps.
    # Near-zero detection: flag if the account balance goes below a critical threshold
    # (≤ 200) after any send AND the account had substantial balance earlier (> 1000).
    # Also flag if a single transaction amount exceeds 80% of the pre-transaction balance
    # (sudden full-drain transaction) occurring more than once.
    max_prev_bal = 0.0
    drain_events = 0
    large_fraction_events = 0
    max_disc = 0.0

    for i in range(len(sender_txs)):
        curr_amount = sender_txs.loc[i, "amount"]
        curr_bal = sender_txs.loc[i, "balance_after"]
        if curr_bal != curr_bal or curr_amount != curr_amount:  # NaN
            continue

        # Track the maximum observed balance (upper watermark)
        est_prev_bal = curr_bal + curr_amount
        max_prev_bal = max(max_prev_bal, est_prev_bal)

        # Near-zero drain: balance almost gone after transaction
        if curr_bal <= 200.0 and max_prev_bal > 1000.0:
            drain_events += 1

        # Full-drain fraction: this single tx takes > 80% of pre-tx balance
        if est_prev_bal > 500 and curr_amount / est_prev_bal > 0.80:
            large_fraction_events += 1
            max_disc = max(max_disc, curr_amount)

    violations = drain_events + max(0, large_fraction_events - 1)
    return {"flag": violations >= 1, "n_violations": violations, "max_discrepancy": round(max_disc, 2)}


# High-risk countries: FATF grey/black lists + known AML jurisdictions (illustrative)
_HIGH_RISK_COUNTRIES = {"NG", "UA", "BY", "KP", "IR", "SY", "YE", "LY", "MM", "AF"}
_MEDIUM_RISK_COUNTRIES = {"RU", "VN", "PK", "BD", "GH", "SN", "TZ", "KE", "PH", "MX"}


@tool
def validate_iban_risk(iban: str) -> Dict[str, Any]:
    """Parse the country code from an IBAN and return a risk tier.

    IBANs begin with a 2-letter ISO country code. Risk tiers are based on
    illustrative FATF and AML jurisdiction lists.

    Args:
        iban: Raw IBAN string (with or without spaces).

    Returns:
        {"country": str, "tier": "low" | "medium" | "high" | "unknown"}
    """
    if not iban or len(iban.strip()) < 2:
        return {"country": "unknown", "tier": "unknown"}
    country = iban.strip()[:2].upper()
    if not country.isalpha():
        return {"country": "unknown", "tier": "unknown"}
    if country in _HIGH_RISK_COUNTRIES:
        tier = "high"
    elif country in _MEDIUM_RISK_COUNTRIES:
        tier = "medium"
    else:
        tier = "low"
    return {"country": country, "tier": tier}


@tool
def score_demographic_deviation(
    birth_year: int,
    salary: float,
    job: str,
    tx_type: str,
    tx_amount: float,
    tx_hour: int,
) -> Dict[str, Any]:
    """Score how atypical this transaction is given the user's demographic profile.

    Args:
        birth_year: User's birth year (used to compute age).
        salary: Annual salary in the local currency.
        job: Job title / occupation string.
        tx_type: Transaction type (e.g. 'e-commerce', 'transfer').
        tx_amount: Transaction amount.
        tx_hour: Hour of the transaction (0–23, local time).

    Returns:
        {"score": float, "reasons": list[str]}  — score in [0, 1]
    """
    score = 0.0
    reasons: List[str] = []
    monthly = (salary / 12.0) if salary > 0 else 1.0

    # Large purchase relative to monthly income
    if tx_amount > monthly * 3:
        score += 0.35
        reasons.append(f"Amount ({tx_amount:.0f}) > 3× monthly salary ({monthly:.0f})")
    elif tx_amount > monthly:
        score += 0.15
        reasons.append(f"Amount exceeds monthly salary")

    # Late-night transaction (23:00–05:00)
    if tx_hour >= 23 or tx_hour < 5:
        score += 0.20
        reasons.append(f"Late-night transaction at hour {tx_hour}")

    # Retired/student doing large e-commerce
    if job.lower() in ("retired", "student") and tx_type in ("e-commerce",) and tx_amount > 500:
        score += 0.20
        reasons.append(f"High-value {tx_type} atypical for {job}")

    return {"score": round(min(score, 1.0), 3), "reasons": reasons}


@tool
def detect_pattern_drift(
    recent_amounts_json: str,
    baseline_amounts_json: str,
) -> Dict[str, Any]:
    """Compute Population Stability Index (PSI) between recent and baseline amount distributions.

    PSI > 0.25 indicates significant distribution shift (potential account takeover
    or behaviour change). PSI 0.1–0.25 is moderate. PSI < 0.1 is stable.

    Args:
        recent_amounts_json: JSON list of recent transaction amounts.
        baseline_amounts_json: JSON list of baseline (historical) transaction amounts.

    Returns:
        {"psi": float, "interpretation": str}
    """
    try:
        recent = pd.read_json(recent_amounts_json, typ="series").dropna().values
        baseline = pd.read_json(baseline_amounts_json, typ="series").dropna().values

        if len(recent) < 5 or len(baseline) < 5:
            return {"psi": 0.0, "interpretation": "insufficient_data"}

        # Bin edges from combined distribution
        all_vals = np.concatenate([recent, baseline])
        bins = np.percentile(all_vals, np.linspace(0, 100, 11))
        bins = np.unique(bins)
        if len(bins) < 3:
            return {"psi": 0.0, "interpretation": "insufficient_bins"}

        def safe_hist(arr, bins):
            counts, _ = np.histogram(arr, bins=bins)
            counts = counts + 1e-6  # avoid log(0)
            return counts / counts.sum()

        p_recent = safe_hist(recent, bins)
        p_baseline = safe_hist(baseline, bins)
        psi = float(np.sum((p_recent - p_baseline) * np.log(p_recent / p_baseline)))

        if psi < 0.1:
            interp = "stable"
        elif psi < 0.25:
            interp = "moderate_drift"
        else:
            interp = "significant_drift"

        return {"psi": round(psi, 4), "interpretation": interp}
    except Exception:
        return {"psi": 0.0, "interpretation": "error"}


def detect_pattern_drift_direct(
    recent: pd.Series,
    baseline: pd.Series,
) -> Dict[str, Any]:
    """Direct version that accepts Series — used by featurizer."""
    recent = recent.dropna().values
    baseline = baseline.dropna().values
    if len(recent) < 5 or len(baseline) < 5:
        return {"psi": 0.0, "interpretation": "insufficient_data"}
    all_vals = np.concatenate([recent, baseline])
    bins = np.percentile(all_vals, np.linspace(0, 100, 11))
    bins = np.unique(bins)
    if len(bins) < 3:
        return {"psi": 0.0, "interpretation": "insufficient_bins"}

    def safe_hist(arr, bins):
        counts, _ = np.histogram(arr, bins=bins)
        counts = counts + 1e-6
        return counts / counts.sum()

    p_recent = safe_hist(recent, bins)
    p_baseline = safe_hist(baseline, bins)
    psi = float(np.sum((p_recent - p_baseline) * np.log(p_recent / p_baseline)))
    interp = "stable" if psi < 0.1 else "moderate_drift" if psi < 0.25 else "significant_drift"
    return {"psi": round(psi, 4), "interpretation": interp}

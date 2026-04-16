"""calibrate.py — Threshold calibration script for the Supervisor Agent.

# REFACTOR (Improvement Area 5): Implements per-level threshold calibration.
# Loads training labels, runs the pipeline on a subset, sweeps fraud_threshold and
# review_threshold across a grid, and writes the best pair to a JSON config file
# that supervisor_agent.py reads at inference time.

Usage:
    python calibrate.py --level brave-new-world --labels-file outputs/brave-new-world_train_labels.txt
    python calibrate.py --level brave-new-world --labels-file labels.txt --max-transactions 50

Output:
    thresholds/{level}_thresholds.json  — {"fraud_threshold": 0.72, "review_threshold": 0.45}
"""

import argparse
import json
import os
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv()


def load_ids(path: str) -> set:
    """Load a set of transaction IDs from a one-per-line ASCII file."""
    with open(path, "r", encoding="ascii", errors="ignore") as f:
        return {line.strip() for line in f if line.strip()}


def _score_thresholds(
    states: List[dict],
    ground_truth: set,
    total: int,
    fraud_threshold: float,
    review_threshold: float,
) -> Dict[str, float]:
    """Apply thresholds to agent signals and compute precision/recall/F1/FPR."""
    predicted: set = set()
    for s in states:
        avg = (s.get("transaction_fraud_signal", 0.0) + s.get("comms_fraud_signal", 0.0)) / 2.0
        crs = s.get("combined_risk_score", 0.0)
        tx_sig = s.get("transaction_fraud_signal", 0.0)
        comms_sig = s.get("comms_fraud_signal", 0.0)

        # Apply same logic as supervisor_prompt REFACTOR thresholds
        is_fraud = (
            (tx_sig >= fraud_threshold and comms_sig >= fraud_threshold and crs >= 0.50)
            or crs >= 0.85
        )
        is_review = (
            tx_sig >= review_threshold
            or comms_sig >= review_threshold
            or avg >= review_threshold
            or (crs >= 0.65 and max(tx_sig, comms_sig) >= review_threshold * 0.7)
        )

        if is_fraud or is_review:
            predicted.add(s["transaction_id"])

    tp = len(predicted & ground_truth)
    fp = len(predicted - ground_truth)
    fn = len(ground_truth - predicted)
    tn = total - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_predicted": len(predicted),
    }


def _run_pipeline_for_calibration(
    dataset: dict,
    max_transactions: int,
) -> List[dict]:
    """Run featurizer + both agents on all transactions, return list of partial FraudStates.

    Only runs featurizer + agents (not supervisor) to collect raw signals.
    """
    from featurizer import make_featurizer_node, _build_user_stats_cache
    from agents.transaction_agent import run_transaction_agent
    from agents.comms_agent import run_comms_agent
    from state import default_state

    txs = dataset["transactions"]
    if max_transactions:
        txs = txs.head(max_transactions)

    featurizer = make_featurizer_node(dataset)
    states = []

    for i, (_, row) in enumerate(txs.iterrows(), 1):
        tx_id = str(row["transaction_id"])
        initial = default_state(tx_id, str(row.get("sender_id", "")))
        try:
            feat_state = featurizer(initial)
            tx_state = run_transaction_agent(feat_state)
            full_state = run_comms_agent(tx_state)
            states.append(full_state)
        except Exception as e:
            print(f"  [WARN] tx {tx_id} failed: {e}", file=sys.stderr)

        if i % 10 == 0 or i == len(txs):
            print(f"  calibration [{i}/{len(txs)}]")

    return states


def main() -> None:
    """Entry point for the calibration CLI."""
    parser = argparse.ArgumentParser(description="Reply Mirror threshold calibrator")
    parser.add_argument("--level", required=True, help="Level name (e.g. brave-new-world)")
    parser.add_argument("--labels-file", required=True, help="Ground-truth label file path")
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "train-validation"))
    parser.add_argument("--output-dir", default="thresholds")
    parser.add_argument(
        "--max-transactions", type=int, default=None,
        help="Cap number of transactions (default: use all)"
    )
    parser.add_argument(
        "--fraud-grid", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80",
        help="Comma-separated fraud_threshold values to sweep"
    )
    parser.add_argument(
        "--review-grid", default="0.30,0.35,0.40,0.45,0.50,0.55",
        help="Comma-separated review_threshold values to sweep"
    )
    args = parser.parse_args()

    from data_loader import load_dataset

    print(f"[calibrate] Loading {args.level} / {args.split} …")
    dataset = load_dataset(args.data_dir, args.level, args.split)
    total = len(dataset["transactions"])
    ground_truth = load_ids(args.labels_file)
    print(f"[calibrate] {total} transactions, {len(ground_truth)} ground-truth frauds")

    print("[calibrate] Running pipeline to collect raw signals …")
    t0 = time.time()
    states = _run_pipeline_for_calibration(dataset, args.max_transactions)
    print(f"[calibrate] Collected {len(states)} states in {time.time()-t0:.1f}s")

    fraud_grid = [float(x) for x in args.fraud_grid.split(",")]
    review_grid = [float(x) for x in args.review_grid.split(",")]

    print(f"\n[calibrate] Sweeping {len(fraud_grid)*len(review_grid)} threshold combinations …")

    results: List[Tuple[float, float, Dict]] = []
    for ft, rt in product(fraud_grid, review_grid):
        if rt >= ft:
            continue  # review threshold must be lower than fraud threshold
        metrics = _score_thresholds(states, ground_truth, total, ft, rt)
        if metrics["recall"] < 0.15:
            continue  # disqualified
        results.append((ft, rt, metrics))

    if not results:
        print("[calibrate] WARNING: no valid threshold combination found (all recall < 0.15)")
        return

    # Rank by F1; break ties by lower FPR
    results.sort(key=lambda x: (-x[2]["f1"], x[2]["fpr"]))

    print(f"\n{'='*70}")
    print(f"  Top 10 threshold combinations")
    print(f"{'='*70}")
    print(f"  {'fraud_t':>8}  {'review_t':>8}  {'precision':>10}  {'recall':>8}  {'f1':>8}  {'fpr':>8}  {'n_pred':>6}")
    for ft, rt, m in results[:10]:
        print(
            f"  {ft:8.2f}  {rt:8.2f}  "
            f"{m['precision']:10.4f}  {m['recall']:8.4f}  {m['f1']:8.4f}  "
            f"{m['fpr']:8.4f}  {m['n_predicted']:6d}"
        )

    best_ft, best_rt, best_m = results[0]
    print(f"\n[calibrate] Best: fraud_threshold={best_ft}, review_threshold={best_rt}")
    print(f"           F1={best_m['f1']:.4f}  Precision={best_m['precision']:.4f}  Recall={best_m['recall']:.4f}")

    # Write thresholds to JSON config
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output_dir) / f"{args.level}_thresholds.json"
    config = {
        "level": args.level,
        "fraud_threshold": best_ft,
        "review_threshold": best_rt,
        "calibrated_metrics": {
            "f1": round(best_m["f1"], 4),
            "precision": round(best_m["precision"], 4),
            "recall": round(best_m["recall"], 4),
            "fpr": round(best_m["fpr"], 4),
        },
        "n_states": len(states),
    }
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[calibrate] Thresholds written to {out_path}")


if __name__ == "__main__":
    main()

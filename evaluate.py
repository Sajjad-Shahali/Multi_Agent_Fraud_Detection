"""evaluate.py — Local evaluation harness.

Compares pipeline output against ground-truth labels from the training data.
Prints precision, recall, F1, FPR, and LLM cost/efficiency metrics.

Usage:
    python evaluate.py --level brave-new-world --split train
    python evaluate.py --level deus-ex --split train --output-dir outputs
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Fraud indicator patterns in transaction descriptions / amounts
# The training data does NOT have a "label" column — we infer ground truth
# from the official output files produced during the competition.
# For self-evaluation, compare against a previously generated output file.
_USAGE = """
evaluate.py compares a pipeline output file against a reference labels file.

Reference labels file format (one fraudulent transaction_id per line, same as output):
    <transaction_id>
    ...

If no --labels-file is provided, the script looks for:
    outputs/{level}_{split}_labels.txt

To obtain ground-truth labels for the training split:
    - Run the pipeline with your best model.
    - Manually inspect and verify results.
    - Or use the competition portal's training-split scoring endpoint.
"""


def load_ids(path: str) -> set:
    """Load a set of transaction IDs from a one-per-line ASCII file."""
    with open(path, "r", encoding="ascii", errors="ignore") as f:
        return {line.strip() for line in f if line.strip()}


def compute_metrics(predicted: set, ground_truth: set, total_transactions: int) -> dict:
    """Compute precision, recall, F1, and false-positive rate."""
    tp = len(predicted & ground_truth)
    fp = len(predicted - ground_truth)
    fn = len(ground_truth - predicted)
    tn = total_transactions - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
        "total": total_transactions,
        "n_predicted": len(predicted),
        "n_ground_truth": len(ground_truth),
    }


def main() -> None:
    """Entry point for the evaluation CLI."""
    parser = argparse.ArgumentParser(description="Reply Mirror pipeline evaluator", epilog=_USAGE)
    parser.add_argument("--level", required=True)
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--labels-file", default=None,
                        help="Path to ground-truth labels file (one tx_id per line)")
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "train-validation"))
    args = parser.parse_args()

    output_file = Path(args.output_dir) / f"{args.level}_{args.split}_output.txt"
    if not output_file.exists():
        print(f"[evaluate] ERROR: output file not found: {output_file}", file=sys.stderr)
        print(f"[evaluate] Run: python pipeline.py --level {args.level} --split {args.split} first.")
        sys.exit(1)

    # Resolve labels file
    if args.labels_file:
        labels_file = Path(args.labels_file)
    else:
        labels_file = Path(args.output_dir) / f"{args.level}_{args.split}_labels.txt"

    if not labels_file.exists():
        print(f"[evaluate] No labels file found at: {labels_file}")
        print("[evaluate] Provide --labels-file or place ground-truth labels at the expected path.")
        print("[evaluate] Printing output file statistics only.\n")
        predicted = load_ids(str(output_file))
        from data_loader import load_dataset
        dataset = load_dataset(args.data_dir, args.level, args.split)
        total = len(dataset["transactions"])
        print(f"  Transactions in dataset : {total}")
        print(f"  Flagged (FRAUD+REVIEW)  : {len(predicted)}")
        print(f"  Flag rate               : {len(predicted)/total*100:.1f}%")
        sys.exit(0)

    predicted = load_ids(str(output_file))
    ground_truth = load_ids(str(labels_file))

    from data_loader import load_dataset
    dataset = load_dataset(args.data_dir, args.level, args.split)
    total = len(dataset["transactions"])

    m = compute_metrics(predicted, ground_truth, total)

    # ── Print evaluation report ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Reply Mirror — Evaluation Report")
    print(f"  Level: {args.level}  |  Split: {args.split}")
    print(f"{'='*60}")
    print(f"\n  Transactions total   : {m['total']}")
    print(f"  Ground truth fraud   : {m['n_ground_truth']}")
    print(f"  Predicted as fraud   : {m['n_predicted']}")
    print()
    print(f"  True  Positives (TP) : {m['tp']}")
    print(f"  False Positives (FP) : {m['fp']}")
    print(f"  False Negatives (FN) : {m['fn']}")
    print(f"  True  Negatives (TN) : {m['tn']}")
    print()
    print(f"  Precision            : {m['precision']:.4f}")
    print(f"  Recall               : {m['recall']:.4f}")
    print(f"  F1 Score             : {m['f1']:.4f}")
    print(f"  False Positive Rate  : {m['fpr']:.4f}")
    print()

    # Competition disqualification check
    if m["recall"] < 0.15:
        print("  !! WARNING: RECALL < 0.15 — SUBMISSION WOULD BE INVALID !!")
        print("  !! The competition requires at least 15% of fraud correctly identified !!")
    else:
        print(f"  Recall is {m['recall']:.2%} — above the 15% disqualification threshold. OK.")

    # LLM usage stats (from Langfuse trace log file if available)
    trace_log = Path(args.output_dir) / f"{args.level}_{args.split}_trace.json"
    if trace_log.exists():
        import json
        with open(trace_log) as f:
            trace = json.load(f)
        print(f"\n  LLM Calls            : {trace.get('llm_calls', 'n/a')}")
        print(f"  Total Tokens         : {trace.get('total_tokens', 'n/a')}")
        print(f"  Est. Cost (USD)      : ${trace.get('estimated_cost_usd', 0):.4f}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()

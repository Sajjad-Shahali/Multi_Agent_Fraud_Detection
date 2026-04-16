"""pipeline.py — Run the full fraud detection pipeline for one level/split.

Usage:
    python pipeline.py --level brave-new-world --split train
    python pipeline.py --level deus-ex --split validation
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    """Entry point for the pipeline CLI."""
    parser = argparse.ArgumentParser(description="Reply Mirror fraud detection pipeline")
    parser.add_argument("--level", required=True, help="Level name (e.g. brave-new-world)")
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "train-validation"))
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-transactions", type=int, default=None,
                        help="Cap the number of transactions to process (for smoke testing)")
    args = parser.parse_args()

    from data_loader import load_dataset
    from graph import build_graph, write_output
    from session import generate_session_id, langchain_config
    from state import default_state

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"[pipeline] Loading {args.level} / {args.split} …")
    t0 = time.time()
    dataset = load_dataset(args.data_dir, args.level, args.split)
    txs = dataset["transactions"]
    print(f"[pipeline] Loaded {len(txs)} transactions in {time.time() - t0:.1f}s")

    if args.max_transactions:
        txs = txs.head(args.max_transactions)
        print(f"[pipeline] Capped to {len(txs)} transactions (--max-transactions)")

    # ── Build graph ───────────────────────────────────────────────────────────
    graph = build_graph(dataset)
    session_id = generate_session_id()
    cfg = langchain_config(session_id)
    print(f"[pipeline] Session ID: {session_id}")

    # ── Process each transaction ──────────────────────────────────────────────
    results = []
    errors = []
    for i, (_, row) in enumerate(txs.iterrows(), 1):
        tx_id = str(row["transaction_id"])
        initial_state = default_state(tx_id, str(row.get("sender_id", "")))

        try:
            final_state = graph.invoke(initial_state, config=cfg)
            results.append(final_state)
        except Exception as e:
            print(f"  [WARN] tx {tx_id} failed: {e}", file=sys.stderr)
            errors.append(tx_id)
            # On error: default to REVIEW (conservative — don't miss fraud)
            fallback = {**initial_state, "verdict": "REVIEW", "confidence": 0.0,
                        "explanation": f"Pipeline error: {e}"}
            results.append(fallback)

        if i % 10 == 0 or i == len(txs):
            fraud_count = sum(1 for r in results if r.get("verdict") == "FRAUD")
            review_count = sum(1 for r in results if r.get("verdict") == "REVIEW")
            print(f"  [{i}/{len(txs)}] FRAUD={fraud_count} REVIEW={review_count} ERRORS={len(errors)}")

    # ── Write output ──────────────────────────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_dir) / f"{args.level}_{args.split}_output.txt"
    n_written = write_output(results, str(output_path))

    total_time = time.time() - t0
    print(f"\n[pipeline] Done in {total_time:.1f}s")
    print(f"[pipeline] Transactions processed: {len(results)}")
    print(f"[pipeline] Flagged (FRAUD+REVIEW): {n_written}")
    print(f"[pipeline] Errors: {len(errors)}")
    print(f"[pipeline] Output: {output_path}")

    # Flush Langfuse traces
    try:
        from session import langfuse_client
        langfuse_client().flush()
        print("[pipeline] Langfuse traces flushed.")
    except Exception:
        pass  # Langfuse is optional for local runs without credentials


if __name__ == "__main__":
    main()

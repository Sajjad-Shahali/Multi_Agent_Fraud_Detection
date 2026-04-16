"""submit.py — Competition submission runner.

Runs the pipeline on the validation split, writes the ASCII output file,
prints a summary of verdict counts and costs, and traces everything to Langfuse.

Usage:
    python submit.py --level brave-new-world
    python submit.py --level deus-ex --max-transactions 50
"""

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    """Entry point for the submission CLI."""
    parser = argparse.ArgumentParser(description="Reply Mirror competition submission runner")
    parser.add_argument("--level", required=True)
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "train-validation"))
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-transactions", type=int, default=None)
    args = parser.parse_args()

    from data_loader import load_dataset
    from graph import build_graph, write_output
    from session import generate_session_id, langchain_config
    from state import default_state

    session_id = generate_session_id()
    cfg = langchain_config(session_id)

    print(f"\n[submit] Reply Mirror — Competition Submission Runner")
    print(f"[submit] Level  : {args.level}")
    print(f"[submit] Split  : validation")
    print(f"[submit] Session: {session_id}\n")

    # ── Load validation data ──────────────────────────────────────────────────
    t0 = time.time()
    dataset = load_dataset(args.data_dir, args.level, "validation")
    txs = dataset["transactions"]

    if args.max_transactions:
        txs = txs.head(args.max_transactions)

    print(f"[submit] Loaded {len(txs)} transactions.")

    # ── Build and run graph ───────────────────────────────────────────────────
    graph = build_graph(dataset)
    results = []
    errors = []

    for i, (_, row) in enumerate(txs.iterrows(), 1):
        tx_id = str(row["transaction_id"])
        initial_state = default_state(tx_id, str(row.get("sender_id", "")))
        try:
            final_state = graph.invoke(initial_state, config=cfg)
            results.append(final_state)
        except Exception as e:
            print(f"  [WARN] {tx_id} failed: {e}", file=sys.stderr)
            errors.append(tx_id)
            fallback = {**initial_state, "verdict": "REVIEW", "confidence": 0.0,
                        "explanation": f"Pipeline error: {e}"}
            results.append(fallback)

        if i % 10 == 0 or i == len(txs):
            counts = Counter(r.get("verdict", "pending") for r in results)
            print(f"  [{i}/{len(txs)}] FRAUD={counts['FRAUD']} "
                  f"REVIEW={counts['REVIEW']} CLEAN={counts['CLEAN']} ERR={len(errors)}")

    # ── Write output file ─────────────────────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_dir) / f"{args.level}_validation_output.txt"
    n_written = write_output(results, str(output_path))

    # ── Summary ───────────────────────────────────────────────────────────────
    total_time = time.time() - t0
    verdict_counts = Counter(r.get("verdict", "pending") for r in results)
    confidences = [r.get("confidence", 0.0) for r in results if r.get("verdict") in ("FRAUD", "REVIEW")]

    print(f"\n{'='*60}")
    print(f"  Submission Summary — {args.level} / validation")
    print(f"{'='*60}")
    print(f"  Transactions processed : {len(results)}")
    print(f"  FRAUD verdicts         : {verdict_counts['FRAUD']}")
    print(f"  REVIEW verdicts        : {verdict_counts['REVIEW']}")
    print(f"  CLEAN verdicts         : {verdict_counts['CLEAN']}")
    print(f"  Errors (-> REVIEW)     : {len(errors)}")
    print(f"  Flagged in output file : {n_written}")

    if confidences:
        print(f"\n  Confidence distribution (flagged transactions):")
        print(f"    min={min(confidences):.3f}  "
              f"max={max(confidences):.3f}  "
              f"mean={sum(confidences)/len(confidences):.3f}")

    print(f"\n  Session ID : {session_id}")
    print(f"  Wall time  : {total_time:.1f}s")
    print(f"  Output     : {output_path}")
    print(f"{'='*60}\n")

    # ── Flush Langfuse traces ─────────────────────────────────────────────────
    try:
        from session import langfuse_client
        lf = langfuse_client()
        lf.flush()
        print("[submit] Langfuse traces flushed. Check the dashboard.")
        print(f"[submit] Dashboard: {os.getenv('LANGFUSE_HOST', 'https://challenges.reply.com/langfuse')}")
    except Exception as e:
        print(f"[submit] Langfuse flush skipped ({e}) — run without credentials is OK locally.")


if __name__ == "__main__":
    main()

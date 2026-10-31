"""Runs every question in questions.csv and saves the results.

Usage:
    python run_demo.py                          (all rows, Llama 3.1 8B)
    python run_demo.py --limit 6                (only the first 6 rows, for a quick test)
    python run_demo.py --model qwen2.5:7b       (same rows, Qwen 2.5 7B)

It saves after every question, so if you stop it (Ctrl+C) or the laptop sleeps,
just run the same command again: finished questions are skipped.

Output files (one set per model):
    results_<model>.csv    one row per question (the numbers)
    log_<model>.txt        the full readable report for every question
At the end it prints the summary table (also available any time with: python summary.py)
"""
import argparse
import os
import time

import pandas as pd

import edueviGate as eg
from summary import summarize

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=eg.MODEL)
parser.add_argument("--limit", type=int, default=None)
args = parser.parse_args()
eg.MODEL = args.model

tag = args.model.replace(":", "_").replace("/", "_")
results_file, log_file = f"results_{tag}.csv", f"log_{tag}.txt"

done = pd.read_csv(results_file, dtype=str) if os.path.exists(results_file) else pd.DataFrame()
done_ids = set(done["question_id"]) if len(done) else set()

todo = eg.questions if args.limit is None else eg.questions.head(args.limit)
todo = todo[~todo["question_id"].isin(done_ids)]
print(f"Model: {args.model} | to run: {len(todo)} | already done: {len(done_ids)}")

for n, (_, q) in enumerate(todo.iterrows(), start=1):
    print("=" * 70)
    print(f"[{n}/{len(todo)}] {q['question_id']} ... (about a minute)")
    start = time.time()
    r = eg.run_question(q)
    seconds = round(time.time() - start, 1)
    text = eg.report(r)
    print(text)

    row = {
        "question_id": q["question_id"],
        "subject": q.get("subject", ""),
        "question_type": q.get("question_type", ""),
        "expected_state": q.get("expected_state", ""),
        "expected_option": q.get("expected_option", ""),
        "rag_answer": r["rag_answer"],
        "relevance": r["evidence"]["relevant"],
        "sufficient": r["evidence"]["sufficient"],
        "evidence_answer": r["evidence"]["evidence_answer"],
        "memory_answer": r["memory_answer"],
        "conflict": r["conflict"],
        "state": r["state"],
        "action": r["action"],
        "quote": r["evidence"].get("quote", ""),
        "seconds": seconds,
        "model": args.model,
    }
    done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
    done.to_csv(results_file, index=False)                 # saved after every question
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("=" * 70 + "\n" + text + "\n")

print("=" * 70)
print(summarize(done))
print(f"\nSaved {results_file} and {log_file}")

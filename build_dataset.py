"""Builds the PoC test set from full dataset (KnowShiftQA).

It writes two files that edueviGate.py reads:
  questions.csv  one row per test case (Conflicting or Insufficient)
  passages.csv   the KnowShiftQA textbook corpus (textbook.csv)

Which cases are chosen, and why:
  - only the DEV split (the test split is kept untouched for the final results)
  - Conflicting = condition B_updated_shift, kept only if the answer appears in the gold
    paragraph AND the gold paragraph is in the top 3 passages (so the evidence really is there)
  - Insufficient = condition C_insufficient, kept only if auto_c_quality == "likely_ok"
    (neither the answer nor option r1 appears in the remaining passages)
  - the same question is used in both conditions (a paired design)
  - balanced across the five subjects, fixed random seed

Usage:  python build_dataset.py            (60 questions -> 120 rows)
        python build_dataset.py 80         (80 questions -> 160 rows)
"""
import json
import os
import sys

import pandas as pd

FULL_DATASET = "EduTrust_full_dataset.csv"   # from your package: processed/full/
TEXTBOOK = "textbook.csv"                    # the KnowShiftQA textbook file
N_QUESTIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
TOP_K = 3
SEED = 20261031

full = pd.read_csv(FULL_DATASET)
dev = full[full.split == "dev"]

b = dev[(dev.condition == "B_updated_shift")
        & (dev.flag_answer_in_gold == True)          # noqa: E712
        & (dev.gold_rank <= TOP_K)]
c = dev[(dev.condition == "C_insufficient") & (dev.auto_c_quality == "likely_ok")]
eligible = sorted(set(b.question_id) & set(c.question_id))
subject_of = dict(zip(dev.question_id, dev.subject))

# take questions round-robin across subjects so small subjects (Physics) are not crowded out
by_subject = {}
for qid in pd.Series(eligible).sample(frac=1, random_state=SEED):
    by_subject.setdefault(subject_of[qid], []).append(qid)
chosen = []
while len(chosen) < N_QUESTIONS and any(by_subject.values()):
    for subj in sorted(by_subject):
        if by_subject[subj] and len(chosen) < N_QUESTIONS:
            chosen.append(by_subject[subj].pop())


def make_row(r, expected_state):
    ids = json.loads(r.context_passage_ids)[:TOP_K]
    return {
        "question_id": r.case_id,                 # e.g. KSQA-Q0366-B
        "base_question": r.question_id,
        "subject": r.subject,
        "question_type": r.question_type,
        "question": r.question,
        "A": r.option_A, "B": r.option_B, "C": r.option_C, "D": r.option_D,
        "expected_state": expected_state,
        "expected_option": r.expected_option if expected_state == "Conflicting" else "none",
        "context_ids": " ".join(str(i) for i in ids),
    }


rows = []
for qid in chosen:
    rows.append(make_row(b[b.question_id == qid].iloc[0], "Conflicting"))
    rows.append(make_row(c[c.question_id == qid].iloc[0], "Insufficient"))
questions = pd.DataFrame(rows)

# keep the old example files instead of overwriting them
for name in ["questions.csv", "passages.csv"]:
    backup = name.replace(".csv", "_examples.csv")
    if os.path.exists(name) and not os.path.exists(backup):
        os.rename(name, backup)

textbook = pd.read_csv(TEXTBOOK)
passages = pd.DataFrame({"passage_id": textbook["id"].astype(str),
                         "text": textbook["text"].astype(str).str.strip()})

questions.to_csv("questions.csv", index=False)
passages.to_csv("passages.csv", index=False)

print(f"Eligible dev questions: {len(eligible)}  ->  chosen: {len(chosen)}")
print(f"Wrote questions.csv ({len(questions)} rows) and passages.csv ({len(passages)} passages)")
print(questions.groupby(["subject", "expected_state"]).size().unstack().to_string())

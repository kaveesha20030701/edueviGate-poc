"""EduEviGate XAI: explains the evidence check with LIME (sentence level).

Question it answers:
    "Which sentences of the evidence made the system decide the evidence was
     sufficient (and for which option)?"

How it works:
    1. Split the evidence passages into sentences. Each sentence is one LIME feature.
    2. LIME makes many copies of the evidence with random sentences removed.
    3. EduEviGate's evidence check runs on every copy (one model call each).
    4. LIME fits a simple weighted linear model and gives each sentence a weight.
    5. Two checks of the explanation (only when the evidence supported an option):
       - Deletion test: remove the top sentence. Does the decision change?
         (compared with removing a random sentence, as a baseline)
       - Quote agreement: is the top LIME sentence the sentence the model quoted?

Usage:
    python explain.py KSQA-Q2133-B             (one case, 50 samples)
    python explain.py KSQA-Q2133-B --samples 80
    python explain.py --first 6                (the first 6 cases in questions.csv)
    python explain.py --first 6 --model qwen2.5:7b

Each sample is one model call, so ONE case = about 50 calls. On a laptop that can take
20-60 minutes per case: run a few cases overnight. Finished cases are skipped on re-run.

Output:
    explain_<case>.txt             the text report for each case
    lime_<case>.html               LIME's coloured chart (open in a browser)
    explain_results_<model>.csv    one row per case (for the summary numbers)
"""
import argparse
import os
import random
import re
import time

import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer

import edueviGate as eg

SEP = " ||| "      # separator between sentences in the text LIME perturbs
MASK = "REMOVED"   # what LIME puts in place of a removed sentence


# ---------- 1. Turn passages into a list of sentences ----------
def split_sentences(found):
    """[(passage_id, sentence), ...] for the evidence passages."""
    sentences = []
    for p in found:
        for s in re.split(r"(?<=[.!?])\s+|\n+", str(p["text"]).strip()):
            if s.strip():
                sentences.append((p["id"], s.strip().replace("|||", "|")))
    return sentences


def to_lime_text(sentences):
    """LIME works on one string, so each sentence becomes one 'token': P1::text ||| P2::text"""
    return SEP.join(f"{pid}::{s}" for pid, s in sentences)


def from_lime_text(text):
    """Turn a (perturbed) LIME string back into a list of passages for the evidence check."""
    passages = {}
    for part in text.split("|||"):
        part = part.strip()
        if not part or part == MASK or "::" not in part:
            continue  # this sentence was removed by LIME
        pid, sentence = part.split("::", 1)
        passages.setdefault(pid, []).append(sentence)
    return [{"id": pid, "text": " ".join(s), "score": 0} for pid, s in passages.items()]


# ---------- 2. What LIME explains ----------
def make_classifier(q, target_letter):
    """Return a function LIME can call: list of texts -> probabilities [not target, target].

    target_letter = the option the full evidence supported, or None if the full evidence
    was insufficient (then LIME explains 'judged sufficient for any option').
    """
    cache = {}

    def is_target(evidence):
        if evidence["sufficient"] != "yes":
            return 0.0
        if target_letter is None:
            return 1.0
        return 1.0 if evidence["evidence_answer"] == target_letter else 0.0

    def classifier_fn(texts):
        probs = []
        for i, text in enumerate(texts):
            found = from_lime_text(text)
            key = tuple((p["id"], p["text"]) for p in found)
            if key not in cache:
                cache[key] = is_target(eg.check_evidence(q, found)) if found else 0.0
            p = cache[key]
            probs.append([1 - p, p])
            print(f"   sample {i + 1}/{len(texts)}", end="\r", flush=True)
        print()
        return np.array(probs)

    return classifier_fn


def same_sentence(a, b):
    """True if two sentences are (almost) the same: one contains the other, or 70%+ word overlap."""
    a, b = a.lower().strip(), b.lower().strip()
    if not a or not b or b == "none":
        return False
    if a in b or b in a:
        return True
    wa, wb = set(eg.tokenize(a)), set(eg.tokenize(b))
    return len(wa & wb) / max(1, min(len(wa), len(wb))) >= 0.7


# ---------- 3. Explain one case ----------
def explain_question(qid, num_samples=50):
    q = eg.questions[eg.questions["question_id"] == qid].iloc[0]
    found = eg.get_evidence(q)
    start = time.time()

    # The normal EduEviGate result first
    evidence = eg.check_evidence(q, found)
    memory = eg.closed_book(q)
    state, action, _ = eg.decide(evidence, memory)
    target = evidence["evidence_answer"] if evidence["sufficient"] == "yes" else None

    # LIME
    sentences = split_sentences(found)
    explainer = LimeTextExplainer(
        class_names=["not target", "target"],
        split_expression=r"\s*\|\|\|\s*",  # split only between sentences
        bow=False,                        # features are sentence positions
        mask_string=MASK,
        random_state=42,                  # same samples every run
    )
    print(f"Running LIME on {qid}: {len(sentences)} sentences, {num_samples} samples "
          f"(up to {num_samples} model calls)...")
    exp = explainer.explain_instance(
        to_lime_text(sentences), make_classifier(q, target),
        labels=(1,), num_features=len(sentences), num_samples=num_samples,
    )
    weights = sorted(exp.as_list(label=1), key=lambda w: -w[1])  # biggest push first
    top_text = weights[0][0] if weights else ""
    top_pid, top_sentence = (top_text.split("::", 1) + [""])[:2]

    # Checks (only meaningful when the full evidence supported an option)
    deletion_changed = random_changed = quote_agrees = ""
    if target is not None:
        def state_without(removed):
            remaining = [(p, s) for p, s in sentences if f"{p}::{s}" != removed]
            ev = eg.check_evidence(q, from_lime_text(to_lime_text(remaining)))
            return eg.decide(ev, memory)[0]

        deletion_changed = state_without(top_text) != state
        others = [f"{p}::{s}" for p, s in sentences if f"{p}::{s}" != top_text]
        random_pick = random.Random(42).choice(others) if others else ""
        random_changed = state_without(random_pick) != state if random_pick else ""
        quote_agrees = same_sentence(top_sentence, str(evidence.get("quote", "")))

    # ---------- Report ----------
    what = (f"evidence supports option {target}" if target
            else "evidence judged sufficient (for any option)")
    lines = [
        f"LIME EXPLANATION FOR {qid}: {q['question']}",
        f"Model: {eg.MODEL} | Expected state: {q.get('expected_state', '')}",
        f"Decision: {state} -> {action}",
        f"LIME explains: what made the check decide '{what}'",
        f"Sentences: {len(sentences)} | Samples: {num_samples}",
        "",
        "Top sentences (positive = pushed towards the decision):",
    ]
    for feature, w in weights[:8]:
        pid, sentence = feature.split("::", 1)
        short = sentence if len(sentence) <= 100 else sentence[:97] + "..."
        lines.append(f"   {w:+.2f}  [{pid}] {short}")
    lines.append(f"Quoted by the model: {evidence.get('quote', '')}")
    if target is None:
        lines += [
            "",
            "Insufficient decision: weights near 0 mean no sentence made the evidence",
            "sufficient, which supports the decision to abstain. A clearly positive",
            "sentence shows what nearly fooled the check (e.g. a keyword match).",
        ]
    else:
        lines += [
            "",
            "Check 1 - deletion test (faithfulness):",
            f"   Remove the top LIME sentence -> decision "
            f"{'CHANGED (explanation is faithful)' if deletion_changed else 'UNCHANGED'}",
            f"   Remove a random sentence     -> decision "
            f"{'CHANGED' if random_changed else 'UNCHANGED'}  (baseline)",
            "Check 2 - quote agreement:",
            f"   Top LIME sentence {'IS' if quote_agrees else 'is NOT'} the sentence the model quoted",
        ]
    report = "\n".join(lines)
    print(report)

    with open(f"explain_{qid}.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    try:
        exp.save_to_file(f"lime_{qid}.html", labels=(1,))
    except Exception as err:  # the chart is optional
        print(f"(could not save the HTML chart: {err})")

    return {
        "question_id": qid, "model": eg.MODEL,
        "expected_state": q.get("expected_state", ""), "state": state,
        "target_option": target or "none", "n_sentences": len(sentences),
        "samples": num_samples, "top_passage": top_pid, "top_sentence": top_sentence,
        "top_weight": round(weights[0][1], 3) if weights else "",
        "quote": evidence.get("quote", ""), "quote_agrees": quote_agrees,
        "deletion_changed": deletion_changed, "random_deletion_changed": random_changed,
        "seconds": round(time.time() - start, 1),
    }


# ---------- 4. Run one or several cases ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case", nargs="?", help="a question_id, e.g. KSQA-Q2133-B")
    parser.add_argument("--first", type=int, help="explain the first N cases in questions.csv")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--model", default=eg.MODEL)
    args = parser.parse_args()
    eg.MODEL = args.model

    tag = args.model.replace(":", "_").replace("/", "_")
    out = f"explain_results_{tag}.csv"
    done = pd.read_csv(out, dtype=str) if os.path.exists(out) else pd.DataFrame()
    done_ids = set(done["question_id"]) if len(done) else set()

    if args.case:
        cases = [args.case]
    else:
        cases = list(eg.questions["question_id"].head(args.first or 2))
    for qid in cases:
        if qid in done_ids:
            print(f"{qid}: already explained, skipping")
            continue
        print("=" * 70)
        row = explain_question(qid, args.samples)
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
        done.to_csv(out, index=False)

    supported = done[done.target_option.astype(str) != "none"] if len(done) else done
    if len(supported):
        t = lambda col: (supported[col].astype(str) == "True").sum()  # noqa: E731
        n = len(supported)
        print("=" * 70)
        print(f"XAI SUMMARY ({args.model}, {n} cases where the evidence supported an option)")
        print(f"   Removing the top LIME sentence changed the decision: {t('deletion_changed')}/{n}")
        print(f"   Removing a random sentence changed the decision:     {t('random_deletion_changed')}/{n}")
        print(f"   Top LIME sentence = the model's quote:               {t('quote_agrees')}/{n}")
    print(f"Saved {out}")

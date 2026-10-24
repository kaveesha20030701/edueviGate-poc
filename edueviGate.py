"""EduEviGate PoC: the whole pipeline in one file.

retrieve -> conventional RAG answer -> evidence check -> closed-book answer -> decide -> message
"""
import json
import re

import ollama
import pandas as pd
from rank_bm25 import BM25Okapi

MODEL = "llama3.1:8b"   # change to "llama3.2:3b" if your laptop is too slow
TOP_K = 3               # how many passages to retrieve


# ---------- 1. Load the data ----------
passages = pd.read_csv("passages.csv", dtype=str)
questions = pd.read_csv("questions.csv", dtype=str).fillna("")


def tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


bm25 = BM25Okapi([tokenize(t) for t in passages["text"]])


# ---------- 2. Talk to the model ----------
def ask_llm(prompt):
    """Send a prompt to the local model and get a JSON dict back."""
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",                 # forces the model to reply in JSON
        options={"temperature": 0},    # same input -> same output
    )
    try:
        return json.loads(response["message"]["content"])
    except json.JSONDecodeError:
        return {}


def letter(value):
    """Turn 'C', 'c)', 'C) Prism' into 'C'. Anything else becomes 'none'."""
    match = re.match(r"\s*\(?([A-Ea-e])\b", str(value))
    return match.group(1).upper() if match else "none"


# ---------- 3. Helpers ----------
def get_options(q):
    return {L: q[L] for L in ["A", "B", "C", "D", "E"] if L in q and q[L]}


def options_text(q):
    return "\n".join(f"{L}) {text}" for L, text in get_options(q).items())


def retrieve(question_text):
    scores = bm25.get_scores(tokenize(question_text))
    best = scores.argsort()[::-1][:TOP_K]
    return [
        {"id": passages.iloc[i]["passage_id"],
         "text": passages.iloc[i]["text"],
         "score": round(float(scores[i]), 2)}
        for i in best
    ]


def passages_text(found):
    return "\n\n".join(f"[Passage {p['id']}] {p['text']}" for p in found)


# ---------- 4. The three prompts ----------
def conventional_rag(q, found):
    prompt = f"""You are a helpful tutor. Use the passages to answer the question.

Passages:
{passages_text(found)}

Question: {q['question']}
Options:
{options_text(q)}

Reply in JSON: {{"answer": "<option letter>", "explanation": "<one sentence>"}}"""
    result = ask_llm(prompt)
    return letter(result.get("answer")), result.get("explanation", "")

STOPWORDS = {"a", "an", "the", "by", "of", "in", "on", "to", "and", "or",
             "is", "it", "its", "with", "for", "from", "at", "as", "be"}


def option_in_quote(option_text, quote):
    """True if a key word of the chosen option appears in the quoted sentence."""
    words = [w for w in tokenize(option_text) if w not in STOPWORDS]
    quote = quote.lower()
    return bool(words) and any(w in quote for w in words)


def check_evidence(q, found):
    prompt = f"""You are checking evidence. Do NOT use your own knowledge.
These passages come from a course textbook that has been deliberately edited,
so some facts may differ from the real world. That is expected.
Your only job is to check what the passages SAY, not whether it is true.

Passages:
{passages_text(found)}

Question: {q['question']}
Options:
{options_text(q)}

Steps:
- quote: copy the exact sentence from the passages that answers the question.
  If no sentence directly answers it, write "none". Sharing keywords is NOT enough.
- evidence_answer: the option letter that the quoted sentence supports, or "none"
- passage_id: the id of the passage you quoted, or "none"
- relevant: are the passages about the topic of the question? "yes", "partly" or "no"
- reason: one short sentence

Reply in JSON: {{"quote": "", "evidence_answer": "", "passage_id": "", "relevant": "", "reason": ""}}"""
    result = ask_llm(prompt)
    quote = str(result.get("quote", "none")).strip()
    answer = letter(result.get("evidence_answer"))
    found_quote = quote.lower() not in ("", "none")
    opts = get_options(q)
    grounded = found_quote and answer in opts and option_in_quote(opts[answer], quote)
    return {
        "relevant": str(result.get("relevant", "no")).strip().lower(),
        "sufficient": "yes" if grounded else "no",
        "evidence_answer": answer,
        "passage_id": str(result.get("passage_id", "none")).replace("Passage", "").strip(),
        "quote": quote,
        "reason": result.get("reason", ""),
    }


def closed_book(q):
    prompt = f"""Answer from your own knowledge. No passages are given.

Question: {q['question']}
Options:
{options_text(q)}

Reply in JSON: {{"answer": "<option letter>"}}"""
    return letter(ask_llm(prompt).get("answer"))


# ---------- 5. The decision (the core of EduEviGate) ----------
def decide(evidence, memory_answer):
    if evidence["sufficient"] != "yes" or evidence["evidence_answer"] == "none":
        return "Insufficient", "Abstain", "n/a"
    if evidence["evidence_answer"] != memory_answer:
        return "Conflicting", "Report the conflict", "Yes"
    return "Supported", "Answer", "No"


def student_message(q, state, evidence, memory_answer):
    opts = get_options(q)
    if state == "Insufficient":
        return ("The course material doesn't contain enough information to answer this, "
                "so I can't give a supported answer.")
    ev = evidence["evidence_answer"]
    if state == "Conflicting":
        return (f"Your course material says {ev}) {opts.get(ev, '')} (passage {evidence['passage_id']}). "
                f"This differs from widely accepted knowledge, which says {memory_answer}) "
                f"{opts.get(memory_answer, '')}. Follow your course material, and check with your teacher.")
    return f"{ev}) {opts.get(ev, '')} (source: passage {evidence['passage_id']})."


# ---------- 6. Run everything for one question ----------
def run_question(q):
    found = retrieve(q["question"])
    rag_answer, rag_explanation = conventional_rag(q, found)
    evidence = check_evidence(q, found)
    memory_answer = closed_book(q)
    state, action, conflict = decide(evidence, memory_answer)
    return {
        "question": q,
        "found": found,
        "rag_answer": rag_answer,
        "rag_explanation": rag_explanation,
        "evidence": evidence,
        "memory_answer": memory_answer,
        "conflict": conflict,
        "state": state,
        "action": action,
        "message": student_message(q, state, evidence, memory_answer),
    }


def report(r):
    """Turn one result into readable text (used by run_demo.py and app.py)."""
    q, ev, opts = r["question"], r["evidence"], get_options(r["question"])
    lines = [
        f"QUESTION {q['question_id']}: {q['question']}",
        options_text(q),
        "",
        "Retrieved passages: " + ", ".join(f"{p['id']} (score {p['score']})" for p in r["found"]),
        "",
        f"Conventional RAG answered: {r['rag_answer']}) {opts.get(r['rag_answer'], '')}",
        f"   its explanation: {r['rag_explanation']}",
        "",
        "EduEviGate checks:",
        f"   Relevance    : {ev['relevant']}",
        f"   Sufficiency  : {ev['sufficient']}  (model picked: {ev['evidence_answer']}, "
        f"found in quote: {'yes' if ev['sufficient'] == 'yes' else 'no'})",
        f"   Memory answer: {r['memory_answer']}) {opts.get(r['memory_answer'], '')}",
        f"   Conflict     : {r['conflict']}",
        f"   Evidence state: {r['state']}",
        f"   Action       : {r['action']}",
        f"   Judge's reason: {ev['reason']}",
        f"   Quoted evidence: {ev.get('quote', '')}",
        "",
        f"Message to student: {r['message']}",
    ]
    expected = q.get("expected_state", "")
    if expected:
        mark = "CORRECT" if expected.lower() == r["state"].lower() else "WRONG"
        lines.append(f"Expected state: {expected} -> {mark}")
    return "\n".join(lines)

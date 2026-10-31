"""Turns results_<model>.csv files into the summary table.

Usage:
    python summary.py                    (every results_*.csv in the folder, e.g. Llama and Qwen)
    python summary.py results_x.csv      (one file)

The three systems are compared on the SAME model calls:
  Conventional RAG   = always answers (rag_answer)
  Sufficiency-only   = answers if the evidence check says sufficient, otherwise abstains
                       (Baseline 2: it can never report a conflict)
  EduEviGate         = Answer / Abstain / Conflict-aware answer (state)
"""
import glob
import sys

import pandas as pd


def pct(part, whole):
    return f"{100 * part / whole:.0f}% ({part}/{whole})" if whole else "-"


def summarize(df):
    df = df.fillna("")
    conf = df[df.expected_state == "Conflicting"]
    insuf = df[df.expected_state == "Insufficient"]
    nc, ni = len(conf), len(insuf)

    rag_answered_c = conf.rag_answer.isin(list("ABCD"))
    suff_c = conf.sufficient == "yes"
    edu_answered_c = conf.state.isin(["Conflicting", "Supported"])

    rows = [
        ["Conflicting cases", "", "", ""],
        ["  follows the textbook (picks the shifted answer)",
         pct((conf.rag_answer == conf.expected_option).sum(), nc),
         pct((suff_c & (conf.evidence_answer == conf.expected_option)).sum(), nc),
         pct((edu_answered_c & (conf.evidence_answer == conf.expected_option)).sum(), nc)],
        ["  tells the student about the conflict",
         pct(0, nc), pct(0, nc), pct((conf.state == "Conflicting").sum(), nc)],
        ["  wrongly abstains (over-abstention)",
         pct((~rag_answered_c).sum(), nc), pct((~suff_c).sum(), nc),
         pct((conf.state == "Insufficient").sum(), nc)],
        ["Insufficient cases", "", "", ""],
        ["  answers anyway (unsupported answer)",
         pct(insuf.rag_answer.isin(list("ABCD")).sum(), ni),
         pct((insuf.sufficient == "yes").sum(), ni),
         pct((insuf.state != "Insufficient").sum(), ni)],
        ["  correctly abstains",
         pct((~insuf.rag_answer.isin(list("ABCD"))).sum(), ni),
         pct((insuf.sufficient != "yes").sum(), ni),
         pct((insuf.state == "Insufficient").sum(), ni)],
        ["All cases", "", "", ""],
        ["  correct evidence state (EduEviGate only)", "-", "-",
         pct((df.state == df.expected_state).sum(), len(df))],
    ]
    table = pd.DataFrame(rows, columns=["Measure", "Conventional RAG", "Sufficiency-only", "EduEviGate"])

    notes = []
    if nc:
        believes_shift = (conf.memory_answer == conf.expected_option).sum()
        notes.append(f"Closed-book answer already equals the shifted answer in "
                     f"{pct(believes_shift, nc)} of Conflicting cases "
                     f"(EduEviGate then sees no conflict).")
    if "seconds" in df and len(df):
        notes.append(f"Average time per question: {pd.to_numeric(df.seconds).mean():.0f} s")
    model = df.model.iloc[0] if "model" in df and len(df) else "?"
    return f"MODEL: {model}  ({len(df)} cases)\n" + table.to_string(index=False) + "\n" + "\n".join(notes)


if __name__ == "__main__":
    files = sys.argv[1:] or sorted(glob.glob("results_*.csv"))
    for f in files:
        print(summarize(pd.read_csv(f, dtype=str)))
        print()

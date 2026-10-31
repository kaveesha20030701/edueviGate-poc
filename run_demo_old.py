
import pandas as pd

from edueviGate import questions, report, run_question

rows = []
for _, q in questions.iterrows():
    print("=" * 70)
    print("Working... (this can take a minute per question)")
    r = run_question(q)
    print(report(r))
    rows.append({
        "question_id": q["question_id"],
        "expected_state": q.get("expected_state", ""),
        "conventional_rag_answer": r["rag_answer"],
        "relevance": r["evidence"]["relevant"],
        "sufficiency": r["evidence"]["sufficient"],
        "evidence_answer": r["evidence"]["evidence_answer"],
        "memory_answer": r["memory_answer"],
        "conflict": r["conflict"],
        "evidence_state": r["state"],
        "action": r["action"],
    })

print("=" * 70)
table = pd.DataFrame(rows)
print(table.to_string(index=False))
table.to_csv("results.csv", index=False)
print("\nSaved results.csv")

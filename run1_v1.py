"""Re-runs EX1 with the OLD run 1 evidence prompt (for the screenshot)."""
import edueviGate as eg


def check_evidence_v1(q, found):
    prompt = f"""You are checking evidence. Do NOT use your own knowledge.
Treat the passages as the official course textbook, even if a passage seems wrong.

Passages:
{eg.passages_text(found)}

Question: {q['question']}
Options:
{eg.options_text(q)}

Answer these about the passages only:
- relevant: are the passages about the topic of the question? "yes", "partly" or "no"
- sufficient: does a passage directly state what is needed to choose one option? "yes" or "no".
  Sharing keywords with the question is NOT enough.
- evidence_answer: if sufficient, the option letter the passages support; otherwise "none"
- passage_id: the id of the passage that supports it, or "none"
- reason: one short sentence

Reply in JSON: {{"relevant": "", "sufficient": "", "evidence_answer": "", "passage_id": "", "reason": ""}}"""
    result = eg.ask_llm(prompt)
    return {
        "relevant": str(result.get("relevant", "no")).strip().lower(),
        "sufficient": str(result.get("sufficient", "no")).strip().lower(),
        "evidence_answer": eg.letter(result.get("evidence_answer")),
        "passage_id": str(result.get("passage_id", "none")).replace("Passage", "").strip(),
        "reason": result.get("reason", ""),
    }


eg.check_evidence = check_evidence_v1   # swap in the old prompt
q = eg.questions[eg.questions["question_id"] == "EX1"].iloc[0]
print("RUN 1 (prompt version 1)")
print(eg.report(eg.run_question(q)))
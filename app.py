"""Optional demo web page. Run: python app.py  then open the link it prints."""
import gradio as gr

from edueviGate import questions, report, run_question

choices = [f"{q['question_id']}: {q['question']}" for _, q in questions.iterrows()]


def run(choice):
    qid = choice.split(":")[0]
    q = questions[questions["question_id"] == qid].iloc[0]
    return report(run_question(q))


demo = gr.Interface(
    fn=run,
    inputs=gr.Dropdown(choices, label="Pick a question"),
    outputs=gr.Textbox(label="What happened", lines=25),
    title="EduEviGate proof of concept",
    description="Conventional RAG vs EduEviGate on the same question and passages.",
)
demo.launch()

"""Patch v4: the evidence answer is read from the quote by code (run inside the project folder)."""
import sys
p = sys.argv[1] if len(sys.argv) > 1 else "edueviGate.py"
raw = open(p, newline="").read()
crlf = "\r\n" in raw
s = raw.replace("\r\n", "\n")
old_fn = s[s.index("def option_in_quote("):s.index("def check_evidence(")]
new_fn = '''def option_in_quote(option_text, quote):
    """True if ALL key words of the option appear in the quote (stricter than v3)."""
    words = [w for w in tokenize(option_text) if w not in STOPWORDS]
    quote_words = set(tokenize(quote))
    return bool(words) and all(w in quote_words for w in words)


def options_named_in_quote(q, quote):
    """Option letters whose full text appears in the quote, e.g. 'aluminum oxide'."""
    text = " " + " ".join(tokenize(quote)) + " "
    return [L for L, opt in get_options(q).items()
            if tokenize(opt) and " " + " ".join(tokenize(opt)) + " " in text]


def answer_from_quote(q, quote, model_letter):
    """v4: the evidence answer comes from the QUOTE, not from the model's letter.

    - exactly one option named in the quote -> that option (even if the model picked another)
    - several options named -> the model's letter, only if it is one of them
    - no option named in full -> the model's letter, only if all its key words are in the quote
    """
    named = options_named_in_quote(q, quote)
    if len(named) == 1:
        return named[0]
    if len(named) > 1:
        return model_letter if model_letter in named else "none"
    opts = get_options(q)
    if model_letter in opts and option_in_quote(opts[model_letter], quote):
        return model_letter
    return "none"


'''
s = s.replace(old_fn, new_fn)
old = '''    answer = letter(result.get("evidence_answer"))
    found_quote = quote.lower() not in ("", "none")
    opts = get_options(q)
    grounded = found_quote and answer in opts and option_in_quote(opts[answer], quote)
'''
new = '''    model_letter = letter(result.get("evidence_answer"))
    found_quote = quote.lower() not in ("", "none")
    answer = answer_from_quote(q, quote, model_letter) if found_quote else "none"
    grounded = answer != "none"
'''
assert s.count(old) == 1
s = s.replace(old, new)
old2 = '''        "evidence_answer": answer,
'''
new2 = '''        "evidence_answer": answer,
        "model_letter": model_letter,
'''
assert s.count(old2) == 1
s = s.replace(old2, new2)
old3 = '''(model picked: {ev['evidence_answer']}, "
        f"found in quote: {'yes' if ev['sufficient'] == 'yes' else 'no'})",'''
new3 = '''(model picked: {ev.get('model_letter', '')}, "
        f"answer read from quote: {ev['evidence_answer']})",'''
if s.count(old3) == 1:
    s = s.replace(old3, new3)
if crlf:
    s = s.replace("\n", "\r\n")
open(p, "w", newline="").write(s)
print("patched to v4")

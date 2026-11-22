"""
Self-contained Dialectical & Benchmark Utilities (`ideas.dsd`)
==============================================================
Provides prompt formatting, answer extraction, grading, and dataset path `D`
without any external dependencies outside this repository.
"""
import re
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "data"

def chat(tok, messages, enable_thinking=False):
    """Applies chat template with thinking disabled by default for Qwen3 / Llama compatibility."""
    if getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
            )
        except TypeError:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant: "

def thesis_prompt(item):
    """Formats a single-turn reasoning question with standardized final answer tag."""
    return (
        f"{item['question']}\n\n"
        "Explain your reasoning step by step, and end with the line:\n"
        "_final_answer_: (<your answer>)"
    )

def final(text):
    """Extracts the final answer string inside `_final_answer_: (...)`."""
    if not text:
        return ""
    idx = text.rfind("_final_answer_:")
    sub = text[idx:] if idx >= 0 else text
    m = re.search(r"\(([^()]{1,120})\)", sub)
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in sub.splitlines() if l.strip()]
    return lines[-1] if lines else ""

def _norm(s):
    s = str(s).strip().lower()
    s = re.sub(r"^['\"`]+|['\"`]+$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def correct(item, pred):
    """Grades `pred` against `item['answer']` across num, mcq, bool, and value graders."""
    gold = _norm(item.get("answer", item.get("gold_answer", "")))
    p = _norm(pred)
    if not p:
        return False
    if p == gold:
        return True
    grader = item.get("grader", "")
    if grader == "num":
        try:
            return abs(float(p.replace(",", "")) - float(gold.replace(",", ""))) <= 1e-4
        except ValueError:
            return p == gold
    if grader == "mcq":
        m = re.match(r"^([a-z])\b", p)
        return (m.group(1) == gold) if m else (p == gold)
    if grader == "bool":
        if "yes" in p or "true" in p:
            return gold in ("yes", "true")
        if "no" in p or "false" in p:
            return gold in ("no", "false")
    return p == gold

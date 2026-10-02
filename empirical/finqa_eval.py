"""Shared code for the FinQA experiments (seminar paper: Benchmarks for Financial LLM Agents).

Used by 01_run_experiments.ipynb (runs the models) and 02_analysis.ipynb (scores and compares).

Two settings are evaluated:
  - "cot":  the model reasons step by step and ends with "Answer: <number>"
  - "tool": the model writes Python code, we execute it and feed back errors (max. 1 repair round)
"""

import json
import math
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

FINQA_URL = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json"
OLLAMA_URL = "http://localhost:11434"

# FinBen's instruction for FinQA (FinBen paper, Table 7), reused so the prompts stay comparable.
FINBEN_INSTRUCTION = "Given the financial data and expert analysis, please answer this question:"

SYSTEM_PROMPT = "You are a careful financial analyst. You answer questions about company financial reports."

COT_TASK = (
    "Think step by step and show the calculation. "
    "End your response with a final line of the form 'Answer: <number>'. "
    "Give percentages with a % sign (e.g. 'Answer: 12.5%'). "
    "For yes/no questions end with 'Answer: yes' or 'Answer: no'."
)

TOOL_TASK = (
    "Do not compute the result in your head. Instead, write a short Python program that computes it. "
    "Copy the needed numbers from the report into variables, do the calculation, and print ONLY the final "
    "result with print(). Give percentages as a number with a % sign, e.g. print(f'{x:.2f}%'). "
    "For yes/no questions print 'yes' or 'no'. "
    "Return exactly one code block in the format ```python ... ```."
)

REPAIR_TASK = (
    "Executing your code failed with the following error:\n{error}\n\n"
    "Fix the code. Return exactly one corrected code block in the format ```python ... ```."
)


# ---------------------------------------------------------------- data

def load_finqa(data_dir="data"):
    """Load the FinQA test set (1,147 questions, the split FinBen uses). Downloads it once."""
    path = Path(data_dir) / "finqa_test.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(FINQA_URL, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    raw = json.loads(path.read_text())
    return [
        {
            "id": e["id"],
            "question": e["qa"]["question"],
            "gold_answer": e["qa"]["answer"],   # human-written, rounded (e.g. "14%")
            "gold_exe": e["qa"]["exe_ans"],     # exact program result (e.g. 0.14464) or "yes"/"no"
            "gold_program": e["qa"]["program"],
            "pre_text": e["pre_text"],
            "table": e["table"],
            "post_text": e["post_text"],
        }
        for e in raw
    ]


def sample_items(items, n, seed=42):
    """Fixed random sample, so every model and setting sees the same questions."""
    return random.Random(seed).sample(items, n)


def format_context(item):
    table = "\n".join(" | ".join(cell.strip() for cell in row) for row in item["table"])
    return (
        "Text before the table:\n" + " ".join(item["pre_text"]) + "\n\n"
        "Table:\n" + table + "\n\n"
        "Text after the table:\n" + " ".join(item["post_text"])
    )


def build_prompt(item, setting):
    task = COT_TASK if setting == "cot" else TOOL_TASK
    return (
        f"{FINBEN_INSTRUCTION}\n\n{format_context(item)}\n\n"
        f"Question: {item['question']}\n\n{task}"
    )


# ---------------------------------------------------------------- model calls

def ollama_chat(model, messages, num_ctx=4096, num_predict=1024, seed=42, think=False):
    """One deterministic chat call (temperature 0). Returns the answer text and Ollama's timing info."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": {"temperature": 0, "seed": seed, "num_ctx": num_ctx, "num_predict": num_predict},
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=900)
    if r.status_code == 400 and "think" in r.text:
        # Models without a thinking mode can reject the parameter; they never think anyway.
        payload.pop("think")
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=900)
    r.raise_for_status()
    out = r.json()
    return out["message"]["content"], {
        "prompt_tokens": out.get("prompt_eval_count"),
        "output_tokens": out.get("eval_count"),
        "seconds": out.get("total_duration", 0) / 1e9,
    }


# ---------------------------------------------------------------- tool setting

def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    return blocks[-1] if blocks else None


def run_python(code, timeout=10):
    """Run model-written code in a separate Python process. Returns (stdout, error or None).

    Note: this executes code generated by the model on your machine. It runs in a subprocess
    with a timeout, which is fine for FinQA arithmetic, but it is not a security sandbox.
    """
    with tempfile.TemporaryDirectory() as tmp:
        try:
            p = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp,
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return "", f"Timeout after {timeout} s"
    if p.returncode != 0:
        return p.stdout, p.stderr.strip().splitlines()[-1] if p.stderr.strip() else "unknown error"
    if not p.stdout.strip():
        return "", "The code printed nothing. Print the final result."
    return p.stdout, None


# ---------------------------------------------------------------- one question

def solve(item, model, setting, **chat_kwargs):
    """Answer one question in the given setting. Returns a result record (saved as one JSONL line)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(item, setting)}]
    text, stats = ollama_chat(model, messages, **chat_kwargs)
    calls = [stats]
    record = {"id": item["id"], "model": model, "setting": setting, "response": text}

    if setting == "cot":
        record["final"] = extract_final_answer(text)
    else:
        code = extract_code(text)
        stdout, error = run_python(code) if code else ("", "No ```python code block found.")
        record["n_repairs"] = 0
        if error:  # one repair round: the model sees the error and may fix its code
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": REPAIR_TASK.format(error=error)}]
            text2, stats2 = ollama_chat(model, messages, **chat_kwargs)
            calls.append(stats2)
            record["response_repair"] = text2
            record["n_repairs"] = 1
            code = extract_code(text2)
            stdout, error = run_python(code) if code else ("", "No ```python code block found.")
        record.update(code=code, stdout=stdout, exec_error=error)
        lines = stdout.strip().splitlines()
        record["final"] = lines[-1].strip() if lines and not error else None

    record["seconds"] = sum(c["seconds"] for c in calls)
    record["prompt_tokens"] = calls[0]["prompt_tokens"]
    record["output_tokens"] = sum(c["output_tokens"] or 0 for c in calls)
    record["truncated_prompt"] = (calls[0]["prompt_tokens"] or 0) >= chat_kwargs.get("num_ctx", 4096) - 8
    return record


def results_path(results_dir, model, setting):
    return Path(results_dir) / f"{model.replace(':', '_').replace('/', '_')}__{setting}.jsonl"


def run_experiment(items, model, setting, results_dir="results", **chat_kwargs):
    """Answer all items and append each result to a JSONL file.

    Questions already in the file are skipped, so the run can be interrupted and restarted
    at any time without losing work.
    """
    path = results_path(results_dir, model, setting)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        done = {json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()}
    todo = [it for it in items if it["id"] not in done]
    print(f"{model} / {setting}: {len(done)} done, {len(todo)} to go")
    start = time.time()
    for i, item in enumerate(todo, 1):
        try:
            rec = solve(item, model, setting, **chat_kwargs)
        except requests.RequestException as e:
            print(f"  request failed for {item['id']}: {e}. Stopping; rerun to continue.")
            break
        with path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if i % 10 == 0 or i == len(todo):
            per_q = (time.time() - start) / i
            print(f"  {i}/{len(todo)}  {per_q:.1f} s/question  ~{per_q * (len(todo) - i) / 60:.0f} min left")
    return path


# ---------------------------------------------------------------- answer parsing and scoring

NUMBER_RE = re.compile(r"\(?-?\$?\s*-?\d[\d,]*\.?\d*\)?\s*%?")


def extract_final_answer(text):
    """The text after the last 'Answer:' (or None if the model did not follow the format)."""
    matches = re.findall(r"answer\s*[:：]\s*(.+)", text, flags=re.I)
    return matches[-1].strip().strip("*").strip() if matches else None


def parse_number(s):
    """Parse a final answer like '$1,234.5', '-4.4%', '(35)', '12.5 %' into (value, is_percent).

    Returns (None, False) if no number is found; yes/no answers are handled separately.
    """
    if not isinstance(s, str):  # None, or NaN after loading into pandas
        return None, False
    m = NUMBER_RE.search(s.replace("−", "-"))
    if not m:
        return None, False
    tok = m.group(0)
    is_percent = tok.rstrip().endswith("%")
    negative = "-" in tok or (tok.strip().startswith("(") and tok.strip().rstrip("%").endswith(")"))
    digits = re.sub(r"[^\d.]", "", tok).rstrip(".")
    if not digits or digits == ".":
        return None, False
    value = float(digits)
    return (-value if negative else value), is_percent


def parse_yes_no(s):
    if not isinstance(s, str):
        return None
    m = re.search(r"\b(yes|no)\b", s.lower())
    return m.group(1) if m else None


def score_strict(final, item):
    """Strict exact match, close to FinBen's EmAcc: the number as written (no rescaling of
    percentages), rounded to 2 decimals, must equal the exact gold result rounded to 2 decimals."""
    gold = item["gold_exe"]
    if isinstance(gold, str):
        return parse_yes_no(final) == gold
    value, _ = parse_number(final)
    return value is not None and round(value, 2) == round(gold, 2)


def score_tolerant(final, item, rel_tol=0.01):
    """Tolerant numeric match: within 1% of the exact result or of the rounded human answer,
    accepting the percent/decimal scale either way (14% = 0.14 = 14)."""
    gold = item["gold_exe"]
    if isinstance(gold, str):
        return parse_yes_no(final) == gold
    value, is_percent = parse_number(final)
    if value is None:
        return False
    targets = [gold]
    ans_value, ans_percent = parse_number(item["gold_answer"])
    if ans_value is not None:
        targets.append(ans_value / 100 if ans_percent else ans_value)
    candidates = {value, value / 100, value * 100}
    return any(math.isclose(c, t, rel_tol=rel_tol, abs_tol=1e-4) for c in candidates for t in targets)

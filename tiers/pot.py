"""
Program-of-Thought (PoT) module for arithmetic questions.

Instead of asking the LLM to reason in text (CoT), we ask it to write Python
code that extracts the relevant numbers and computes the answer. Python then
executes the code — giving us deterministic, error-free arithmetic.

This is "CoT without CoT": the reasoning happens in code, not in text tokens.
"""

import re
import io
import contextlib

POT_PROMPT = """You are a financial analyst. Solve this step by step.

Evidence (structured key-value facts):
{context}

Question:
{question}

Write Python code in TWO clearly labelled steps:

## Step 1: Extract values
# Write one variable per value you need, with a comment showing exactly where it came from
# Example: revenue_2019 = 48900  # Row: Total revenue, Column: 2019

## Step 2: Compute
# Perform the calculation
# The LAST line MUST be: print(result)

Rules:
- Strip commas and $ signs from numbers (e.g. "$5,048" → 5048)
- For percentage change: result = round(((new - old) / abs(old)) * 100, 2)
- For average: result = round(sum([...]) / n, 2)
- Round to 2 decimal places; if result is a whole number Python will print it as int
- NO imports, NO file I/O — pure arithmetic only

```python
""".strip()

# Restricted builtins for safe execution
_SAFE_GLOBALS = {
    "__builtins__": {
        "print": print,
        "round": round,
        "abs": abs,
        "max": max,
        "min": min,
        "sum": sum,
        "len": len,
        "int": int,
        "float": float,
        "str": str,
        "range": range,
        "True": True,
        "False": False,
        "None": None,
        "pow": pow,
        "divmod": divmod,
    }
}

_FORBIDDEN = ["import ", "__", "open(", "eval(", "exec(", "compile(",
              "globals(", "locals(", "getattr(", "setattr(", "delattr("]


def _extract_code(raw_text):
    """Pull Python code out of a fenced code block."""
    m = re.search(r"```python\s*(.*?)```", raw_text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", raw_text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw_text.strip()


def _safe_exec(code):
    """Execute code in a restricted sandbox. Returns printed output or None on failure."""
    for bad in _FORBIDDEN:
        if bad in code:
            return None

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, dict(_SAFE_GLOBALS))  # pylint: disable=exec-used
        output = buf.getvalue().strip()
        return output if output else None
    except Exception:
        return None


def _normalize_result(result_str):
    """Strip trailing .0 from whole numbers; keep 2dp for decimals."""
    try:
        val = float(result_str.replace(",", "").strip())
        if val == int(val) and abs(val) < 1e12:
            return str(int(val))
        return f"{val:.2f}"
    except (ValueError, OverflowError):
        return result_str.strip()


def extract_structured_facts(retrieved):
    """
    Convert retrieved chunks into clean key-value facts for PoT.
    Table cells → "Row (Column): Value" — eliminates pipe-delimited noise.
    Text chunks → raw text (truncated).
    """
    facts = []
    seen = set()
    for chunk in retrieved:
        ctype = chunk.get("chunk_type", "text")
        if ctype == "table_cell":
            row = (chunk.get("row_label") or "").strip()
            col = (chunk.get("column_header") or "").strip()
            val = (chunk.get("value") or "").strip()
            sec = (chunk.get("section_label") or "").strip()
            if row and col and val:
                key = f"{row}||{col}"
                if key not in seen:
                    seen.add(key)
                    label = f"{row} [{sec}]" if sec else row
                    facts.append(f"{label} ({col}): {val}")
        elif ctype in ("table_row", "table_section"):
            pass  # table_cell captures same data more cleanly
        else:
            text = chunk.get("text", "").strip()[:300]
            if text and text not in seen:
                seen.add(text)
                facts.append(text)
    return "\n".join(facts[:35])


def pot_rag(question, retrieved, client, BASE_MODEL):
    """
    Program-of-Thought arithmetic solver.

    Returns:
        (answer_str, latency, tokens) — answer is None if code fails to execute.
    """
    from tiers.llm_utils import ask_llm

    context = extract_structured_facts(retrieved)
    prompt = POT_PROMPT.format(context=context, question=question)

    out = ask_llm(prompt, client=client, model=BASE_MODEL, temperature=0.0, max_tokens=500)
    code = _extract_code(out["text"])
    raw_result = _safe_exec(code)

    if raw_result is None:
        return None, out["latency"], out["tokens"]

    answer = _normalize_result(raw_result)
    return answer, out["latency"], out["tokens"]

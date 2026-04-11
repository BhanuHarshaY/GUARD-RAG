import re
from tiers.llm_utils import ask_llm, format_retrieved_context


_ARITHMETIC_PATTERNS = re.compile(
    r"\b("
    r"difference between"
    r"|difference in"
    r"|percentage change"
    r"|percent change"
    r"|% change"
    r"|how much more"
    r"|how much less"
    r"|what is the change"
    r"|what was the change"
    r"|ratio of"
    r"|sum of"
    r"|increased from"
    r"|decreased from"
    r"|grew by"
    r"|fell by"
    r"|average of"
    r"|what is the average"
    r"|what was the average"
    r"|calculate"
    r")\b",
    re.IGNORECASE,
)

# Separate pattern for increase/(decrease) style — word boundary doesn't work with slash
_INCREASE_DECREASE_PAT = re.compile(r"increase\s*/\s*\(", re.IGNORECASE)


def is_arithmetic_question(question):
    return bool(_ARITHMETIC_PATTERNS.search(question) or _INCREASE_DECREASE_PAT.search(question))


def _extract_final_answer(text):
    """Strip CoT reasoning — return only the final numeric result."""
    # 1. Explicit 'Final answer:' label — take the last occurrence
    matches = list(re.finditer(r"Final answer\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE))
    if matches:
        return matches[-1].group(1).strip()

    # 2. Expression ending in '= <number>[%]' e.g. "2.9 - 2.7 = 0.2" → "0.2"
    eq_match = re.search(r"=\s*([-+]?[\d,]+\.?\d*)[%]?\s*$", text.strip(), re.MULTILINE)
    if eq_match:
        return eq_match.group(1).strip()

    # 3. Fallback: last non-empty line
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else text.strip()


def baseline_rag(question, retrieved, client, BASE_MODEL):
    context = format_retrieved_context(retrieved)

    if is_arithmetic_question(question):
        prompt = f"""You are answering a financial question that requires calculation.

Evidence:
{context}

Question:
{question}

Work through this carefully:
Step 1 - Extract values: identify the exact numbers needed from the evidence.
Step 2 - Calculate: show the arithmetic explicitly.
Step 3 - Final answer: state ONLY the final numeric result. No units, no formula, no explanation — just the number.

Final answer:""".strip()
    else:
        prompt = f"""You are answering a question using ONLY the provided evidence.

Rules:
1. Use only facts explicitly present in the evidence.
2. If the question asks what something consists of / includes / contains, list the supported components explicitly.
3. If the evidence is partial, give the supported partial answer rather than saying "Insufficient information."
4. Answer with ONLY the value(s) or fact(s) asked for. No explanations, no reasoning steps, no sentences like "Based on the evidence" or "According to the data". Just the answer.
5. Do not mention document IDs or retrieval process.

Evidence:
{context}

Question:
{question}

Answer:""".strip()

    max_tokens = 400 if is_arithmetic_question(question) else 220
    out = ask_llm(prompt, client=client, model=BASE_MODEL, temperature=0.0, max_tokens=max_tokens)
    answer = _extract_final_answer(out["text"]) if is_arithmetic_question(question) else out["text"]

    return {
        "answer": answer,
        "latency": out["latency"],
        "tokens": out["tokens"],
    }

import re
from tiers.llm_utils import ask_llm, format_retrieved_context, strip_refinement_prefix
from tiers.baseline import baseline_rag, is_arithmetic_question, _extract_final_answer


def _numeric_magnitude_check(draft, refined, max_ratio=100):
    """Return True if refined is within max_ratio× of draft (sanity check for arithmetic)."""
    def _extract_num(s):
        s = str(s).replace(",", "").replace("(", "-").replace(")", "")
        m = re.search(r"-?\d+\.?\d*", s)
        return float(m.group()) if m else None

    d, r = _extract_num(draft), _extract_num(refined)
    if d is None or r is None:
        return True  # can't check, allow it
    if r == 0:
        return True
    ratio = abs(d / r) if r != 0 else float("inf")
    return ratio <= max_ratio and (1 / ratio) <= max_ratio


def refinement_rag(question, retrieved, client, BASE_MODEL, baseline_result=None):
    """
    Tier 2: Self-refine — two sequential calls.
    Call 1 (Baseline): get a draft answer.
    Call 2 (Verify): re-derive for arithmetic; ultra-conservative fact-check for text.
    Strong prior toward keeping the draft — only overrides on explicit contradiction.
    """
    if baseline_result is None:
        baseline_result = baseline_rag(question, retrieved, client, BASE_MODEL)

    draft = baseline_result["answer"]
    context = format_retrieved_context(retrieved)
    is_arith = is_arithmetic_question(question)

    if is_arith:
        prompt = f"""You are checking a financial calculation for common formula mistakes.

Common errors to catch:
- Sum instead of average: "What is the average?" → must divide by count, not just add
- Wrong direction: percentage change = (new - old) / old × 100, NOT (old - new) / old
- Wrong year: make sure the value matches the year asked in the question
- Off-by-one: check the draft used the exact row/column the question asked for

Example 1 (sum vs average error):
Evidence: Row: Cloud revenue | Column: 2019 | Value: 9,200 ; Column: 2018 | Value: 8,902
Question: What is the average of Cloud revenue in 2019 and 2018?
Draft: 18102
Check: question asks AVERAGE not sum → (9200 + 8902) / 2 = 9051. Draft is WRONG.
Final answer: 9051

Example 2 (wrong direction error):
Evidence: Row: Operating expenses | Column: 2019 | Value: 3,200 ; Column: 2018 | Value: 2,800
Question: What is the percentage change in operating expenses from 2018 to 2019?
Draft: -12.5
Check: percentage change = (3200 - 2800) / 2800 × 100 = 14.3. Draft divided wrong way.
Final answer: 14.3

Example 3 (correct draft):
Evidence: Row: Net revenues | Column: 2019 | Value: 48,900 ; Column: 2018 | Value: 42,300
Question: What was the percentage change in net revenues from 2018 to 2019?
Draft: 15.6
Check: (48900 - 42300) / 42300 × 100 = 15.6. Draft is correct.
Final answer: 15.6

Now check this:

Evidence:
{context}

Question:
{question}

Draft: {draft}

Check for formula errors. If draft is correct return it exactly. If wrong return only the correct number.
Final answer:""".strip()
        max_tokens = 300

    else:
        prompt = f"""You are fact-checking a draft answer. Assume the draft is correct unless you find clear proof otherwise.

Evidence:
{context}

Question:
{question}

Draft answer: {draft}

- If the draft is supported by the evidence (even partially): return it EXACTLY as-is, word for word.
- ONLY return a different answer if the evidence explicitly states a specific different value that directly contradicts the draft.
- When in doubt: return the draft exactly as-is.

Verified answer:""".strip()
        max_tokens = 150

    out = ask_llm(prompt, client=client, model=BASE_MODEL, temperature=0.0, max_tokens=max_tokens)

    final_answer = strip_refinement_prefix(out["text"])
    if is_arith:
        final_answer = _extract_final_answer(final_answer)

    if not final_answer.strip():
        final_answer = draft

    if is_arith and not _numeric_magnitude_check(draft, final_answer):
        final_answer = draft

    return {
        "initial_answer": draft,
        "answer": final_answer,
        "latency": round(baseline_result["latency"] + out["latency"], 3),
        "tokens": baseline_result["tokens"] + out["tokens"],
    }

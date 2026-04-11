from tiers.llm_utils import ask_llm, format_retrieved_context, strip_refinement_prefix
from tiers.baseline import baseline_rag, is_arithmetic_question, _extract_final_answer


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
        prompt = f"""You are verifying a financial calculation.

Evidence:
{context}

Question:
{question}

A previous attempt answered: {draft}

Re-derive the answer step by step using ONLY the evidence values.
- If your calculation matches the draft: return the draft EXACTLY as-is.
- If your calculation gives a clearly different number: return ONLY that number.

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

    return {
        "initial_answer": draft,
        "answer": final_answer,
        "latency": round(baseline_result["latency"] + out["latency"], 3),
        "tokens": baseline_result["tokens"] + out["tokens"],
    }

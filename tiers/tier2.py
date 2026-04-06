from tiers.llm_utils import ask_llm, format_retrieved_context, strip_refinement_prefix
from tiers.tier1 import tier1_basic_rag
from evaluation.metrics import compute_f1


def _majority_vote(answers):
    """Return the answer with highest average pairwise F1 against all others."""
    if len(answers) == 1:
        return answers[0]
    best_idx, best_score = 0, -1.0
    for i, a in enumerate(answers):
        score = sum(compute_f1(a, answers[j]) for j in range(len(answers)) if j != i)
        if score > best_score:
            best_score, best_idx = score, i
    return answers[best_idx]


def tier2_self_refine(question, retrieved, client, BASE_MODEL, tier1_result=None, k=3):  # noqa: tier1_result unused — kept for call-site compatibility
    # --- self-consistency: k samples at temperature=0.7 ---
    context = format_retrieved_context(retrieved)
    sc_prompt = f"""Answer the following question using ONLY the provided evidence.

Question:
{question}

Evidence:
{context}

Return ONLY the answer, no explanation.
""".strip()

    candidates = []
    total_latency = 0.0
    total_tokens = 0

    for _ in range(k):
        out = ask_llm(sc_prompt, client=client, model=BASE_MODEL, temperature=0.3, max_tokens=150)
        candidates.append(strip_refinement_prefix(out["text"]))
        total_latency += out["latency"]
        total_tokens += out["tokens"]

    consensus = _majority_vote(candidates)

    # build a synthetic tier1_result from the consensus so the rest of the
    # function works unchanged whether called standalone or from the pipeline
    tier1_result = {"answer": consensus, "latency": total_latency, "tokens": total_tokens}

    draft_answer = tier1_result["answer"]

    prompt = f"""
You are revising a draft answer using ONLY the provided evidence.

Question:
{question}

Evidence:
{context}

Draft answer:
{draft_answer}

Instructions:
- Remove any claims from the draft answer that are not supported by the evidence.
- If the draft answer is fully supported, keep it.
- If nothing is supported, output exactly "Insufficient information."
- Output ONLY the exact value, name, or list. Do not include conversational filler like 'The answer is' or 'Based on the evidence'.

Final answer:
""".strip()

    out = ask_llm(prompt, client=client, model=BASE_MODEL, temperature=0.0, max_tokens=150)
    final_answer = strip_refinement_prefix(out["text"])

    return {
        "initial_answer": draft_answer,
        "sc_candidates": candidates,
        "answer": final_answer,
        "latency": round(tier1_result["latency"] + out["latency"], 3),
        "tokens": tier1_result["tokens"] + out["tokens"],
    }

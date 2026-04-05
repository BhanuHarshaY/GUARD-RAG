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
        out = ask_llm(sc_prompt, client=client, model=BASE_MODEL, temperature=0.7, max_tokens=220)
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
- Keep all supported parts of the draft.
- Remove unsupported claims.
- If the question is asking for components and the evidence supports only some components, return the supported components instead of falling back to "Insufficient information."
- Do NOT explain your reasoning.
- Return ONLY the corrected final answer.

Final answer:
""".strip()

    out = ask_llm(prompt, client=client, model=BASE_MODEL, temperature=0.0, max_tokens=220)
    final_answer = strip_refinement_prefix(out["text"])

    return {
        "initial_answer": draft_answer,
        "sc_candidates": candidates,
        "answer": final_answer,
        "latency": round(tier1_result["latency"] + out["latency"], 3),
        "tokens": tier1_result["tokens"] + out["tokens"],
    }

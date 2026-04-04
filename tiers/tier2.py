from tiers.llm_utils import ask_llm, format_retrieved_context, strip_refinement_prefix
from tiers.tier1 import tier1_basic_rag


def tier2_self_refine(question, retrieved, client, BASE_MODEL, tier1_result=None):
    if tier1_result is None:
        tier1_result = tier1_basic_rag(question, retrieved, client, BASE_MODEL)

    context = format_retrieved_context(retrieved)
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
        "answer": final_answer,
        "latency": round(tier1_result["latency"] + out["latency"], 3),
        "tokens": tier1_result["tokens"] + out["tokens"],
    }

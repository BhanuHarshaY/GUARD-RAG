import re

from tiers.llm_utils import ask_llm, format_retrieved_context, strip_refinement_prefix
from tiers.tier1 import tier1_basic_rag
from indexing.table_parser import normalize_text_for_match
from retrieval.retriever import question_is_list_like


def clean_for_gatekeeper(text):
    text = str(text)
    text = re.sub(r"\b[0-9a-f]{8}-[0-9a-f\-]{20,}\b", " ", text, flags=re.I)
    text = re.sub(r"\b[0-9a-f]{16,}\b", " ", text, flags=re.I)
    text = re.sub(r"\b[a-z0-9_]*text_\d+\b", " ", text, flags=re.I)
    text = re.sub(r"\b[a-z0-9_]*table_[a-z]+_\d+\b", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_supported_row_labels(retrieved):
    labels = []
    for r in retrieved:
        row_label = r.get("row_label")
        if row_label:
            labels.append(row_label.strip())
    return list(dict.fromkeys(labels))


def gatekeeper_v3(question, retrieved, answer):
    signals = []

    list_like = question_is_list_like(question)
    structured_chunks = [r for r in retrieved if r.get("chunk_type") in {"table_section", "table_cell", "table_row"}]
    row_labels = extract_supported_row_labels(retrieved)

    cleaned_answer = clean_for_gatekeeper(answer)
    answer_norm = normalize_text_for_match(cleaned_answer)

    if structured_chunks and list_like:
        mentioned_rows = 0
        for label in row_labels:
            label_norm = normalize_text_for_match(label)
            if label_norm and label_norm in answer_norm:
                mentioned_rows += 1

        if "insufficient information" in answer_norm and len(row_labels) >= 1:
            signals.append("Structured evidence exists but answer fell back to insufficient information")

        if len(row_labels) >= 2 and mentioned_rows < min(2, len(row_labels)):
            signals.append("Possible missed structured evidence")

    if len(cleaned_answer.split()) > 80:
        signals.append("Answer suspiciously long relative to evidence")

    return signals


def tier3_selective_debate(question, retrieved, client, BASE_MODEL, JUDGE_MODEL, tier1_result=None):
    if tier1_result is None:
        tier1_result = tier1_basic_rag(question, retrieved, client, BASE_MODEL)

    draft_answer = tier1_result["answer"]
    signals = gatekeeper_v3(question, retrieved, draft_answer)

    if not signals:
        return {
            "answer": draft_answer,
            "latency": tier1_result["latency"],
            "tokens": tier1_result["tokens"],
            "debate_triggered": False,
            "gatekeeper_signals": [],
        }

    context = format_retrieved_context(retrieved, max_chars=4500)

    prompt = f"""
You are a careful verifier answering ONLY from the evidence below.

Question:
{question}

Evidence:
{context}

Draft answer:
{draft_answer}

Concerns raised:
{chr(10).join("- " + s for s in signals)}

Instructions:
- Use only the evidence.
- Prefer precision over verbosity.
- If the question is asking for components, list the supported components explicitly.
- If the evidence is partial, give the supported partial answer.
- Do NOT mention the concerns or reasoning.
- Return ONLY the final answer.

Final answer:
""".strip()

    out = ask_llm(prompt, client=client, model=JUDGE_MODEL, temperature=0.0, max_tokens=260)
    final_answer = strip_refinement_prefix(out["text"])

    return {
        "answer": final_answer,
        "latency": round(tier1_result["latency"] + out["latency"], 3),
        "tokens": tier1_result["tokens"] + out["tokens"],
        "debate_triggered": True,
        "gatekeeper_signals": signals,
    }

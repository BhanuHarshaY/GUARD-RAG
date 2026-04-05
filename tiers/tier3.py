import re

from tiers.llm_utils import ask_llm, format_retrieved_context, strip_refinement_prefix
from tiers.tier1 import tier1_basic_rag
from indexing.table_parser import normalize_text_for_match
from retrieval.retriever import question_is_list_like

# ---------------------------------------------------------------------------
# Signal weights — tune these during threshold sweeping
# ---------------------------------------------------------------------------
SIGNAL_WEIGHTS = {
    "heuristic_insufficient_info": 0.4,
    "heuristic_missing_rows":      0.3,
    "heuristic_too_long":          0.2,
    "nli_low_grounding":           0.5,
}

# Label order returned by cross-encoder/nli-deberta-v3-small:
# index 0 = contradiction, index 1 = entailment, index 2 = neutral
_NLI_ENTAILMENT_IDX = 1


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


def _split_into_sentences(text):
    """Simple sentence splitter — avoids adding a heavy dependency."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 8]


def nli_grounding_score(answer, retrieved, nli_model):
    """
    Returns the fraction of answer sentences entailed by the retrieved context.
    Uses cross-encoder/nli-deberta-v3-small (labels: contradiction, entailment, neutral).
    Returns 1.0 (fully grounded) when the answer has no scoreable sentences.
    """
    sentences = _split_into_sentences(answer)
    if not sentences:
        return 1.0

    # Cap context to avoid OOM on long retrievals
    context = " ".join(r["text"] for r in retrieved)[:3000]

    pairs = [(context, s) for s in sentences]
    scores = nli_model.predict(pairs)   # shape: (n_sentences, 3)

    entailed = sum(1 for s in scores if s[_NLI_ENTAILMENT_IDX] >= 0.5)
    return entailed / len(sentences)


def gatekeeper_v4(question, retrieved, answer, nli_model=None, threshold=0.20):
    """
    Weighted scoring gatekeeper. Runs all checks (heuristic + NLI) and sums
    their weights into a single gatekeeper_score. Debate triggers when
    gatekeeper_score >= threshold.

    Returns:
        {
            "signals":          list[str]   — fired signal names (for logging/prompt)
            "weights":          dict        — signal_name -> weight for fired signals
            "gatekeeper_score": float       — sum of fired signal weights
            "threshold":        float       — threshold used
            "trigger_debate":   bool        — True when score >= threshold
        }
    """
    fired_signals = {}

    list_like = question_is_list_like(question)
    structured_chunks = [
        r for r in retrieved
        if r.get("chunk_type") in {"table_section", "table_cell", "table_row"}
    ]
    row_labels = extract_supported_row_labels(retrieved)

    cleaned_answer = clean_for_gatekeeper(answer)
    answer_norm = normalize_text_for_match(cleaned_answer)

    # --- Heuristic: insufficient-info fallback ---
    if structured_chunks and list_like:
        if "insufficient information" in answer_norm and len(row_labels) >= 1:
            fired_signals["heuristic_insufficient_info"] = SIGNAL_WEIGHTS["heuristic_insufficient_info"]

    # --- Heuristic: missing row labels ---
    if structured_chunks and list_like and len(row_labels) >= 2:
        mentioned_rows = sum(
            1 for label in row_labels
            if normalize_text_for_match(label) and normalize_text_for_match(label) in answer_norm
        )
        if mentioned_rows < min(2, len(row_labels)):
            fired_signals["heuristic_missing_rows"] = SIGNAL_WEIGHTS["heuristic_missing_rows"]

    # --- Heuristic: suspiciously long answer ---
    if len(cleaned_answer.split()) > 80:
        fired_signals["heuristic_too_long"] = SIGNAL_WEIGHTS["heuristic_too_long"]

    # --- NLI grounding check ---
    if nli_model is not None:
        grounding = nli_grounding_score(cleaned_answer, retrieved, nli_model)
        if grounding < 0.5:
            fired_signals["nli_low_grounding"] = SIGNAL_WEIGHTS["nli_low_grounding"]

    gatekeeper_score = round(sum(fired_signals.values()), 4)

    return {
        "signals":          list(fired_signals.keys()),
        "weights":          fired_signals,
        "gatekeeper_score": gatekeeper_score,
        "threshold":        threshold,
        "trigger_debate":   gatekeeper_score >= threshold,
    }


def tier3_selective_debate(question, retrieved, client, BASE_MODEL, JUDGE_MODEL,
                           tier1_result=None, nli_model=None, threshold=0.45):
    if tier1_result is None:
        tier1_result = tier1_basic_rag(question, retrieved, client, BASE_MODEL)

    draft_answer = tier1_result["answer"]
    gate = gatekeeper_v4(question, retrieved, draft_answer,
                         nli_model=nli_model, threshold=threshold)

    if not gate["trigger_debate"]:
        return {
            "answer":             draft_answer,
            "latency":            tier1_result["latency"],
            "tokens":             tier1_result["tokens"],
            "debate_triggered":   False,
            "gatekeeper_signals": gate["signals"],
            "gatekeeper_score":   gate["gatekeeper_score"],
            "threshold":          gate["threshold"],
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
{chr(10).join("- " + s for s in gate["signals"])}

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
        "answer":             final_answer,
        "latency":            round(tier1_result["latency"] + out["latency"], 3),
        "tokens":             tier1_result["tokens"] + out["tokens"],
        "debate_triggered":   True,
        "gatekeeper_signals": gate["signals"],
        "gatekeeper_score":   gate["gatekeeper_score"],
        "threshold":          gate["threshold"],
    }

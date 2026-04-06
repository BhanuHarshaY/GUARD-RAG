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
    return [p.strip() for p in parts if len(p.strip()) > 1]


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


def _run_skeptic(question, draft_answer, context, client, BASE_MODEL, gatekeeper_signals=None):
    """
    Skeptic agent: identifies claims in the draft answer that are NOT directly
    supported by the retrieved evidence. Returns the raw text output and token count.
    """
    signals_section = ""
    if gatekeeper_signals:
        bullets = "\n".join(f"- {s}" for s in gatekeeper_signals)
        signals_section = f"\nConcerns flagged by automated checks:\n{bullets}\n"

    prompt = f"""You are a Skeptic reviewing a draft answer for unsupported claims.

Question:
{question}

Evidence:
{context}

Draft answer:
{draft_answer}
{signals_section}
Task:
List each specific claim in the draft answer that is NOT directly supported by the evidence above.
- Pay special attention to any concerns flagged above. If missing rows or evidence gaps were flagged, check whether the draft answer omits information that IS present in the evidence.
- Be precise: quote or closely paraphrase the claim.
- Ignore claims that ARE clearly supported.
- If all claims are supported, write: "No unsupported claims found."
- Output ONLY a bullet list of challenged claims. No explanation.

Challenged claims:
""".strip()

    out = ask_llm(prompt, client=client, model=BASE_MODEL, temperature=0.0, max_tokens=300)
    return out["text"].strip(), out["latency"], out["tokens"]


def _verify_grounder_citations(grounder_output, context):
    """
    Post-processing: for each SUPPORTED line, check that the quoted evidence
    actually appears in the context (fuzzy: all words of the quoted phrase
    present in context, case-insensitive). Flip to CONCEDED if not found.
    """
    lines = grounder_output.splitlines()
    verified = []
    for line in lines:
        if line.strip().upper().startswith("SUPPORTED:") and "— Evidence:" in line:
            # extract quoted evidence after "Evidence:"
            evidence_part = line.split("— Evidence:", 1)[1].strip().strip('"').strip("'")
            # fuzzy check: every word of the cited phrase must appear in context
            ctx_lower = context.lower()
            words = [w for w in evidence_part.lower().split() if len(w) > 1]
            if words and not all(w in ctx_lower for w in words):
                # citation not found — flip to CONCEDED
                claim_part = line.split("— Evidence:", 1)[0].replace("SUPPORTED:", "", 1).strip()
                line = f"CONCEDED: {claim_part}"
        verified.append(line)
    return "\n".join(verified)


def _run_grounder(question, draft_answer, skeptic_output, context, client, BASE_MODEL):
    """
    Grounder agent: defends or concedes each challenged claim using only cited evidence.
    Applies post-processing to flip unverifiable citations to CONCEDED.
    Returns the raw text output and token count.
    """
    prompt = f"""You are a Grounder defending a draft answer using ONLY the evidence provided.

Question:
{question}

Evidence:
{context}

Draft answer:
{draft_answer}

Challenged claims (from Skeptic):
{skeptic_output}

Task:
For each challenged claim:
- If the evidence supports it: write "SUPPORTED: <claim> — Evidence: <exact quote or value from evidence>"
- If the evidence does NOT support it: write "CONCEDED: <claim>"

Output ONLY the per-claim verdicts. No extra commentary.

Grounder verdicts:
""".strip()

    out = ask_llm(prompt, client=client, model=BASE_MODEL, temperature=0.0, max_tokens=400)
    raw = out["text"].strip()
    verified = _verify_grounder_citations(raw, context)
    return verified, out["latency"], out["tokens"]


def _parse_adjudicator_verdict(text):
    """
    Parse [APPROVE], [REVISE], or [ABSTAIN] prefix from adjudicator output.
    Returns (verdict_label, answer_text).
    Falls back to inferring from content if no prefix present.
    """
    text = text.strip()
    for label in ("APPROVE", "REVISE", "ABSTAIN"):
        if text.upper().startswith(f"[{label}]"):
            answer = text[len(f"[{label}]"):].strip()
            return label, answer

    # fallback inference
    if "insufficient information" in text.lower():
        return "ABSTAIN", text
    return "REVISE", text


def _regenerate_from_evidence(question, grounder_output, context, client, JUDGE_MODEL):
    """
    Regenerates a clean answer from the Grounder's verified SUPPORTED claims.
    Used on the REVISE path instead of the adjudicator's own patched answer.
    Returns (answer_text, latency, tokens).
    """
    facts = []
    for line in grounder_output.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("SUPPORTED:") and "— Evidence:" in stripped:
            evidence = stripped.split("— Evidence:", 1)[1].strip().strip('"').strip("'")
            if evidence:
                facts.append(evidence)

    if not facts:
        return "Insufficient information.", 0.0, 0

    numbered = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts))

    prompt = f"""Answer the following question using ONLY these verified facts.

Question:
{question}

Verified facts:
{numbered}

Full evidence (for reference):
{context}

Rules:
- Use ONLY the verified facts and full evidence above.
- Return ONLY the direct answer. No explanations, no reasoning.
- If the question asks for multiple values, include all of them.

Answer:
""".strip()

    out = ask_llm(prompt, client=client, model=JUDGE_MODEL, temperature=0.0, max_tokens=150)
    return out["text"].strip(), out["latency"], out["tokens"]


def _run_adjudicator(question, draft_answer, skeptic_output, grounder_output,
                     context, client, JUDGE_MODEL):
    """
    Adjudicator: reads the full debate exchange and produces the final answer.
    Prefixes output with [APPROVE], [REVISE], or [ABSTAIN] for logging.
    Returns (final_answer, verdict_label, latency, tokens).
    """
    prompt = f"""You are an Adjudicator producing a final answer after a debate.

Question:
{question}

Evidence:
{context}

Draft answer:
{draft_answer}

Skeptic's challenged claims:
{skeptic_output}

Grounder's verdicts:
{grounder_output}

Instructions:
- If all challenged claims were SUPPORTED by the Grounder: prefix your response with [APPROVE] and return the draft answer.
- If some claims were CONCEDED: prefix your response with [REVISE] and return a corrected answer using ONLY the supported claims and evidence.
- If ALL claims were CONCEDED and nothing is supported: prefix your response with [ABSTAIN] and return exactly "Insufficient information."
- CRITICAL: Return ONLY the direct factual answer. No explanations, no reasoning, no phrases like "Based on the evidence". Just the answer value(s).
- Use only facts from the evidence. Do NOT introduce new claims.
- Format: [VERDICT] <concise answer>

Response:
""".strip()

    out = ask_llm(prompt, client=client, model=JUDGE_MODEL, temperature=0.0, max_tokens=280)
    raw = strip_refinement_prefix(out["text"])
    verdict, final_answer = _parse_adjudicator_verdict(raw)
    return final_answer, verdict, out["latency"], out["tokens"]


def tier3_selective_debate(question, retrieved, client, BASE_MODEL, JUDGE_MODEL,
                           tier1_result=None, nli_model=None, threshold=0.20):
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
            "debate_transcript":  None,
        }

    context = format_retrieved_context(retrieved, max_chars=4500)

    # --- Step 1: Skeptic ---
    skeptic_out, s_lat, s_tok = _run_skeptic(
        question, draft_answer, context, client, BASE_MODEL,
        gatekeeper_signals=gate["signals"]
    )

    # Short-circuit: if Skeptic found nothing to challenge, skip Grounder + Adjudicator
    if "no unsupported claims" in skeptic_out.lower():
        return {
            "answer":             draft_answer,
            "latency":            round(tier1_result["latency"] + s_lat, 3),
            "tokens":             tier1_result["tokens"] + s_tok,
            "debate_triggered":   True,
            "adjudicator_verdict": "APPROVE",
            "gatekeeper_signals": gate["signals"],
            "gatekeeper_score":   gate["gatekeeper_score"],
            "threshold":          gate["threshold"],
            "debate_transcript":  {
                "skeptic":  skeptic_out,
                "grounder": None,
            },
        }

    # --- Step 2: Grounder (with citation verification) ---
    grounder_out, g_lat, g_tok = _run_grounder(
        question, draft_answer, skeptic_out, context, client, BASE_MODEL
    )

    # --- Step 3: Adjudicator ---
    final_answer, verdict, a_lat, a_tok = _run_adjudicator(
        question, draft_answer, skeptic_out, grounder_out, context, client, JUDGE_MODEL
    )

    total_latency = round(tier1_result["latency"] + s_lat + g_lat + a_lat, 3)
    total_tokens  = tier1_result["tokens"] + s_tok + g_tok + a_tok

    # --- Step 4: Regeneration (REVISE path only) ---
    regenerated = False
    if verdict == "REVISE":
        regen_answer, r_lat, r_tok = _regenerate_from_evidence(
            question, grounder_out, context, client, JUDGE_MODEL
        )
        final_answer = regen_answer
        total_latency = round(total_latency + r_lat, 3)
        total_tokens  = total_tokens + r_tok
        regenerated = True

    return {
        "answer":              final_answer,
        "latency":             total_latency,
        "tokens":              total_tokens,
        "debate_triggered":    True,
        "adjudicator_verdict": verdict,          # "APPROVE" | "REVISE" | "ABSTAIN"
        "regenerated":         regenerated,
        "gatekeeper_signals":  gate["signals"],
        "gatekeeper_score":    gate["gatekeeper_score"],
        "threshold":           gate["threshold"],
        "debate_transcript":   {
            "skeptic":  skeptic_out,
            "grounder": grounder_out,
        },
    }

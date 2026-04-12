import re

from tiers.llm_utils import ask_llm, format_retrieved_context, strip_refinement_prefix
from tiers.baseline import baseline_rag, is_arithmetic_question
from indexing.table_parser import normalize_text_for_match
from retrieval.retriever import question_is_list_like, question_is_multispan

# ---------------------------------------------------------------------------
# Signal weights
# ---------------------------------------------------------------------------
SIGNAL_WEIGHTS = {
    "heuristic_insufficient_info": 0.4,
    "heuristic_missing_rows":      0.3,
    "heuristic_too_long":          0.2,
    "nli_low_grounding":           0.3,
    "heuristic_arithmetic":        0.6,
    "heuristic_multispan":         0.5,
}

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
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 1]


def nli_grounding_score(answer, retrieved, nli_model):
    sentences = _split_into_sentences(answer)
    if not sentences:
        return 1.0
    context = " ".join(r["text"] for r in retrieved)[:3000]
    pairs = [(context, s) for s in sentences]
    scores = nli_model.predict(pairs)
    entailed = sum(1 for s in scores if s[_NLI_ENTAILMENT_IDX] >= 0.5)
    return entailed / len(sentences)


def gatekeeper_v4(question, retrieved, answer, nli_model=None, threshold=0.35, disabled_signals=None):
    if disabled_signals is None:
        disabled_signals = set()

    fired_signals = {}

    list_like = question_is_list_like(question)
    structured_chunks = [
        r for r in retrieved
        if r.get("chunk_type") in {"table_section", "table_cell", "table_row"}
    ]
    row_labels = extract_supported_row_labels(retrieved)

    cleaned_answer = clean_for_gatekeeper(answer)
    answer_norm = normalize_text_for_match(cleaned_answer)

    if "heuristic_arithmetic" not in disabled_signals and is_arithmetic_question(question):
        fired_signals["heuristic_arithmetic"] = SIGNAL_WEIGHTS["heuristic_arithmetic"]

    if "heuristic_multispan" not in disabled_signals and question_is_multispan(question):
        fired_signals["heuristic_multispan"] = SIGNAL_WEIGHTS["heuristic_multispan"]

    if "heuristic_insufficient_info" not in disabled_signals and structured_chunks and list_like:
        if "insufficient information" in answer_norm and len(row_labels) >= 1:
            fired_signals["heuristic_insufficient_info"] = SIGNAL_WEIGHTS["heuristic_insufficient_info"]

    if "heuristic_missing_rows" not in disabled_signals and structured_chunks and list_like and len(row_labels) >= 2:
        mentioned_rows = sum(
            1 for label in row_labels
            if normalize_text_for_match(label) and normalize_text_for_match(label) in answer_norm
        )
        if mentioned_rows < min(2, len(row_labels)):
            fired_signals["heuristic_missing_rows"] = SIGNAL_WEIGHTS["heuristic_missing_rows"]

    if "heuristic_too_long" not in disabled_signals and len(cleaned_answer.split()) > 80:
        fired_signals["heuristic_too_long"] = SIGNAL_WEIGHTS["heuristic_too_long"]

    if "nli_low_grounding" not in disabled_signals and nli_model is not None:
        grounding = nli_grounding_score(cleaned_answer, retrieved, nli_model)
        if grounding < 0.35:
            fired_signals["nli_low_grounding"] = SIGNAL_WEIGHTS["nli_low_grounding"]

    gatekeeper_score = round(sum(fired_signals.values()), 4)

    return {
        "signals":          list(fired_signals.keys()),
        "weights":          fired_signals,
        "gatekeeper_score": gatekeeper_score,
        "threshold":        threshold,
        "trigger_debate":   gatekeeper_score >= threshold,
    }


def _run_math_skeptic(question, draft_answer, context, client, BASE_MODEL):
    """Math-specialized skeptic for arithmetic questions — re-derives the calculation."""
    prompt = f"""You are verifying a financial calculation.

Evidence:
{context}

Question:
{question}

Draft answer: {draft_answer}

Re-derive the answer step by step using ONLY the values in the evidence:
Step 1 - Extract the exact numbers needed.
Step 2 - Perform the calculation explicitly.
Step 3 - Compare your result to the draft.

Output format (choose one):
- If your result matches the draft: "CORRECT: {draft_answer}"
- If your result differs: "INCORRECT: draft says {draft_answer}, correct answer is <your_result>"

One line only. No extra commentary.
""".strip()

    out = ask_llm(prompt, client=client, model=BASE_MODEL, temperature=0.0, max_tokens=200)
    return out["text"].strip(), out["latency"], out["tokens"]


def _run_skeptic(question, draft_answer, context, client, BASE_MODEL, gatekeeper_signals=None):
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
- Pay special attention to any concerns flagged above.
- For numeric answers: only challenge if you find a DIFFERENT specific value in the evidence.
- Be precise: quote or closely paraphrase the claim.
- Ignore claims that ARE clearly supported.
- If all claims are supported or correct, write: "No unsupported claims found."
- Output ONLY a bullet list of challenged claims. No explanation.

Challenged claims:
""".strip()

    out = ask_llm(prompt, client=client, model=BASE_MODEL, temperature=0.0, max_tokens=300)
    return out["text"].strip(), out["latency"], out["tokens"]


def _verify_grounder_citations(grounder_output, context):
    lines = grounder_output.splitlines()
    verified = []
    for line in lines:
        if line.strip().upper().startswith("SUPPORTED:") and "— Evidence:" in line:
            evidence_part = line.split("— Evidence:", 1)[1].strip().strip('"').strip("'")
            ctx_lower = context.lower()
            words = [w for w in evidence_part.lower().split() if len(w) > 1]
            if words and not all(w in ctx_lower for w in words):
                claim_part = line.split("— Evidence:", 1)[0].replace("SUPPORTED:", "", 1).strip()
                line = f"CONCEDED: {claim_part}"
        verified.append(line)
    return "\n".join(verified)


def _run_grounder(question, draft_answer, skeptic_output, context, client, BASE_MODEL):
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


def _run_adjudicator(question, draft_answer, skeptic_output, grounder_output,
                     context, client, JUDGE_MODEL):
    is_arith = is_arithmetic_question(question)
    is_multi = question_is_multispan(question)

    if is_arith:
        answer_instructions = """- This question requires calculation. Recompute the answer independently from the evidence — do NOT just copy the draft.
- Show your reasoning briefly, then state ONLY the final numeric result on the last line prefixed with 'Final answer:'
- If your computed result matches the draft, use [APPROVE]. If different, use [REVISE] with your computed result."""
    elif is_multi:
        answer_instructions = """- This question asks for values across multiple years or entities. Return ALL requested values in the same order asked.
- Format: just the values separated by commas or newlines (e.g. "X, Y" or "X\nY"). Do NOT add year labels or headers.
- Do not omit any of the requested values. No explanations."""
    else:
        answer_instructions = """- Return ONLY the direct factual answer. No explanations, no reasoning, no phrases like 'Based on the evidence'. Just the answer value(s).
- If the question asks for multiple values, include all of them."""

    prompt = f"""You are an expert Adjudicator. Your job is to produce the best possible answer to the question using the evidence and the debate context below.

Question:
{question}

Evidence:
{context}

Draft answer (from Tier 2 Refinement):
{draft_answer}

Skeptic's challenged claims:
{skeptic_output}

Grounder's verdicts:
{grounder_output}

Instructions:
- Read the evidence carefully and independently determine the correct answer.
- Use the debate to understand what was challenged and what was defended.
- Your response MUST begin with exactly one of: [APPROVE], [REVISE], or [ABSTAIN] — no other text before it.
- [APPROVE]: the draft answer is correct. Return it unchanged after the tag.
- [REVISE]: the draft has errors. Return your corrected answer after the tag.
- [ABSTAIN]: evidence is insufficient. Return exactly "Insufficient information." after the tag.
{answer_instructions}
- Use ONLY facts from the evidence. Do NOT introduce new claims.

Response:
""".strip()

    out = ask_llm(prompt, client=client, model=JUDGE_MODEL, temperature=0.0, max_tokens=350)
    raw = strip_refinement_prefix(out["text"])

    verdict = "REVISE"
    final_answer = raw
    for label in ("APPROVE", "REVISE", "ABSTAIN"):
        if raw.upper().startswith(f"[{label}]"):
            verdict = label
            final_answer = raw[len(f"[{label}]"):].strip()
            break

    if is_arith and verdict != "ABSTAIN":
        matches = list(re.finditer(r"Final answer\s*:\s*(.+?)(?:\n|$)", final_answer, re.IGNORECASE))
        if matches:
            final_answer = matches[-1].group(1).strip()

    if "insufficient information" in final_answer.lower() and verdict == "REVISE":
        verdict = "ABSTAIN"

    return final_answer, verdict, out["latency"], out["tokens"]


def guardrag_debate(question, retrieved, client, BASE_MODEL, JUDGE_MODEL,
                    baseline_result=None, nli_model=None, threshold=0.35, disabled_signals=None):
    if baseline_result is None:
        baseline_result = baseline_rag(question, retrieved, client, BASE_MODEL)

    draft_answer = baseline_result["answer"]
    gate = gatekeeper_v4(question, retrieved, draft_answer,
                         nli_model=nli_model, threshold=threshold, disabled_signals=disabled_signals)

    if not gate["trigger_debate"]:
        return {
            "answer":             draft_answer,
            "latency":            baseline_result["latency"],
            "tokens":             baseline_result["tokens"],
            "debate_triggered":   False,
            "gatekeeper_signals": gate["signals"],
            "gatekeeper_score":   gate["gatekeeper_score"],
            "threshold":          gate["threshold"],
            "debate_transcript":  None,
        }

    context = format_retrieved_context(retrieved, max_chars=4500)

    skeptic_out, s_lat, s_tok = _run_skeptic(
        question, draft_answer, context, client, BASE_MODEL,
        gatekeeper_signals=gate["signals"]
    )

    if "no unsupported claims" in skeptic_out.lower():
        return {
            "answer":              draft_answer,
            "latency":             round(baseline_result["latency"] + s_lat, 3),
            "tokens":              baseline_result["tokens"] + s_tok,
            "debate_triggered":    True,
            "adjudicator_verdict": "APPROVE",
            "gatekeeper_signals":  gate["signals"],
            "gatekeeper_score":    gate["gatekeeper_score"],
            "threshold":           gate["threshold"],
            "debate_transcript":   {"skeptic": skeptic_out, "grounder": None},
        }

    grounder_out, g_lat, g_tok = _run_grounder(
        question, draft_answer, skeptic_out, context, client, BASE_MODEL
    )

    # Run adjudicator if grounder conceded at least 1 claim
    conceded = [l for l in grounder_out.splitlines() if l.strip().upper().startswith("CONCEDED")]
    if len(conceded) < 1:
        total_latency = round(baseline_result["latency"] + s_lat + g_lat, 3)
        total_tokens  = baseline_result["tokens"] + s_tok + g_tok
        return {
            "answer":              draft_answer,
            "latency":             total_latency,
            "tokens":              total_tokens,
            "debate_triggered":    True,
            "adjudicator_verdict": "APPROVE",
            "gatekeeper_signals":  gate["signals"],
            "gatekeeper_score":    gate["gatekeeper_score"],
            "threshold":           gate["threshold"],
            "debate_transcript":   {"skeptic": skeptic_out, "grounder": grounder_out},
        }

    final_answer, verdict, a_lat, a_tok = _run_adjudicator(
        question, draft_answer, skeptic_out, grounder_out, context, client, JUDGE_MODEL
    )

    # If adjudicator approves, always return the original draft (refinement's answer)
    # to avoid any reformatting or corruption introduced by the model output
    if verdict == "APPROVE":
        final_answer = draft_answer

    total_latency = round(baseline_result["latency"] + s_lat + g_lat + a_lat, 3)
    total_tokens  = baseline_result["tokens"] + s_tok + g_tok + a_tok

    return {
        "answer":              final_answer,
        "latency":             total_latency,
        "tokens":              total_tokens,
        "debate_triggered":    True,
        "adjudicator_verdict": verdict,
        "regenerated":         verdict == "REVISE",
        "gatekeeper_signals":  gate["signals"],
        "gatekeeper_score":    gate["gatekeeper_score"],
        "threshold":           gate["threshold"],
        "debate_transcript":   {
            "skeptic":  skeptic_out,
            "grounder": grounder_out,
        },
    }

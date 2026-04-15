import re
import os

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
    "heuristic_arithmetic":        0.6,   # always triggers debate → PoT verification
    "heuristic_multispan":         0.5,
}

_NLI_ENTAILMENT_IDX = 1


def _clean_answer(text):
    """Strip formatting artifacts that hurt F1 scoring."""
    text = str(text).strip()
    # Remove markdown headers (###, ##, #)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    # Remove markdown bold/italic
    text = re.sub(r'\*+', '', text)
    # Remove "Answer:" or "Final answer:" prefixes
    text = re.sub(r'^(final\s+)?answer\s*:\s*', '', text, flags=re.IGNORECASE)
    # Remove bracket tags like [REVISE], [APPROVE]
    text = re.sub(r'^\[(APPROVE|REVISE|ABSTAIN)\]\s*', '', text, flags=re.IGNORECASE)
    # Strip trailing punctuation artifacts
    text = text.strip().strip('.')
    return text.strip()


def _is_valid_numeric_answer(text):
    """Return True if text looks like a plausible numeric financial answer."""
    text = str(text).strip()
    # Reject empty, pure markdown, or garbage
    if not text or re.match(r'^[#\}\{\\\|]+$', text):
        return False
    # Must contain at least one digit
    if not re.search(r'\d', text):
        return False
    return True


def _quick_normalize(s):
    """Quick numeric normalization for comparison — strips currency/percent, parses floats."""
    s = str(s).lower().strip()
    s = re.sub(r'[$%,\s]', '', s)
    s = re.sub(r'\((\d+\.?\d*)\)', r'-\1', s)   # accounting negatives: (123) → -123
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else f"{round(v, 4)}"
    except (ValueError, TypeError):
        return s


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
    """Math-specialized skeptic — re-derives calculation from scratch to verify."""
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


def _run_grounder(question, draft_answer, skeptic_output, context, client, JUDGE_MODEL):
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

    out = ask_llm(prompt, client=client, model=JUDGE_MODEL, temperature=0.0, max_tokens=400)
    raw = out["text"].strip()
    verified = _verify_grounder_citations(raw, context)
    return verified, out["latency"], out["tokens"]


def _generate_guardrag_answer(question, retrieved, client, JUDGE_MODEL):
    """
    GPT-4o generates a fresh answer for GUARD-RAG tier.
    - Arithmetic: uses structured key-value facts (cleaner than pipe-delimited chunks)
    - Multispan text: explicit multi-value extraction with structured facts
    - Text/span: concise direct answer with evidence-grounding rules
    """
    from tiers.pot import extract_structured_facts

    is_arith = is_arithmetic_question(question)
    is_multi = question_is_multispan(question)

    if is_arith:
        structured = extract_structured_facts(retrieved)
        context_for_prompt = structured if structured.strip() else format_retrieved_context(retrieved, max_chars=4500)
        prompt = f"""You are a financial analyst. Solve this step by step.

Structured evidence (Row [Section] (Column): Value):
{context_for_prompt}

Question:
{question}

Step 1: Extract the exact values needed — state the row and column for each.
Step 2: Apply the correct formula (e.g., % change = (new - old) / old × 100).
Step 3: Compute the result. Round to 2 decimal places if not a whole number.
Final answer:""".strip()
        max_tokens = 400

    elif is_multi:
        structured = extract_structured_facts(retrieved)
        context_for_prompt = structured if structured.strip() else format_retrieved_context(retrieved, max_chars=4500)
        prompt = f"""You are a financial analyst. Extract the specific values requested.

Evidence:
{context_for_prompt}

Question:
{question}

Rules:
- Return ONLY the values, in the same order as asked, separated by commas.
- No labels, no units, no explanations — just the numbers or names.
- Example: if asked for 2017 and 2018 revenues: "1234, 5678"

Answer:""".strip()
        max_tokens = 120

    else:
        context = format_retrieved_context(retrieved, max_chars=4500)
        prompt = f"""You are a financial analyst answering using ONLY the evidence below.

Evidence:
{context}

Question:
{question}

Rules:
- Return ONLY the direct answer. No explanations, no reasoning, no "Based on evidence" phrases.
- Be as concise as possible — just the value(s) asked for.
- If asking for a name/entity, return only the name. If asking for a number, return only the number.
- Keep answer under 15 words unless a list is specifically requested.

Answer:""".strip()
        max_tokens = 100

    out = ask_llm(prompt, client=client, model=JUDGE_MODEL, temperature=0.0, max_tokens=max_tokens)
    answer = out["text"].strip()
    if is_arith:
        from tiers.baseline import _extract_final_answer
        answer = _extract_final_answer(answer)
    return _clean_answer(answer), out["latency"], out["tokens"]


def _run_span_extractor(question, draft_answer, context, client, BASE_MODEL):
    """For text questions: extract the precise span from evidence, trim verbosity."""
    prompt = f"""Extract the precise answer to the question from the evidence.

Evidence:
{context}

Question:
{question}

Current answer: {draft_answer}

Rules:
- If the current answer is already a precise value or short phrase: return it unchanged.
- If the current answer is too verbose or contains reasoning: extract ONLY the key value(s).
- Return ONLY the answer. No explanations, no "Based on evidence", no full sentences.
- If the question asks for multiple values, list them concisely separated by commas.

Answer:""".strip()

    out = ask_llm(prompt, client=client, model=BASE_MODEL, temperature=0.0, max_tokens=80)
    return out["text"].strip(), out["latency"], out["tokens"]


def _run_adjudicator(question, draft_answer, skeptic_output, grounder_output,
                     context, client, JUDGE_MODEL, pot_answer=None):
    is_arith = is_arithmetic_question(question)
    is_multi = question_is_multispan(question)

    pot_section = ""
    if pot_answer is not None:
        pot_section = f"\nPython-verified answer (from code execution — DEFINITIVE for arithmetic): {pot_answer}\n"

    if is_arith:
        answer_instructions = """- This question requires arithmetic. If a Python-verified answer is provided above, it IS the final answer — use it unconditionally.
- [APPROVE] if the Python answer matches the draft exactly. [REVISE] with the Python answer if it differs.
- If no Python answer: independently recompute using the evidence and provide your result after [REVISE]."""
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
{pot_section}
Draft answer (from generation):
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
    is_arith = is_arithmetic_question(question)
    is_multi = question_is_multispan(question)

    gate = gatekeeper_v4(question, retrieved, draft_answer,
                         nli_model=nli_model, threshold=threshold, disabled_signals=disabled_signals)

    context = format_retrieved_context(retrieved, max_chars=4500)

    # ── Path 1: Text/span questions (not arithmetic, not multispan) ───────────
    # Only use GPT-4o when gatekeeper fires (baseline is probably wrong/uncertain).
    # When gatekeeper doesn't fire, baseline mini answer is likely correct — don't break it.
    if not is_arith and not is_multi:
        if gate["trigger_debate"]:
            gpt4o_ans, g4o_lat, g4o_tok = _generate_guardrag_answer(
                question, retrieved, client, JUDGE_MODEL
            )
            final = _clean_answer(gpt4o_ans) if gpt4o_ans.strip() else draft_answer
        else:
            final = draft_answer
            g4o_lat = g4o_tok = 0

        return {
            "answer":             final,
            "latency":            round(baseline_result["latency"] + g4o_lat, 3),
            "tokens":             baseline_result["tokens"] + g4o_tok,
            "debate_triggered":   gate["trigger_debate"],
            "gatekeeper_signals": gate["signals"],
            "gatekeeper_score":   gate["gatekeeper_score"],
            "threshold":          gate["threshold"],
            "debate_transcript":  None,
        }

    # ── Path 2: Multispan text questions (not arithmetic) ─────────────────────
    # Only use GPT-4o when gatekeeper fires — don't break already-correct answers.
    # Use full context (not structured facts) so text answers aren't stripped to numbers.
    if is_multi and not is_arith:
        if gate["trigger_debate"]:
            # Use full-context GPT-4o for text multispan (structured facts strips text answers)
            ctx_full = format_retrieved_context(retrieved, max_chars=4500)
            prompt = f"""You are a financial analyst. Answer using ONLY the evidence below.

Evidence:
{ctx_full}

Question:
{question}

Rules:
- Return ONLY the direct answer. No explanations, no "Based on evidence" phrases.
- If multiple values are requested, provide them in the same order asked, separated by semicolons.
- Preserve original text/descriptions exactly — do not paraphrase or abbreviate.

Answer:""".strip()
            out = ask_llm(prompt, client=client, model=JUDGE_MODEL, temperature=0.0, max_tokens=150)
            final = _clean_answer(out["text"]) if out["text"].strip() else draft_answer
            g4o_lat, g4o_tok = out["latency"], out["tokens"]
        else:
            final = draft_answer
            g4o_lat = g4o_tok = 0

        return {
            "answer":             final,
            "latency":            round(baseline_result["latency"] + g4o_lat, 3),
            "tokens":             baseline_result["tokens"] + g4o_tok,
            "debate_triggered":   gate["trigger_debate"],
            "gatekeeper_signals": gate["signals"],
            "gatekeeper_score":   gate["gatekeeper_score"],
            "threshold":          gate["threshold"],
            "debate_transcript":  None,
        }

    # ── Path 3: Arithmetic questions ──────────────────────────────────────────
    # Three-signal verification: GPT-4o (structured facts) + PoT (Python) + Math Skeptic
    # Each signal independently derives the answer; agreement → high confidence shortcut.

    # Step A: GPT-4o with structured key-value facts (cleaner extraction than raw chunks)
    gpt4o_answer, g4o_lat, g4o_tok = _generate_guardrag_answer(
        question, retrieved, client, JUDGE_MODEL
    )
    # Only use GPT-4o answer if it's a valid numeric result (not markdown garbage)
    if gpt4o_answer.strip() and _is_valid_numeric_answer(gpt4o_answer):
        draft_answer = gpt4o_answer
    # else keep the T2 draft

    # Step B: PoT — Python executes the calculation deterministically
    from tiers.pot import pot_rag
    _pot_answer, _pot_lat, _pot_tok = pot_rag(question, retrieved, client, JUDGE_MODEL)
    # Validate PoT result too
    if _pot_answer and not _is_valid_numeric_answer(_pot_answer):
        _pot_answer = None

    # Step C: Agreement shortcut — if PoT and GPT-4o independently produce the same answer,
    # return immediately with high confidence (two independent methods rarely agree on a wrong answer)
    if _pot_answer is not None and _quick_normalize(_pot_answer) == _quick_normalize(draft_answer):
        return {
            "answer":              _clean_answer(_pot_answer),
            "latency":             round(baseline_result["latency"] + g4o_lat + _pot_lat, 3),
            "tokens":              baseline_result["tokens"] + g4o_tok + _pot_tok,
            "debate_triggered":    True,
            "adjudicator_verdict": "APPROVE",
            "gatekeeper_signals":  gate["signals"],
            "gatekeeper_score":    gate["gatekeeper_score"],
            "threshold":           gate["threshold"],
            "debate_transcript":   {"skeptic": "PoT+GPT-4o agreement shortcut", "grounder": None},
        }

    # Step D: Math Skeptic — independently re-derives the calculation step by step
    # (was defined in the original code but never invoked in the debate path)
    math_out, ms_lat, ms_tok = _run_math_skeptic(
        question, draft_answer, context, client, BASE_MODEL
    )

    base_lat = baseline_result["latency"] + g4o_lat + _pot_lat + ms_lat
    base_tok = baseline_result["tokens"] + g4o_tok + _pot_tok + ms_tok

    # Math skeptic confirms GPT-4o answer → APPROVE (two independent derivations agree)
    if "CORRECT:" in math_out.upper():
        return {
            "answer":              _clean_answer(draft_answer),
            "latency":             round(base_lat, 3),
            "tokens":              base_tok,
            "debate_triggered":    True,
            "adjudicator_verdict": "APPROVE",
            "gatekeeper_signals":  gate["signals"],
            "gatekeeper_score":    gate["gatekeeper_score"],
            "threshold":           gate["threshold"],
            "debate_transcript":   {"skeptic": math_out, "grounder": None},
        }

    # Math skeptic found an error — extract corrected answer
    if "INCORRECT:" in math_out.upper():
        m = re.search(r"correct answer is\s+(.+?)(?:\.|$)", math_out, re.IGNORECASE)
        if m:
            corrected = _clean_answer(m.group(1).strip())
            if re.search(r'-?\d+\.?\d*', corrected):   # must be a valid numeric answer
                if _pot_answer and _quick_normalize(_pot_answer) == _quick_normalize(corrected):
                    # PoT + Math Skeptic both agree on the correction → very high confidence
                    return {
                        "answer":              corrected,
                        "latency":             round(base_lat, 3),
                        "tokens":              base_tok,
                        "debate_triggered":    True,
                        "adjudicator_verdict": "REVISE",
                        "gatekeeper_signals":  gate["signals"],
                        "gatekeeper_score":    gate["gatekeeper_score"],
                        "threshold":           gate["threshold"],
                        "debate_transcript":   {"skeptic": math_out, "grounder": "PoT corroborated"},
                    }
                # Math skeptic alone found error → trust its step-by-step re-derivation
                return {
                    "answer":              corrected,
                    "latency":             round(base_lat, 3),
                    "tokens":              base_tok,
                    "debate_triggered":    True,
                    "adjudicator_verdict": "REVISE",
                    "gatekeeper_signals":  gate["signals"],
                    "gatekeeper_score":    gate["gatekeeper_score"],
                    "threshold":           gate["threshold"],
                    "debate_transcript":   {"skeptic": math_out, "grounder": None},
                }

    # Step E: Math skeptic output is ambiguous — fall back to PoT as primary if available
    if _pot_answer is not None:
        return {
            "answer":              _clean_answer(_pot_answer),
            "latency":             round(base_lat, 3),
            "tokens":              base_tok,
            "debate_triggered":    True,
            "adjudicator_verdict": "REVISE",
            "gatekeeper_signals":  gate["signals"],
            "gatekeeper_score":    gate["gatekeeper_score"],
            "threshold":           gate["threshold"],
            "debate_transcript":   {"skeptic": math_out, "grounder": "PoT primary (skeptic ambiguous)"},
        }

    # Step F: Full debate fallback — PoT failed AND math skeptic ambiguous
    # Use standard Skeptic → Grounder → Adjudicator chain
    skeptic_out, s_lat, s_tok = _run_skeptic(
        question, draft_answer, context, client, BASE_MODEL,
        gatekeeper_signals=gate["signals"]
    )

    if "no unsupported claims" in skeptic_out.lower():
        return {
            "answer":              draft_answer,
            "latency":             round(base_lat + s_lat, 3),
            "tokens":              base_tok + s_tok,
            "debate_triggered":    True,
            "adjudicator_verdict": "APPROVE",
            "gatekeeper_signals":  gate["signals"],
            "gatekeeper_score":    gate["gatekeeper_score"],
            "threshold":           gate["threshold"],
            "debate_transcript":   {"skeptic": skeptic_out, "grounder": None},
        }

    grounder_out, g_lat, g_tok = _run_grounder(
        question, draft_answer, skeptic_out, context, client, JUDGE_MODEL
    )

    conceded = [l for l in grounder_out.splitlines() if l.strip().upper().startswith("CONCEDED")]
    if len(conceded) < 1:
        return {
            "answer":              draft_answer,
            "latency":             round(base_lat + s_lat + g_lat, 3),
            "tokens":              base_tok + s_tok + g_tok,
            "debate_triggered":    True,
            "adjudicator_verdict": "APPROVE",
            "gatekeeper_signals":  gate["signals"],
            "gatekeeper_score":    gate["gatekeeper_score"],
            "threshold":           gate["threshold"],
            "debate_transcript":   {"skeptic": skeptic_out, "grounder": grounder_out},
        }

    final_answer, verdict, a_lat, a_tok = _run_adjudicator(
        question, draft_answer, skeptic_out, grounder_out, context, client, JUDGE_MODEL,
        pot_answer=_pot_answer
    )

    if verdict == "APPROVE":
        final_answer = draft_answer

    final_answer = _clean_answer(final_answer)
    total_lat = round(base_lat + s_lat + g_lat + a_lat, 3)
    total_tok = base_tok + s_tok + g_tok + a_tok

    return {
        "answer":              final_answer,
        "latency":             total_lat,
        "tokens":              total_tok,
        "debate_triggered":    True,
        "adjudicator_verdict": verdict,
        "gatekeeper_signals":  gate["signals"],
        "gatekeeper_score":    gate["gatekeeper_score"],
        "threshold":           gate["threshold"],
        "debate_transcript":   {"skeptic": skeptic_out, "grounder": grounder_out},
    }

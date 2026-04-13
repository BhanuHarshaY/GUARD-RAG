import re
import os

from tiers.llm_utils import ask_llm, format_retrieved_context, strip_refinement_prefix
from tiers.baseline import baseline_rag, is_arithmetic_question
from indexing.table_parser import normalize_text_for_match
from retrieval.retriever import question_is_list_like, question_is_multispan

# ---------------------------------------------------------------------------
# Optional: load DSPy-compiled adjudicator if available
# ---------------------------------------------------------------------------
_COMPILED_ADJUDICATOR = None
_COMPILED_ADJ_PATH = "dspy_compiled_adjudicator.json"

def _try_load_compiled_adjudicator():
    global _COMPILED_ADJUDICATOR
    if not os.path.exists(_COMPILED_ADJ_PATH):
        return
    try:
        import dspy
        from dspy_adjudicator import DSPyAdjudicator
        adj = DSPyAdjudicator()
        adj.load(_COMPILED_ADJ_PATH)
        _COMPILED_ADJUDICATOR = adj
        print(f"Loaded compiled DSPy adjudicator from {_COMPILED_ADJ_PATH}")
    except Exception as e:
        print(f"Warning: could not load compiled adjudicator ({e}), using default.")

_try_load_compiled_adjudicator()

# ---------------------------------------------------------------------------
# DSPy ChainOfThought adjudicator (lazy init — no compilation cost)
# ---------------------------------------------------------------------------
_DSPY_COT = None
_DSPY_COT_INIT_DONE = False

def _get_dspy_cot():
    """Lazy-initialize DSPy ChainOfThought adjudicator using OpenRouter."""
    global _DSPY_COT, _DSPY_COT_INIT_DONE
    if _DSPY_COT_INIT_DONE:
        return _DSPY_COT
    _DSPY_COT_INIT_DONE = True
    try:
        import dspy
        from dspy_adjudicator import AdjudicatorSignature
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            return None
        lm = dspy.LM(
            model="openai/gpt-4o",
            api_key=api_key,
            api_base="https://openrouter.ai/api/v1",
            temperature=0.0,
            max_tokens=450,
        )
        dspy.configure(lm=lm)
        _DSPY_COT = dspy.ChainOfThought(AdjudicatorSignature)
        print("DSPy ChainOfThought adjudicator ready")
    except Exception as e:
        print(f"DSPy CoT init skipped: {e}")
        _DSPY_COT = None
    return _DSPY_COT

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
    # Remove markdown bold/italic
    text = re.sub(r'\*+', '', text)
    # Remove "Answer:" or "Final answer:" prefixes
    text = re.sub(r'^(final\s+)?answer\s*:\s*', '', text, flags=re.IGNORECASE)
    # Remove bracket tags like [REVISE], [APPROVE]
    text = re.sub(r'^\[(APPROVE|REVISE|ABSTAIN)\]\s*', '', text, flags=re.IGNORECASE)
    # Strip trailing punctuation artifacts
    text = text.strip().strip('.')
    return text.strip()


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
    GPT-4o generates a fresh, independent answer for GUARD-RAG tier.
    Stronger than mini — used as the debate draft when gatekeeper fires.
    """
    from tiers.llm_utils import format_retrieved_context
    context = format_retrieved_context(retrieved, max_chars=4500)
    is_arith = is_arithmetic_question(question)

    if is_arith:
        prompt = f"""You are a financial analyst. Answer the question using ONLY the evidence.

Evidence:
{context}

Question:
{question}

Work through the calculation step by step, then state only the final number.
Final answer:""".strip()
        max_tokens = 300
    else:
        prompt = f"""You are a financial analyst. Answer the question using ONLY the evidence.

Rules:
- Return ONLY the direct answer value(s). No explanation, no reasoning sentences.
- If multiple values are asked for, list them concisely.
- Do not say "Based on the evidence" or similar phrases.

Evidence:
{context}

Question:
{question}

Answer:""".strip()
        max_tokens = 150

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
    # Use DSPy-compiled adjudicator if available (better prompts via MIPROv2)
    if _COMPILED_ADJUDICATOR is not None:
        import time
        t0 = time.time()
        try:
            pred = _COMPILED_ADJUDICATOR.forward(
                question=question,
                evidence=context,
                draft_answer=draft_answer,
                skeptic_output=skeptic_output,
                grounder_output=grounder_output,
            )
            latency = round(time.time() - t0, 3)
            raw = strip_refinement_prefix(pred.final_answer or "")
            verdict, final_answer = "REVISE", raw
            for label in ("APPROVE", "REVISE", "ABSTAIN"):
                if raw.upper().startswith(f"[{label}]"):
                    verdict = label
                    final_answer = raw[len(f"[{label}]"):].strip()
                    break
            if "insufficient information" in final_answer.lower() and verdict == "REVISE":
                verdict = "ABSTAIN"
            return final_answer, verdict, latency, 0  # token count not available from DSPy
        except Exception:
            pass  # fall through to standard adjudicator

    is_arith = is_arithmetic_question(question)
    is_multi = question_is_multispan(question)

    pot_section = ""
    if pot_answer is not None:
        pot_section = f"\nPython-verified answer (from code execution — treat as highly trusted): {pot_answer}\n"

    # Try DSPy ChainOfThought adjudicator (adds explicit reasoning step before verdict)
    # Only for non-arithmetic where PoT isn't available — CoT helps text/span decisions
    if not is_arith and _COMPILED_ADJUDICATOR is None:
        import time as _time
        cot = _get_dspy_cot()
        if cot is not None:
            try:
                t0 = _time.time()
                evidence_for_dspy = context[:3000]
                pred = cot(
                    question=question,
                    evidence=evidence_for_dspy,
                    draft_answer=draft_answer,
                    skeptic_output=skeptic_output,
                    grounder_output=grounder_output,
                )
                latency = round(_time.time() - t0, 3)
                raw = (pred.final_answer or "").strip()
                raw = strip_refinement_prefix(raw)
                verdict, final_answer = "REVISE", raw
                for label in ("APPROVE", "REVISE", "ABSTAIN"):
                    if raw.upper().startswith(f"[{label}]"):
                        verdict = label
                        final_answer = raw[len(f"[{label}]"):].strip()
                        break
                if "insufficient information" in final_answer.lower() and verdict == "REVISE":
                    verdict = "ABSTAIN"
                return final_answer, verdict, latency, 0
            except Exception:
                pass  # fall through to standard prompt

    if is_arith:
        answer_instructions = """- This question requires calculation. If a Python-verified answer is provided above, use it as your FINAL answer — Python does not make arithmetic errors.
- Do NOT recompute. Trust the Python answer.
- If the Python answer matches the draft: use [APPROVE].
- If the Python answer differs from the draft: use [REVISE] with ONLY the Python answer.
- If no Python answer is provided: recompute independently and prefix result with 'Final answer:'"""
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

    # Text/span questions: debate hurts — return refinement answer directly
    if not is_arithmetic_question(question) and not question_is_multispan(question):
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

    # GPT-4o generates a fresh independent answer — this is the debate draft
    # Stronger than mini; debate verifies it's grounded in the evidence
    gpt4o_answer, g4o_lat, g4o_tok = _generate_guardrag_answer(
        question, retrieved, client, JUDGE_MODEL
    )
    # Use GPT-4o answer as the draft for debate (replaces T2 mini answer)
    draft_answer = gpt4o_answer if gpt4o_answer.strip() else draft_answer

    # For arithmetic: also run PoT as adjudicator signal
    _pot_answer = None
    _pot_lat = 0
    _pot_tok = 0
    if is_arithmetic_question(question):
        from tiers.pot import pot_rag
        _pot_answer, _pot_lat, _pot_tok = pot_rag(question, retrieved, client, JUDGE_MODEL)

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
        question, draft_answer, skeptic_out, context, client, JUDGE_MODEL
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
        question, draft_answer, skeptic_out, grounder_out, context, client, JUDGE_MODEL,
        pot_answer=_pot_answer
    )

    # If adjudicator approves, return draft unchanged
    if verdict == "APPROVE":
        final_answer = draft_answer

    # ── Iterative debate: second round on REVISE cases only ───────────────────
    # Only fires on ~12% of samples. Catches cases where the revised answer
    # still has unsupported claims. APPROVE and non-debated paths untouched.
    iter_lat, iter_tok = 0, 0
    if verdict == "REVISE" and final_answer.strip():
        final_answer = _clean_answer(final_answer)
        s2_out, s2_lat, s2_tok = _run_skeptic(
            question, final_answer, context, client, BASE_MODEL,
            gatekeeper_signals=gate["signals"]
        )
        iter_lat += s2_lat
        iter_tok += s2_tok
        if "no unsupported claims" not in s2_out.lower():
            g2_out, g2_lat, g2_tok = _run_grounder(
                question, final_answer, s2_out, context, client, JUDGE_MODEL
            )
            iter_lat += g2_lat
            iter_tok += g2_tok
            conceded2 = [l for l in g2_out.splitlines() if l.strip().upper().startswith("CONCEDED")]
            if conceded2:
                fa2, v2, a2_lat, a2_tok = _run_adjudicator(
                    question, final_answer, s2_out, g2_out, context, client, JUDGE_MODEL,
                    pot_answer=_pot_answer
                )
                iter_lat += a2_lat
                iter_tok += a2_tok
                if v2 != "ABSTAIN" and fa2.strip():
                    final_answer = _clean_answer(fa2) if v2 == "REVISE" else final_answer
                    verdict = v2

    final_answer = _clean_answer(final_answer)

    total_latency = round(baseline_result["latency"] + g4o_lat + _pot_lat + s_lat + g_lat + a_lat + iter_lat, 3)
    total_tokens  = baseline_result["tokens"] + g4o_tok + _pot_tok + s_tok + g_tok + a_tok + iter_tok

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

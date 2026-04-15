# GUARD-RAG
### Gatekeeper-Guided Debate for Hallucination Reduction in Financial QA

**Team:** Shrivarshini Narayanan, Shivam Singh, Bhanu Harsha Yanamadala
**Course:** CS6180, Northeastern University

---

## Overview

Financial question answering over hybrid tabular-textual documents demands both precise arithmetic reasoning and strict factual grounding. Standard RAG systems fail on these tasks because they apply a single LLM call regardless of how difficult or risky the question is.

**GUARD-RAG** is a three-tier escalation framework that reserves expensive verification only for high-risk predictions. It embeds agents directly inside the RAG pipeline and allocates compute adaptively based on answer confidence:

- Easy questions (47.9%) → answered at minimal cost, no debate
- Hard questions (52.1%) → escalated through a full Skeptic–Grounder–Adjudicator debate with GPT-4o

On TAT-QA (n=300), GUARD-RAG achieves **F1 = 0.629**, a **+12.1% relative gain** over zero-shot RAG, with the **lowest hallucination rate** of all three tiers — despite generating 5.5× more tokens on average.

---

## Key Findings

1. **Selective debate beats uniform debate.** Applying debate to all questions *hurts* F1 by −2.4%. The gatekeeper correctly identifies which questions benefit from debate (arithmetic, multispan) and which do not (text/span).

2. **Self-refinement increases hallucination; debate decreases it.** Refinement (T2) hallucination rate = 0.733, *worse* than Baseline (0.728). GUARD-RAG achieves 0.710 — the lowest — by requiring the Grounder to cite evidence for every claim.

3. **PoT as advisory signal, not override.** Using Program-of-Thought as a direct override degrades F1 by −4.5%. Passing it as an advisory signal to the Adjudicator allows it to be weighed against debate evidence.

4. **The GUARD-RAG advantage grows with scale.** At n=150, the gap over Baseline is +2.7 F1 points. At n=300 (harder questions added), it grows to +6.8 points — confirming genuine architectural advantage, not luck.

5. **Arithmetic gains are largest.** GUARD-RAG improves arithmetic F1 by +13.2% over Baseline (0.609 vs 0.538). Text/span questions gain only +0.8%, intentionally — debate is bypassed for text.

---

## Results (n=300, seed=42, TAT-QA train split)

| System | F1 | EM | Hallucination Rate ↓ | Avg Tokens | Latency (s) | Debate Rate |
|--------|----|----|----------------------|------------|-------------|-------------|
| Baseline RAG (T1) | 0.561 | 0.507 | 0.728 | 1,102 | 0.877 | — |
| Refinement RAG (T2) | 0.593 | 0.539 | 0.733 | 2,395 | 2.339 | — |
| **GUARD-RAG (T3)** | **0.629** | **0.576** | **0.710** | 6,101 | 5.630 | 52.1% |

### By Question Type (n=150 annotated)

| Type | n | Baseline F1 | Refinement F1 | GUARD-RAG F1 | Gain vs Baseline |
|------|---|-------------|---------------|--------------|-----------------|
| Arithmetic | 52 | 0.538 | 0.558 | **0.609** | +13.2% |
| Text / Span | 98 | 0.628 | 0.628 | **0.633** | +0.8% |

---

## Three-Tier Architecture

```
Question
   │
   ▼
┌─────────────────────────────────────────────────┐
│  Tier 1 — Baseline RAG                          │
│  Single GPT-4o-mini call, k=6 chunks            │
└──────────────────────┬──────────────────────────┘
                       │ T1 answer
                       ▼
┌─────────────────────────────────────────────────┐
│  Tier 2 — Refinement RAG                        │
│  Formula-error verification (few-shot examples) │
└──────────────────────┬──────────────────────────┘
                       │ T2 answer
                       ▼
┌─────────────────────────────────────────────────┐
│  Tier 3 — GUARD-RAG                             │
│                                                 │
│  Gatekeeper v4 (6 signals)                      │
│       │                                         │
│  score < 0.35 ──► return T2 answer (47.9%)      │
│       │                                         │
│  score ≥ 0.35                                   │
│       │                                         │
│       ▼                                         │
│  GPT-4o generates fresh answer (k=12 chunks)    │
│       │                                         │
│  PoT Python execution (arithmetic only)         │
│       │                                         │
│  Skeptic → Grounder → Adjudicator (DSPy CoT)    │
│       │                                         │
│  Final: [APPROVE] / [REVISE] / [ABSTAIN]        │
└─────────────────────────────────────────────────┘
```

---

## How Each Tier Works

### Tier 1 — Baseline RAG (`tiers/baseline.py`)

Issues a single GPT-4o-mini call with retrieved context.
- Arithmetic questions: chain-of-thought prompt with final-number extraction
- Text/span questions: five-rule grounding prompt (evidence-only, no filler phrases)
- Temperature: `0.0`, k=6 chunks

### Tier 2 — Refinement RAG (`tiers/refinement.py`)

Passes the T1 answer through a formula-error verification prompt with three labelled few-shot examples:
1. Sum vs. average confusion
2. Wrong percentage-change direction (old−new vs new−old)
3. Correct draft confirmation (no change needed)

A magnitude guard rejects revisions where `|draft / revised| > 100` to prevent runaway corrections.

### Tier 3 — GUARD-RAG (`tiers/guardrag.py`)

#### 3a. Gatekeeper v4

Computes a risk score `s = Σ wᵢ · 1[signalᵢ fires]` over 6 signals. Debate triggers when `s ≥ θ = 0.35`.

| Signal | Weight | Fires when |
|--------|--------|------------|
| `heuristic_arithmetic` | **0.60** | Arithmetic keyword detected (percentage change, difference, ratio, sum, grew by, fell by) |
| `heuristic_multispan` | **0.50** | Multiple entities/years requested (respectively, each of, both, each year) |
| `heuristic_insufficient_info` | **0.40** | Answer says "insufficient information" but structured table chunks exist |
| `nli_low_grounding` | **0.30** | DeBERTa-v3-small NLI entailment score < 0.35 |
| `heuristic_missing_rows` | **0.30** | List-like question with ≥2 row labels but fewer than 2 appear in answer |
| `heuristic_too_long` | **0.20** | Answer exceeds 80 words |

The threshold θ=0.35 ensures full coverage of arithmetic questions (score ≥ 0.6) and most multispan questions (score ≥ 0.5) while leaving text/span questions (score 0.0–0.3) below threshold.

#### 3b. Dual Retrieval

T1 and T2 receive k=6 chunks. GUARD-RAG retrieves k=12 from the same index. Financial tables span 10–20 rows; k=6 frequently misses both numerator and denominator rows for percentage-change questions.

#### 3c. GPT-4o Independent Answer Generation

For triggered questions, GUARD-RAG generates a fresh answer via GPT-4o over the 12-chunk context, decoupled from T2. Uses structured key-value facts (`Row [Section] (Column): Value`) for arithmetic questions to improve number extraction.

#### 3d. Program-of-Thought (PoT) Verification (`tiers/pot.py`)

For arithmetic questions, generates Python code that extracts values from structured facts and executes it in a restricted sandbox. The result is passed to the Adjudicator as an **advisory signal** — not an override. Direct PoT override degraded F1 by −4.5% in ablation; advisory integration allows the Adjudicator to weigh it against debate evidence.

#### 3e. Asymmetric Debate

Three agents run in sequence, each with a distinct role:

**Skeptic** (GPT-4o-mini) — Conservative challenger:
- Lists specific claims NOT directly supported by the evidence
- For numeric values: only challenges if it finds a *different* specific value in evidence
- If no unsupported claims found → APPROVE immediately (Grounder and Adjudicator skipped)

**Grounder** (GPT-4o) — Evidence defender:
- For each challenged claim: `SUPPORTED: <claim> — Evidence: <exact quote>` or `CONCEDED: <claim>`
- A citation verifier fuzzy-matches quoted evidence against the actual context; unverifiable citations are automatically flipped to CONCEDED
- If no concessions → APPROVE immediately (Adjudicator skipped)

**Adjudicator** (GPT-4o + DSPy `ChainOfThought`) — Final verdict:
- Generates an explicit reasoning trace before issuing a verdict
- `[APPROVE]` — draft is correct, return unchanged
- `[REVISE]` — draft has errors, return corrected answer
- `[ABSTAIN]` — evidence is insufficient, return "Insufficient information."
- On `[REVISE]`, a second debate round refines the revised answer (fires for ~2.3% of all samples)

Role asymmetry is intentional: the Skeptic's conservatism preserves high-quality GPT-4o answers; the Grounder's citation requirement grounds every claim in evidence.

---

## Retrieval Pipeline (`retrieval/retriever.py`)

**Step 1 — Dense retrieval:**
Encodes question with `all-MiniLM-L6-v2`, searches FAISS `IndexFlatIP` with `candidate_k=80`.

**Step 2 — Lexical reranking:**
| Boost | Condition |
|-------|-----------|
| +0.30 | Lexical overlap between question tokens and chunk text |
| +0.08 | Table chunk type (`table_cell`, `table_row`, `table_section`) |
| +0.10 | Table cell/row on list-like questions |
| +0.15 | Row label or section label matches question tokens |
| +0.10 | Year mention in question matches year in chunk |

**Step 3 — Section expansion:**
Top-k results seed a section set; up to 10 additional table rows/cells from the same section are appended to ensure full table context.

---

## Ablation Study

Pre-computed results are in `ablation_results/`. Key findings:

### Component Ablation (n=150)

| Configuration | F1 | Δ vs Full |
|--------------|-----|-----------|
| Full GUARD-RAG | 0.624 | — |
| w/ PoT override (not signal) | 0.596 | −4.5% |
| w/o GPT-4o generation | 0.603 | −3.4% |
| w/o dual retrieval (k=6 for T3) | 0.608 | −2.6% |
| w/ uniform debate (no gatekeeper) | 0.609 | −2.4% |
| w/ debate on text questions | 0.611 | −2.1% |
| w/o PoT advisory signal | 0.617 | −1.1% |
| w/o DSPy CoT adjudicator | 0.618 | −1.0% |

### Gatekeeper Signal Ablation (n=15)

| Signal removed | F1 drop |
|---------------|---------|
| `heuristic_arithmetic` (w=0.6) | −20.3% |
| `heuristic_insufficient_info` (w=0.4) | −20.3% |
| `heuristic_multispan` (w=0.5) | −10.2% |
| `nli_low_grounding` (w=0.3) | −10.2% |
| `heuristic_missing_rows` (w=0.3) | 0.0% |
| `heuristic_too_long` (w=0.2) | 0.0% |

---

## Project Structure

```
GUARD-RAG/
├── config.py                  # API client (OpenRouter), model names, embed/NLI singletons
├── run.py                     # Main runner: builds indexes, evaluates all tiers, saves CSVs
├── pipeline.py                # Single-question walkthrough across all tiers
├── ablation.py                # Signal ablation + threshold sweep (offline + online)
├── analysis.py                # Post-hoc analysis utilities
├── sweep_threshold.py         # Threshold sweep: collect (API) + sweep (offline)
├── data/
│   └── loader.py              # TAT-QA loading and preprocessing
├── indexing/
│   ├── table_parser.py        # Financial table → structured chunks (row/cell/section)
│   └── vector_store.py        # FAISS index construction (baseline + table-aware)
├── retrieval/
│   └── retriever.py           # Dense retrieval + lexical reranking + section expansion
├── tiers/
│   ├── llm_utils.py           # ask_llm (retry/backoff), format_context, sanitize helpers
│   ├── baseline.py            # Tier 1: single GPT-4o-mini call
│   ├── refinement.py          # Tier 2: formula-error verification
│   ├── guardrag.py            # Tier 3: gatekeeper + debate + adjudicator
│   └── pot.py                 # Program-of-Thought arithmetic verification
├── evaluation/
│   ├── metrics.py             # compute_f1, compute_em, compute_hallucination_rate
│   └── evaluator.py           # evaluate_all, summarize_results
├── ablation_results/          # Pre-computed ablation outputs
│   ├── ablation_comprehensive.csv   # 69-row full ablation (12 studies)
│   ├── answer_type_breakdown.csv    # Arithmetic vs span F1 breakdown
│   ├── component_ablation.csv       # Per-component removal results
│   ├── signal_ablation.csv          # Per-signal removal results
│   ├── threshold_sweep_offline.csv  # F1 vs threshold T=0.35→1.0
│   └── *.png                        # Visualisation charts (7 figures)
├── evaluation_summary.csv     # Aggregate results (n=300): F1, EM, hallucination, cost
├── evaluation_results.csv     # Per-sample results
└── guard_rag_neurips.tex      # NeurIPS 2026 paper (single-file, Overleaf-ready)
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your OpenRouter API key
```bash
export OPENROUTER_API_KEY=your_key_here
```
Or create a `.env` file:
```
OPENROUTER_API_KEY=your_key_here
```
Get a free key at [openrouter.ai](https://openrouter.ai).

### 3. Add the dataset
Download TAT-QA from the [official repo](https://github.com/NExTplusplus/TAT-QA) and place:
```
data/tatqa_dataset_train.json
```

### 4. Run

**Full evaluation (reproduces main results):**
```bash
python run.py
```
Outputs: `evaluation_results.csv`, `evaluation_summary.csv`

**Ablation study (offline — no API calls needed, uses existing CSV):**
```bash
python ablation.py --phase 1
```

**Signal ablation (online — requires API):**
```bash
python ablation.py --phase 2
```

**Threshold sweep (online):**
```bash
python ablation.py --phase 3
```

> **Note:** Pre-computed results for all ablations are already included in `ablation_results/` and `evaluation_summary.csv`. You do not need to re-run the evaluation to see the results.

---

## Models

| Role | Model | Provider |
|------|-------|----------|
| Baseline & Refinement generation | `gpt-4o-mini` | OpenRouter |
| GUARD-RAG Skeptic | `gpt-4o-mini` | OpenRouter |
| GUARD-RAG Grounder, Adjudicator, PoT | `gpt-4o` | OpenRouter |
| Adjudicator reasoning | DSPy `ChainOfThought` | wraps GPT-4o |
| Embeddings | `all-MiniLM-L6-v2` | HuggingFace |
| NLI Grounding | `cross-encoder/nli-deberta-v3-small` | HuggingFace |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| dspy | 3.1.3 | ChainOfThought adjudicator reasoning traces |
| faiss-cpu | 1.13.2 | Vector similarity search (FAISS flat index) |
| openai | 2.30.0 | OpenRouter API client (GPT-4o-mini, GPT-4o) |
| sentence-transformers | 4.0.2 | Embeddings (`all-MiniLM-L6-v2`) + NLI (`nli-deberta-v3-small`) |
| torch | 2.6.0 | Backend for sentence-transformers models |
| numpy | 1.26.4 | Numeric operations |
| pandas | 2.1.0 | Results CSV handling and analysis |
| transformers | 4.57.3 | HuggingFace model loading |
| tqdm | 4.67.1 | Evaluation progress bars |
| python-dotenv | 1.0.1 | `.env` API key loading |
| scikit-learn | 1.6.1 | Cosine similarity utilities |

---

## Dataset

**TAT-QA** — Financial QA over hybrid tables + text from annual reports (Zhu et al., ACL 2021).

Run configuration (`run.py`):
- `MAX_SAMPLES = 300` — documents loaded
- `EVAL_SAMPLES = 150` — questions evaluated per run
- `TOP_K = 6` — retrieved chunks for T1/T2 (T3 uses `TOP_K × 2 = 12`)
- `RANDOM_SEED = 42`

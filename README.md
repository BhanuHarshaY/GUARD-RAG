# GUARD-RAG
### Gatekeeper-Guided Debate for Hallucination Reduction in Financial QA

GUARD-RAG is a three-tier Retrieval-Augmented Generation system that reduces hallucinations in financial question answering. Agents are embedded inside the RAG pipeline, allocating compute adaptively based on answer confidence.

**Team:** Shrivarshini Narayanan, Shivam Singh, Bhanu Harsha Yanamadala
**Course:** CS6180, Northeastern University

---

## Three-Tier Architecture

| Tier | System | Description |
|------|--------|-------------|
| T1 | Baseline RAG | Single LLM call with retrieved context. No validation. |
| T2 | Refinement RAG | Formula-error detection + claim-level self-refinement. |
| T3 | GUARD-RAG | Weighted gatekeeper + asymmetric Skeptic/Grounder debate + Adjudicator. |

---

## How Each Tier Works

### Tier 1 — Baseline RAG (`tiers/baseline.py`)

Retrieves top-k chunks and issues a single LLM call with a strict grounding prompt.

- Model: `gpt-4o-mini` via OpenRouter
- Temperature: `0.0` (deterministic)
- Prompt rules: answer with only facts from evidence, no filler phrases

**Output:** `{answer, latency, tokens}`

---

### Tier 2 — Refinement RAG (`tiers/refinement.py`)

Builds on the Tier 1 answer with a two-phase refinement pass:

1. Detects formula or arithmetic errors in the T1 answer
2. Removes any claims not directly supported by the retrieved evidence

**Output:** `{answer, latency, tokens}`

---

### Tier 3 — GUARD-RAG (`tiers/guardrag.py`)

Three sub-components run in sequence:

#### 3a. Weighted Gatekeeper (`gatekeeper_v4`)

Evaluates the Tier 2 draft answer using 6 weighted heuristic + NLI signals. Debate is triggered only when `gatekeeper_score >= threshold` (default: `0.35`).

| Signal | Weight | Fires when... |
|--------|--------|---------------|
| `heuristic_arithmetic` | 0.60 | Arithmetic keyword detected (percentage change, difference, ratio, sum, etc.) |
| `heuristic_multispan` | 0.50 | Multiple entities/years requested (respectively, each of, both, etc.) |
| `heuristic_insufficient_info` | 0.40 | Answer says "insufficient information" but structured table chunks exist |
| `nli_low_grounding` | 0.30 | NLI entailment score < 0.35 (DeBERTa-v3-small) |
| `heuristic_missing_rows` | 0.30 | List-like question with ≥2 row labels, but fewer than 2 mentioned in answer |
| `heuristic_too_long` | 0.20 | Answer exceeds 80 words |

NLI model: `cross-encoder/nli-deberta-v3-small`

If no signals fire, the Tier 2 answer is returned directly — no debate, minimal cost.

#### 3b. Asymmetric Debate

Only runs when the gatekeeper triggers:

**Skeptic** (`gpt-4o-mini`) — Reviews the draft answer against retrieved evidence; lists each claim NOT directly supported.

**Grounder** (`gpt-4o`) — Responds to each challenged claim with `SUPPORTED: <claim> — Evidence: <exact quote>` or `CONCEDED: <claim>`. Cited quotes are fuzzy-matched against actual context; unverifiable citations are flipped to CONCEDED.

**Adjudicator** (`gpt-4o` + DSPy `ChainOfThought`) — Reads the full debate transcript, generates an explicit reasoning trace, and produces a final verdict:
- `APPROVE` — all claims supported, return draft unchanged
- `REVISE` — some claims conceded, return revised answer
- `ABSTAIN` — all claims conceded, return "Insufficient information."

#### 3c. Program-of-Thought (PoT) Advisory

For arithmetic questions, a Python code snippet is generated and executed to produce a verified numerical answer. This is passed to the Adjudicator as an advisory signal (not an override).

**Output:** `{answer, latency, tokens, debate_triggered, adjudicator_verdict, gatekeeper_signals, gatekeeper_score, threshold}`

---

## Retrieval Pipeline (`retrieval/retriever.py`)

**Step 1 — Dense retrieval:**
- Encodes the question with `all-MiniLM-L6-v2`
- Searches FAISS `IndexFlatIP` with `candidate_k=80` candidates

**Step 2 — Lexical reranking:**
- Base score: cosine similarity from FAISS
- `+0.30` for lexical overlap between question and chunk text
- `+0.08` for table chunks (`table_cell`, `table_row`, `table_section`)
- `+0.10` additional boost for table cells/rows on list-like questions
- `+0.15` each for row label or section label overlap with the question
- `+0.10` year boost for year-mention questions

**Step 3 — Section expansion:**
- Top-k results seed a section set
- Up to 10 additional table rows/cells from the same section are appended

T3 (GUARD-RAG) retrieves `top_k * 2` chunks for broader context; T1/T2 use `top_k`.

---

## Indexing (`indexing/`)

**Baseline** (`use_table_aware_chunking=False`): tables serialized as flat text blobs.

**Improved** (`use_table_aware_chunking=True`): tables parsed into structured chunks:
- `table_section` — full table as text block
- `table_row` — one row with `row_label`, `column_header`, `value` metadata
- `table_cell` — individual cell with full column context

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| F1 | Token-level F1 against gold answers |
| Exact Match (EM) | Normalized string match |
| Hallucination Rate | Fraction of answer sentences NOT entailed by retrieved context (NLI-based) |
| Debate Rate | Fraction of queries where gatekeeper triggered debate |
| Abstention / Approve / Revise Rate | Adjudicator verdict distribution |
| Latency | End-to-end seconds per query |
| Token Cost | Total tokens per query |

---

## Results (n=300, seed=42, TAT-QA train split)

| System | F1 | EM | Hallucination Rate | Avg Tokens | Debate Rate |
|--------|----|----|--------------------|-----------:|-------------|
| Baseline (T1) | 0.561 | 0.507 | 0.728 | 1,102 | — |
| Refinement (T2) | 0.593 | 0.539 | 0.733 | 2,395 | — |
| GUARD-RAG (T3) | **0.629** | **0.576** | **0.710** | 6,101 | 52.1% |

GUARD-RAG achieves **+12.1% relative F1** over Baseline and the **lowest hallucination rate** of all three tiers. Refinement increases hallucination (+0.005) while GUARD-RAG decreases it (−0.018 vs Baseline). The largest gains fall on arithmetic questions (+13.2% F1 over Baseline).

---

## Project Structure

```
GUARD-RAG/
├── config.py                  # API client, model names, embed/NLI model singletons
├── run.py                     # Main runner: builds indexes, evaluates, saves CSVs
├── pipeline.py                # run_comparison — single question walkthrough
├── sweep_threshold.py         # Threshold sweep: collect (API) + sweep (offline)
├── ablation.py                # Signal ablation experiments
├── analysis.py                # Post-hoc analysis utilities
├── data/
│   └── loader.py              # TAT-QA dataset loading and preprocessing
├── indexing/
│   ├── table_parser.py        # Financial table → structured chunks
│   └── vector_store.py        # FAISS index construction
├── retrieval/
│   └── retriever.py           # Dense retrieval + lexical reranking + section expansion
├── tiers/
│   ├── llm_utils.py           # ask_llm (retry/backoff), format_context helpers
│   ├── baseline.py            # Tier 1: basic RAG
│   ├── refinement.py          # Tier 2: refinement RAG
│   ├── guardrag.py            # Tier 3: gatekeeper + debate + adjudicator
│   └── pot.py                 # Program-of-Thought arithmetic verification
├── evaluation/
│   ├── metrics.py             # compute_f1, compute_em, compute_hallucination_rate
│   └── evaluator.py           # evaluate_all, summarize_results
├── ablation_results/          # Pre-computed ablation CSVs and charts
│   ├── ablation_comprehensive.csv  # 69-row full ablation (12 studies)
│   ├── answer_type_breakdown.csv   # Arithmetic vs span F1
│   ├── component_ablation.csv      # Per-component removal results
│   ├── signal_ablation.csv         # Per-gatekeeper-signal removal results
│   ├── threshold_sweep_offline.csv # F1 vs threshold T=0.35→1.0
│   └── *.png                       # Visualisation charts
├── evaluation_summary.csv     # Aggregate results (n=300)
├── evaluation_results.csv     # Per-sample results
└── guard_rag_neurips.tex      # NeurIPS 2026 paper (Overleaf-ready)
```

---

## Dataset

**TAT-QA** — financial QA over hybrid tables + text from annual reports.

Download from the [official repo](https://github.com/NExTplusplus/TAT-QA) and place at:
```
data/tatqa_dataset_train.json
```

Run configuration (in `run.py`):
- `MAX_SAMPLES = 300` — samples loaded from dataset
- `EVAL_SAMPLES = 150` — samples evaluated per run
- `TOP_K = 6` — retrieved chunks per query (T3 uses `TOP_K * 2 = 12`)
- `RANDOM_SEED = 42`

---

## Models

| Role | Model | Provider |
|------|-------|----------|
| Baseline & Refinement generation | `gpt-4o-mini` | OpenRouter |
| GUARD-RAG Skeptic | `gpt-4o-mini` | OpenRouter |
| GUARD-RAG Grounder & Adjudicator | `gpt-4o` | OpenRouter |
| Embeddings | `all-MiniLM-L6-v2` | HuggingFace |
| NLI Grounding | `cross-encoder/nli-deberta-v3-small` | HuggingFace |

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
Or add it to a `.env` file:
```
OPENROUTER_API_KEY=your_key_here
```
Get a key at [openrouter.ai](https://openrouter.ai).

### 3. Add the dataset
Place `tatqa_dataset_train.json` in the `data/` directory.

### 4. Run

**Full evaluation:**
```bash
python run.py
```
Outputs: `evaluation_results.csv`, `evaluation_summary.csv`

**Threshold sweep:**
```bash
python sweep_threshold.py --mode collect   # API calls, saves sweep_data.csv
python sweep_threshold.py --mode sweep     # offline analysis
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| dspy | 3.1.3 | ChainOfThought adjudicator reasoning |
| faiss-cpu | 1.13.2 | Vector similarity search |
| openai | 2.30.0 | OpenRouter API client (GPT-4o-mini, GPT-4o) |
| sentence-transformers | 4.0.2 | Embeddings (`all-MiniLM-L6-v2`) + NLI (`nli-deberta-v3-small`) |
| torch | 2.6.0 | Backend for sentence-transformers |
| numpy | 1.26.4 | Numeric operations |
| pandas | 2.1.0 | Results CSV handling |
| transformers | 4.57.3 | HuggingFace model loading |
| tqdm | 4.67.1 | Progress bars |
| python-dotenv | 1.0.1 | `.env` API key loading |
| scikit-learn | 1.6.1 | Cosine similarity utilities |

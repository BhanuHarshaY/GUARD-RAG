# GUARD-RAG
### Gatekeeper-Guided Debate for Hallucination Reduction in Financial QA

GUARD-RAG is an agentic Retrieval-Augmented Generation system that reduces hallucinations **by design** — not by detecting them after the fact. Agents are embedded inside the RAG pipeline itself, allocating compute adaptively based on answer confidence.

---

## The Core Idea

Standard RAG gives the LLM retrieved context and hopes for the best. GUARD-RAG adds three layers of protection:

1. **Adaptive Gatekeeper** — evaluates confidence of the initial answer using heuristic signals and NLI grounding. High confidence → pass through directly (cheap). Low confidence → escalate to debate (expensive only when needed).
2. **Asymmetric Debate** — a Skeptic agent challenges unsupported claims; a Grounder agent defends each claim with direct evidence citations. Citations are verified programmatically before reaching the Adjudicator.
3. **Adjudicator** — reads the full debate exchange and produces a final verdict: `[APPROVE]`, `[REVISE]`, or `[ABSTAIN]`.

---

## Three-Tier Architecture

| Tier | System | Description |
|------|--------|-------------|
| 1 | Basic RAG | Single LLM call with retrieved context. No validation. |
| 2 | Self-Consistency + Self-Refine | k=3 independent samples at temperature=0.7, majority vote via pairwise token-F1, then a refinement pass to remove unsupported claims. |
| 3 | GUARD-RAG (ours) | Weighted gatekeeper + asymmetric Skeptic/Grounder debate + Adjudicator verdict. |

---

## How Each Tier Works

### Tier 1 — Basic RAG (`tiers/tier1.py`)

Retrieves top-k chunks and issues a single LLM call with a strict grounding prompt.

- Model: `llama-3.1-8b-instant` via Groq
- Temperature: `0.0` (deterministic)
- Max tokens: `220`
- Prompt rules: answer with only facts from evidence, no filler phrases, no document IDs

**Output:** `{answer, latency, tokens}`

---

### Tier 2 — Self-Consistency + Self-Refine (`tiers/tier2.py`)

Two-phase process:

**Phase 1 — Self-Consistency:**
- Generates `k=3` independent answers at `temperature=0.7`
- Selects the consensus answer via pairwise token-F1 majority vote (`_majority_vote`)
- The candidate with the highest average F1 against all others wins

**Phase 2 — Self-Refine:**
- Passes the consensus answer through a refinement prompt at `temperature=0.0`
- Removes any claims not supported by the retrieved evidence
- Falls back to `"Insufficient information."` if nothing is supported

**Output:** `{answer, initial_answer, sc_candidates, latency, tokens}`

---

### Tier 3 — GUARD-RAG: Gatekeeper + Asymmetric Debate (`tiers/tier3.py`)

Three sub-components run in sequence:

#### 3a. Weighted Gatekeeper (`gatekeeper_v4`)

Evaluates the Tier 1 draft answer using four signals. Each signal has a weight; signals are summed into a single `gatekeeper_score`. Debate is triggered only when `gatekeeper_score >= threshold` (default: `0.20`).

| Signal | Weight | Fires when... |
|--------|--------|---------------|
| `heuristic_insufficient_info` | 0.40 | Answer says "insufficient information" but retrieved rows exist |
| `heuristic_missing_rows` | 0.30 | List-like question with ≥2 row labels, but fewer than 2 are mentioned in the answer |
| `heuristic_too_long` | 0.20 | Answer is longer than 80 words (may be over-generating) |
| `nli_low_grounding` | 0.50 | NLI model scores fewer than 50% of answer sentences as entailed by context |

NLI model: `cross-encoder/nli-deberta-v3-small` (label index 1 = entailment).

If no signals fire, the Tier 1 answer is returned directly — no debate, minimal cost.

#### 3b. Asymmetric Debate

Only runs when the gatekeeper triggers. Three agents run in sequence:

**Skeptic** (`llama-3.1-8b-instant`)
- Reviews the draft answer against the retrieved evidence
- Lists each claim that is NOT directly supported by the evidence
- Receives the fired gatekeeper signals as additional context ("Concerns flagged by automated checks")
- Short-circuit: if Skeptic finds no unsupported claims, the draft is approved immediately — Grounder and Adjudicator are skipped

**Grounder** (`llama-3.1-8b-instant`)
- Responds to each challenged claim from the Skeptic
- Labels each claim `SUPPORTED: <claim> — Evidence: <exact quote>` or `CONCEDED: <claim>`
- Post-processing: cited quotes are fuzzy-matched against the actual context; SUPPORTED verdicts with unverifiable citations are flipped to CONCEDED

**Adjudicator** (`llama-3.3-70b-versatile`)
- Reads the full debate transcript
- Produces a final verdict prefixed with `[APPROVE]`, `[REVISE]`, or `[ABSTAIN]`
  - `APPROVE`: all claims were supported — return draft answer unchanged
  - `REVISE`: some claims were conceded — return revised answer using only supported claims
  - `ABSTAIN`: all claims conceded — return `"Insufficient information."`

**Output:** `{answer, latency, tokens, debate_triggered, adjudicator_verdict, gatekeeper_signals, gatekeeper_score, threshold, debate_transcript: {skeptic, grounder}}`

---

## Retrieval Pipeline (`retrieval/retriever.py`)

Two retrieval modes are compared — **Baseline** (text-only) vs **Improved** (table-aware):

**Step 1 — Dense retrieval:**
- Encodes the question with `all-MiniLM-L6-v2`
- Searches FAISS `IndexFlatIP` with `candidate_k=40` candidates

**Step 2 — Lexical reranking (`rerank_candidates`):**
- Base score: cosine similarity from FAISS
- `+0.30` for lexical overlap between question and chunk text (stopwords removed)
- `+0.08` for table chunks (`table_cell`, `table_row`, `table_section`)
- `+0.10` additional boost for table cells/rows on list-like questions
- `+0.15` each for row label or section label overlap with the question

**Step 3 — Section expansion (`expand_same_section`):**
- Top-k reranked results seed a section set
- Up to 10 additional table rows/cells from the same section are appended
- Ensures full table context is available even when only one row scored highly

**Final output:** deduplicated list of up to `top_k + 10` chunks with scores and metadata.

---

## Indexing (`indexing/`)

**Baseline** (`use_table_aware_chunking=False`): tables are serialized as flat text blobs — one chunk per table.

**Improved** (`use_table_aware_chunking=True`): tables are parsed into structured chunks:
- `table_section` — full table as a text block
- `table_row` — one row with `row_label`, `column_header`, `value` metadata
- `table_cell` — individual cell with full column context

Each chunk carries metadata: `doc_id`, `chunk_id`, `chunk_type`, `section_label`, `row_label`, `column_header`, `value`.

---

## Evaluation Metrics (`evaluation/`)

| Metric | Description |
|--------|-------------|
| F1 | Token-level F1 against gold answers (articles/punctuation normalized) |
| Exact Match (EM) | Normalized string match |
| Hallucination Rate | Fraction of answer sentences NOT entailed by retrieved context (NLI-based, `cross-encoder/nli-deberta-v3-small`) |
| Debate Rate | Fraction of queries where gatekeeper triggered debate |
| Gatekeeper Score | Mean weighted signal score across queries |
| Abstention Rate | Fraction of Adjudicator verdicts that were `ABSTAIN` |
| Approve / Revise Rate | Fraction of debates ending in `APPROVE` vs `REVISE` |
| Latency | End-to-end seconds per query |
| Token Cost | Total tokens per query, per tier |

---

## Project Structure

```
GUARD-RAG/
├── config.py                  # Model names, API client, embed_model, nli_model singletons
├── run.py                     # Main runner: builds indexes, evaluates all tiers, saves CSVs
├── pipeline.py                # run_comparison — single question walkthrough across all tiers
├── sweep_threshold.py         # Threshold sweep: collect mode (API) + sweep mode (offline)
├── data/
│   └── loader.py              # TAT-QA dataset loading and preprocessing
├── indexing/
│   ├── table_parser.py        # Financial table → structured chunks (row/cell/section)
│   └── vector_store.py        # FAISS index construction (baseline + table-aware)
├── retrieval/
│   └── retriever.py           # Dense retrieval + lexical reranking + section expansion
├── tiers/
│   ├── llm_utils.py           # ask_llm (with retry/backoff), format_context, strip_prefix
│   ├── tier1.py               # Basic RAG — single LLM call
│   ├── tier2.py               # Self-consistency (k=3 majority vote) + self-refine
│   └── tier3.py               # Gatekeeper v4 + Skeptic + Grounder + Adjudicator
├── evaluation/
│   ├── metrics.py             # compute_f1, compute_em, compute_hallucination_rate
│   └── evaluator.py           # evaluate_all, summarize_results
└── notebooks/
    └── experiments.ipynb      # Exploratory notebook
```

---

## Dataset

**TAT-QA** — financial QA over hybrid tables + text from annual reports.

Working split (from `tatqa_dataset_train.json`):
- 100 samples loaded (`MAX_SAMPLES=100`)
- 15 samples evaluated per run (`EVAL_SAMPLES=15`)
- Threshold sweep uses samples 200–250 from the full dataset

---

## Models

| Role | Model | Provider |
|------|-------|----------|
| Generation — Tier 1 & 2 | `llama-3.1-8b-instant` | Groq |
| Skeptic + Grounder (Tier 3) | `llama-3.1-8b-instant` | Groq |
| Adjudicator (Tier 3) | `llama-3.3-70b-versatile` | Groq |
| Embeddings | `all-MiniLM-L6-v2` | HuggingFace (sentence-transformers) |
| NLI Grounding | `cross-encoder/nli-deberta-v3-small` | HuggingFace (sentence-transformers) |

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Groq API key
```bash
export GROQ_API_KEY=your_key_here
```
Get a free key at [console.groq.com](https://console.groq.com).

### 3. Add the dataset
Download TAT-QA from the [official repo](https://github.com/NExTplusplus/TAT-QA) and place:
```
data/tatqa_dataset_train.json
data/tatqa_dataset_dev.json
```

### 4. Run

**Full evaluation (all tiers, baseline vs improved):**
```bash
python3 run.py
```
Outputs: `evaluation_baseline.csv`, `evaluation_improved.csv`, `evaluation_summary.csv`

**Threshold sweep (collect then analyze):**
```bash
python3 sweep_threshold.py --mode collect   # runs API calls, saves sweep_data.csv
python3 sweep_threshold.py --mode sweep     # offline analysis, generates sweep_plot.png
```

---

## Implementation Status

- [x] Basic RAG pipeline (Tier 1)
- [x] Self-consistency majority vote, k=3 (Tier 2)
- [x] Self-refine unsupported claim removal (Tier 2)
- [x] Table-aware chunking (row / cell / section)
- [x] Hybrid retrieval: dense + lexical reranking + section expansion
- [x] Weighted gatekeeper (heuristic signals + NLI grounding, `gatekeeper_v4`)
- [x] Asymmetric debate: Skeptic + Grounder + Adjudicator (Tier 3)
- [x] Grounder citation verification (fuzzy match, flips unverifiable citations to CONCEDED)
- [x] Adjudicator verdict labels: `[APPROVE]` / `[REVISE]` / `[ABSTAIN]`
- [x] Gatekeeper signals passed to Skeptic for targeted claim checking
- [x] Hallucination rate metric (NLI-based, per tier)
- [x] Threshold sweep script with signal ablation
- [x] Rate-limit retry with exponential backoff in `ask_llm`
- [ ] DSPy-optimized Adjudicator (BootstrapFewShot)
- [ ] Pareto frontier plot (Accuracy vs Token Cost)

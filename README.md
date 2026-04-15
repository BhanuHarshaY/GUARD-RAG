# GUARD-RAG
### Gatekeeper-Guided Debate for Hallucination Reduction in Financial QA

**Team:** Shrivarshini Narayanan, Shivam Singh, Bhanu Harsha Yanamadala  
**Course:** CS6180, Northeastern University

---

## What is this?

Financial QA over tables and text is tricky. Questions can require multi-row arithmetic, percentage calculations, or extracting values that span across years. A single LLM call gets these wrong more often than not, either by hallucinating values or applying the wrong formula.

GUARD-RAG is a three-tier pipeline that escalates difficult questions through progressively stronger verification. The idea is simple: cheap questions get cheap answers, hard questions get debated. A gatekeeper decides which is which using six signals, and only 52.1% of questions end up going through the full debate path.

On TAT-QA (300 questions), the final system hits **F1 = 0.629** vs 0.561 for basic RAG (+12.1%). More interestingly, it has the lowest hallucination rate of the three tiers, while self-refinement (Tier 2) actually makes hallucination worse.

---

## Results

**Main results (n=300, TAT-QA train split, seed=42):**

| System | F1 | EM | Hallucination | Tokens | Latency (s) | Debate Rate |
|--------|----|----|--------------|--------|-------------|-------------|
| Baseline RAG (T1) | 0.561 | 0.507 | 0.728 | 1,102 | 0.877 | N/A |
| Refinement (T2) | 0.593 | 0.539 | 0.733 | 2,395 | 2.339 | N/A |
| GUARD-RAG (T3) | **0.629** | **0.576** | **0.710** | 6,101 | 5.630 | 52.1% |

Refinement actually increases hallucination (0.733 vs 0.728). GUARD-RAG brings it down to 0.710 despite generating 5.5x more tokens. The gains are biggest on arithmetic questions (+13.2% F1) and minimal on text/span questions (+0.8%), which is intentional.

**By question type (n=150 annotated split):**

| Type | n | Baseline | Refinement | GUARD-RAG | Gain |
|------|---|----------|------------|-----------|------|
| Arithmetic | 52 | 0.538 | 0.558 | 0.609 | +13.2% |
| Text / Span | 98 | 0.628 | 0.628 | 0.633 | +0.8% |

---

## How it works

### Tier 1 - Baseline RAG (`tiers/baseline.py`)

Single GPT-4o-mini call with k=6 retrieved chunks. For arithmetic questions it uses a chain-of-thought prompt and extracts the final number. For text questions it uses a grounding prompt that enforces evidence-only answers. Temperature is 0.0 throughout.

### Tier 2 - Refinement (`tiers/refinement.py`)

Takes the Tier 1 answer and runs it through a formula-error verification prompt. Three few-shot examples cover the most common mistakes: sum vs average confusion, wrong percentage direction (old-new vs new-old), and a correct draft that should not be changed. A magnitude guard blocks any revision where the ratio between old and new answer exceeds 100x.

### Tier 3 - GUARD-RAG (`tiers/guardrag.py`)

**Gatekeeper v4** scores the Tier 2 answer using six signals. If the score is below 0.35, the Tier 2 answer goes straight through. If it hits 0.35 or above, the full debate runs.

| Signal | Weight | What triggers it |
|--------|--------|-----------------|
| `heuristic_arithmetic` | 0.60 | Arithmetic keywords: percentage change, difference, ratio, sum, grew by, fell by |
| `heuristic_multispan` | 0.50 | Multi-value keywords: respectively, each of, both, each year |
| `heuristic_insufficient_info` | 0.40 | Answer says insufficient information but table chunks were retrieved |
| `nli_low_grounding` | 0.30 | NLI entailment score below 0.35 (DeBERTa-v3-small) |
| `heuristic_missing_rows` | 0.30 | List question with 2+ row labels but fewer than 2 appear in the answer |
| `heuristic_too_long` | 0.20 | Answer is over 80 words |

The threshold of 0.35 means any arithmetic question (score 0.6) and most multispan questions (score 0.5) always get debated. Text questions typically score 0.0-0.3 and get skipped.

**Dual retrieval:** Tier 1 and 2 use k=6 chunks. Tier 3 uses k=12 from the same index. Financial tables can span 20+ rows and a percentage-change question needs both the numerator and denominator rows, which often don't both fit in k=6.

**GPT-4o answer generation:** For triggered questions, GUARD-RAG generates a fresh answer with GPT-4o over the 12-chunk context. Arithmetic questions get structured key-value facts (Row [Section] (Column): Value format) for cleaner number extraction.

**Program-of-Thought (`tiers/pot.py`):** For arithmetic questions, Python code gets generated and executed to compute the answer from the structured facts. This result is passed to the adjudicator as an advisory note. We tried using it as a direct override but that dropped F1 by 4.5% because extraction errors propagate. As an advisory signal the adjudicator can weigh it against the debate.

**The debate itself** runs three agents:

*Skeptic (GPT-4o-mini)* - Reviews the draft against evidence and lists unsupported claims. It only challenges a numeric value if it can point to a different specific value in the evidence. If nothing is challenged, the answer is approved without running the other two agents.

*Grounder (GPT-4o)* - For each challenged claim, either provides an exact evidence quote (SUPPORTED) or concedes (CONCEDED). A citation verifier checks that quoted text actually appears in the context; quotes that don't match get flipped to CONCEDED automatically. If nothing is conceded, the draft is approved without the adjudicator.

*Adjudicator (GPT-4o + DSPy ChainOfThought)* - Reads the full debate and generates a reasoning trace before issuing a verdict: APPROVE (keep the draft), REVISE (return a corrected answer), or ABSTAIN (evidence is insufficient). For non-arithmetic questions this uses DSPy ChainOfThought for the explicit reasoning step. If the verdict is REVISE, a second debate round runs on the revised answer (this happens for about 2.3% of all questions).

---

## Retrieval (`retrieval/retriever.py`)

Starts with dense retrieval using `all-MiniLM-L6-v2` embeddings against a FAISS flat index (candidate_k=80). Then reranks with these boosts:

| Boost | When |
|-------|------|
| +0.30 | Token overlap between question and chunk text |
| +0.15 | Row label or section label matches question tokens |
| +0.10 | Table cell or row on a list-like question |
| +0.10 | Year mentioned in question appears in chunk |
| +0.08 | Chunk is a table type (table_cell, table_row, table_section) |

After reranking, section expansion adds up to 10 more rows/cells from the same table section as the top results. This ensures full table context when only one row scored highly.

---

## Ablation Study

All results are pre-computed in `ablation_results/`. No API calls needed to see them.

**Removing each component one at a time (n=150):**

| What was changed | F1 | Change |
|-----------------|-----|--------|
| Full GUARD-RAG | 0.624 | baseline |
| PoT as direct override instead of signal | 0.596 | -4.5% |
| No GPT-4o generation (debate the T2 draft) | 0.603 | -3.4% |
| T3 retrieves k=6 instead of k=12 | 0.608 | -2.6% |
| No gatekeeper, debate everything | 0.609 | -2.4% |
| Debate text/span questions too | 0.611 | -2.1% |
| Remove PoT advisory signal | 0.617 | -1.1% |
| Remove DSPy CoT from adjudicator | 0.618 | -1.0% |

**Removing each gatekeeper signal one at a time (n=15):**

| Signal removed | F1 change |
|----------------|-----------|
| heuristic_arithmetic (w=0.6) | -20.3% |
| heuristic_insufficient_info (w=0.4) | -20.3% |
| heuristic_multispan (w=0.5) | -10.2% |
| nli_low_grounding (w=0.3) | -10.2% |
| heuristic_missing_rows (w=0.3) | 0.0% |
| heuristic_too_long (w=0.2) | 0.0% |

---

## Project Structure

```
GUARD-RAG/
├── config.py                 # API client setup, model names, embedding/NLI singletons
├── run.py                    # Main evaluation script
├── pipeline.py               # Single question walkthrough (useful for debugging)
├── ablation.py               # Runs ablation studies (offline and online)
├── analysis.py               # Post-hoc analysis helpers
├── sweep_threshold.py        # Threshold sweep experiments
├── data/
│   └── loader.py             # TAT-QA loading and preprocessing
├── indexing/
│   ├── table_parser.py       # Parses financial tables into row/cell/section chunks
│   └── vector_store.py       # Builds FAISS indexes
├── retrieval/
│   └── retriever.py          # Dense retrieval, lexical reranking, section expansion
├── tiers/
│   ├── llm_utils.py          # LLM wrapper with retry/backoff and input sanitization
│   ├── baseline.py           # Tier 1
│   ├── refinement.py         # Tier 2
│   ├── guardrag.py           # Tier 3 (gatekeeper, debate, adjudicator)
│   └── pot.py                # Program-of-Thought execution
├── evaluation/
│   ├── metrics.py            # F1, EM, hallucination rate
│   └── evaluator.py          # Runs evaluation across all tiers
├── ablation_results/
│   ├── ablation_comprehensive.csv  # All ablation results (69 rows, 12 studies)
│   ├── component_ablation.csv      # Component removal results
│   ├── signal_ablation.csv         # Signal removal results
│   ├── answer_type_breakdown.csv   # Arithmetic vs span breakdown
│   ├── threshold_sweep_offline.csv # Threshold vs F1
│   └── *.png                       # Charts for all of the above
├── evaluation_summary.csv    # Aggregate n=300 results
├── evaluation_results.csv    # Per-sample results
└── guard_rag_neurips.tex     # Paper (NeurIPS 2026 format, single file for Overleaf)
```

---

## Setup

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Set API key:**
```bash
export OPENROUTER_API_KEY=your_key_here
```
Or put it in a `.env` file. Get a key at [openrouter.ai](https://openrouter.ai).

**Add the dataset:**  
Download TAT-QA from [github.com/NExTplusplus/TAT-QA](https://github.com/NExTplusplus/TAT-QA) and put `tatqa_dataset_train.json` in the `data/` folder.

**Run:**
```bash
python run.py
```
This runs all three tiers on 150 questions (from a pool of 300, seed=42) and saves `evaluation_results.csv` and `evaluation_summary.csv`.

For ablation analysis without API calls:
```bash
python ablation.py --phase 1
```

Note: pre-computed results are already in `ablation_results/` and `evaluation_summary.csv` if you just want to see the numbers.

---

## Models and Libraries

**Models used:**

| Role | Model | Via |
|------|-------|-----|
| Tier 1 and 2 generation | gpt-4o-mini | OpenRouter |
| Tier 3 Skeptic | gpt-4o-mini | OpenRouter |
| Tier 3 Grounder, Adjudicator, PoT | gpt-4o | OpenRouter |
| Adjudicator reasoning | DSPy ChainOfThought | wraps gpt-4o |
| Embeddings | all-MiniLM-L6-v2 | HuggingFace |
| NLI grounding | cross-encoder/nli-deberta-v3-small | HuggingFace |

**Python packages:**

| Package | Version | Used for |
|---------|---------|----------|
| dspy | 3.1.3 | ChainOfThought adjudicator |
| faiss-cpu | 1.13.2 | Vector search |
| openai | 2.30.0 | API client for OpenRouter |
| sentence-transformers | 4.0.2 | Embeddings and NLI model |
| torch | 2.6.0 | Backend for sentence-transformers |
| numpy | 1.26.4 | Numeric ops |
| pandas | 2.1.0 | CSV handling |
| transformers | 4.57.3 | HuggingFace model loading |
| tqdm | 4.67.1 | Progress bars |
| python-dotenv | 1.0.1 | Loading .env file |
| scikit-learn | 1.6.1 | Cosine similarity |

---

## Dataset

TAT-QA (Zhu et al., ACL 2021) is a financial QA benchmark over hybrid table+text documents from annual reports. We use the training split with 300 documents sampled at seed=42, evaluating on 150 questions per run.

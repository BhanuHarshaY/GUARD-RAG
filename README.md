# GUARD-RAG
### Gatekeeper-Guided Debate for Hallucination Reduction in Financial QA

GUARD-RAG is an agentic Retrieval-Augmented Generation system that reduces hallucinations **by design** — not by detecting them after the fact. Agents are embedded inside the RAG pipeline itself, allocating compute adaptively based on answer confidence.

---

## The Core Idea

Standard RAG gives the LLM retrieved context and hopes for the best. GUARD-RAG adds three layers of protection:

1. **Adaptive Gatekeeper** — evaluates confidence of the initial answer. High confidence → pass through directly (cheap). Low confidence → escalate to debate (expensive only when needed).
2. **Asymmetric Debate** — a Skeptic agent finds unsupported claims; a Grounder agent defends using only cited evidence from the retrieved documents.
3. **DSPy-Optimized Adjudicator** — produces a final grounded answer: Approve / Revise from evidence / Abstain.

---

## Three-Tier Comparison

| Tier | System | Description |
|------|--------|-------------|
| 1 | Basic RAG | Fixed pipeline. Single LLM call. No validation. |
| 2 | Naive Agentic RAG | Self-refine pass. Removes unsupported claims from Tier 1 draft. |
| 3 | GUARD-RAG (ours) | Gatekeeper + asymmetric debate + DSPy adjudicator. |

---

## Project Structure

```
GUARD-RAG/
├── config.py                  # Model names, API keys, shared singletons
├── data/
│   └── loader.py              # TAT-QA dataset loading and preprocessing
├── indexing/
│   ├── table_parser.py        # Financial table → structured chunks
│   └── vector_store.py        # FAISS index construction
├── retrieval/
│   └── retriever.py           # Dense retrieval + lexical reranking + section expansion
├── tiers/
│   ├── llm_utils.py           # Shared LLM helpers (ask_llm, format_context, etc.)
│   ├── tier1.py               # Basic RAG
│   ├── tier2.py               # Self-refine
│   └── tier3.py               # Gatekeeper + debate (GUARD-RAG)
├── evaluation/
│   ├── metrics.py             # F1, Exact Match scoring
│   └── evaluator.py           # evaluate_all, summarize_results
├── pipeline.py                # run_comparison — orchestrates all three tiers
└── notebooks/
    └── experiments.ipynb      # Driver notebook — no function definitions
```

---

## Dataset

Primary: **TAT-QA** — financial question answering over tables + text (annual reports).  
Optional: FinanceBench, FinQA (if time permits).

Working subset: 300–500 samples  
- Train (~200): DSPy optimization  
- Validation (~50): hyperparameter tuning  
- Test (100–200): final evaluation across all tiers  

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Answer F1 | Token-level F1 against gold answers |
| Exact Match (EM) | Normalized string match against gold |
| Hallucination Rate | % of answers containing unsupported claims (NLI-based) |
| Abstention Rate | % of queries where system returns no answer |
| Latency | End-to-end seconds per query, per tier |
| Token Cost | Total tokens per query, broken down by component |

Key output: **Pareto frontier plot** — Accuracy vs. Token Cost across all three tiers.

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/GUARD-RAG.git
cd GUARD-RAG
```

### 2. Install dependencies
```bash
pip install groq faiss-cpu sentence-transformers pandas tqdm transformers
```

### 3. Set your Groq API key
```bash
export GROQ_API_KEY=your_key_here
```
Get a free key at [console.groq.com](https://console.groq.com).

### 4. Add the dataset
Download TAT-QA from the [official repo](https://github.com/NExTplusplus/TAT-QA) and place:
```
data/tatqa_dataset_train.json
data/tatqa_dataset_dev.json
```

### 5. Run
Open `notebooks/experiments.ipynb` and run all cells.  
Or from Python:
```python
from config import client, embed_model, BASE_MODEL, JUDGE_MODEL
from data.loader import load_tatqa
from indexing.vector_store import build_vector_store
from pipeline import run_comparison

samples = load_tatqa("data/tatqa_dataset_train.json", max_samples=100)
baseline_index, baseline_chunks, baseline_metadata = build_vector_store(samples, embed_model, use_table_aware_chunking=False)
improved_index, improved_chunks, improved_metadata = build_vector_store(samples, embed_model, use_table_aware_chunking=True)

t1, t2, t3 = run_comparison(
    samples[0]["question"],
    baseline_index, baseline_chunks, baseline_metadata,
    improved_index, improved_chunks, improved_metadata,
    embed_model, client, BASE_MODEL, JUDGE_MODEL,
    mode="improved"
)
```

---

## Models Used

| Role | Model | Provider |
|------|-------|----------|
| Base generation (Tier 1 & 2) | `llama-3.1-8b-instant` | Groq |
| Judge / Debate verifier (Tier 3) | `llama-3.3-70b-versatile` | Groq |
| Embeddings | `all-MiniLM-L6-v2` | HuggingFace |

---

## Implementation Roadmap

- [x] Basic RAG pipeline (Tier 1)
- [x] Self-refine agentic RAG (Tier 2)
- [x] Gatekeeper + debate system (Tier 3)
- [x] Hybrid retrieval with section expansion
- [x] F1 / EM evaluation framework
- [ ] Self-consistency (k=3 majority vote) for Tier 2
- [ ] Signal-based gatekeeper (logprobs + NLI score)
- [ ] Asymmetric Skeptic + Grounder debate agents
- [ ] DSPy-optimized adjudicator (Approve / Revise / Abstain)
- [ ] Hallucination rate metric (NLI-based)
- [ ] Pareto frontier plot

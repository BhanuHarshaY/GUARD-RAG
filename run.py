"""
GUARD-RAG — Quick runner
Runs all three tiers on a sample of TAT-QA and prints a comparison summary.

Usage:
    export GROQ_API_KEY=your_key_here
    python run.py
"""

from config import client, embed_model, nli_model, BASE_MODEL, JUDGE_MODEL
from data.loader import load_tatqa
from indexing.vector_store import build_vector_store
from pipeline import run_comparison
from evaluation.evaluator import evaluate_all, summarize_results
import pandas as pd

DATA_PATH = "data/tatqa_dataset_train.json"
MAX_SAMPLES = 300
EVAL_SAMPLES = 50
TOP_K = 6

# ── Load data ──────────────────────────────────────────────────────────────
samples = load_tatqa(DATA_PATH, max_samples=MAX_SAMPLES)
print(f"\nLoaded {len(samples)} samples")

# ── Build indexes ──────────────────────────────────────────────────────────
baseline_index, baseline_chunks, baseline_metadata = build_vector_store(
    samples, embed_model, use_table_aware_chunking=False
)
improved_index, improved_chunks, improved_metadata = build_vector_store(
    samples, embed_model, use_table_aware_chunking=True
)

# ── Single question walkthrough ────────────────────────────────────────────
print("\n" + "=" * 70)
print("SINGLE QUESTION WALKTHROUGH")
print("=" * 70)
run_comparison(
    samples[0]["question"],
    baseline_index, baseline_chunks, baseline_metadata,
    improved_index, improved_chunks, improved_metadata,
    embed_model, client, BASE_MODEL, JUDGE_MODEL,
    mode="improved", top_k=TOP_K, nli_model=nli_model
)

# ── Full evaluation ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"FULL EVALUATION ({EVAL_SAMPLES} samples)")
print("=" * 70)

#df_baseline = evaluate_all(
  #  samples,
   # baseline_index, baseline_chunks, baseline_metadata,
    #improved_index, improved_chunks, improved_metadata,
    #embed_model, client, BASE_MODEL, JUDGE_MODEL,
    #mode="baseline", max_samples=EVAL_SAMPLES, top_k=TOP_K, nli_model=nli_model
#)#

df_improved = evaluate_all(
    samples,
    baseline_index, baseline_chunks, baseline_metadata,
    improved_index, improved_chunks, improved_metadata,
    embed_model, client, BASE_MODEL, JUDGE_MODEL,
    mode="improved", max_samples=EVAL_SAMPLES, top_k=TOP_K, nli_model=nli_model
)

summary = pd.concat([
    #summarize_results(df_baseline, "baseline"),
    summarize_results(df_improved, "improved"),
], ignore_index=True)

print("\nSUMMARY:")
print(summary.to_string(index=False))


#df_baseline.to_csv("evaluation_baseline.csv", index=False)
df_improved.to_csv("evaluation_improved.csv", index=False)
summary.to_csv("evaluation_summary.csv", index=False)
print("\nResults saved to evaluation_baseline.csv, evaluation_improved.csv, evaluation_summary.csv")

"""
DSPy Adjudicator Optimization
==============================
Uses MIPROv2 to optimize the adjudicator prompt using labeled examples
from evaluation_results.csv.

Usage:
    python dspy_adjudicator.py --compile   # optimize + save compiled prompt
    python dspy_adjudicator.py --eval      # evaluate compiled vs baseline

The compiled prompt is saved to dspy_compiled_adjudicator.json and
automatically loaded by guardrag.py if it exists.
"""

import argparse
import json
import os
import random
import pandas as pd

import dspy
from dspy import InputField, OutputField, Signature, Module, Predict
from dspy.teleprompt import MIPROv2

from evaluation.metrics import compute_f1

COMPILED_PATH = "dspy_compiled_adjudicator.json"
RANDOM_SEED   = 42


# ─────────────────────────────────────────────────────────────────────────────
# DSPy Signature
# ─────────────────────────────────────────────────────────────────────────────

class AdjudicatorSignature(Signature):
    """
    You are an expert Adjudicator for financial QA.
    Given a question, evidence, draft answer, and debate context,
    produce the best possible final answer.
    Prefix your answer with [APPROVE], [REVISE], or [ABSTAIN].
    """
    question        = InputField(desc="The financial question to answer")
    evidence        = InputField(desc="Retrieved financial document chunks")
    draft_answer    = InputField(desc="The baseline RAG draft answer")
    skeptic_output  = InputField(desc="Skeptic's challenged claims (may be empty)")
    grounder_output = InputField(desc="Grounder's verdicts (may be empty)")
    final_answer    = OutputField(desc="[APPROVE/REVISE/ABSTAIN] followed by the answer")


# ─────────────────────────────────────────────────────────────────────────────
# DSPy Module
# ─────────────────────────────────────────────────────────────────────────────

class DSPyAdjudicator(Module):
    def __init__(self):
        super().__init__()
        self.adjudicate = Predict(AdjudicatorSignature)

    def forward(self, question, evidence, draft_answer, skeptic_output="", grounder_output=""):
        return self.adjudicate(
            question=question,
            evidence=evidence,
            draft_answer=draft_answer,
            skeptic_output=skeptic_output,
            grounder_output=grounder_output,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Metric
# ─────────────────────────────────────────────────────────────────────────────

def adjudicator_metric(example, prediction, trace=None):
    """F1 score of predicted final_answer vs gold."""
    raw = prediction.final_answer or ""
    # Strip [APPROVE]/[REVISE]/[ABSTAIN] prefix
    import re
    for label in ("APPROVE", "REVISE", "ABSTAIN"):
        if raw.upper().startswith(f"[{label}]"):
            raw = raw[len(f"[{label}]"):].strip()
            break
    return compute_f1(raw, example.gold_answer)


# ─────────────────────────────────────────────────────────────────────────────
# Build training examples from evaluation_results.csv
# ─────────────────────────────────────────────────────────────────────────────

def build_examples(csv_path="evaluation_results.csv", max_train=None, max_dev=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found — run run.py first")

    df = pd.read_csv(csv_path)
    # Only use debated samples — others have no debate transcript
    df = df[df["guardrag_debate_triggered"] == True].reset_index(drop=True)

    if len(df) < 10:
        raise ValueError(f"Only {len(df)} debated samples — need at least 10")

    # Auto-size train/dev split if not specified
    n = len(df)
    if max_train is None:
        max_train = int(n * 0.75)
    if max_dev is None:
        max_dev = n - max_train

    random.seed(RANDOM_SEED)
    indices = list(range(len(df)))
    random.shuffle(indices)

    train_examples, dev_examples = [], []
    for i, idx in enumerate(indices):
        row = df.iloc[idx]
        ex = dspy.Example(
            question=row["question"],
            evidence="",          # not stored in CSV — adjudicator uses context at runtime
            draft_answer=row["baseline_answer"],
            skeptic_output="",
            grounder_output="",
            gold_answer=row["gold_answer"],
        ).with_inputs("question", "evidence", "draft_answer", "skeptic_output", "grounder_output")

        if i < max_train:
            train_examples.append(ex)
        elif i < max_train + max_dev:
            dev_examples.append(ex)

    print(f"  Train: {len(train_examples)} | Dev: {len(dev_examples)} debated examples")
    return train_examples, dev_examples


# ─────────────────────────────────────────────────────────────────────────────
# Compile
# ─────────────────────────────────────────────────────────────────────────────

def compile_adjudicator():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    lm = dspy.LM(
        model="openai/gpt-4o",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        temperature=0.0,
        max_tokens=350,
    )
    dspy.configure(lm=lm)

    train_examples, dev_examples = build_examples()
    print(f"  Using {len(train_examples)} train / {len(dev_examples)} dev examples")

    adjudicator = DSPyAdjudicator()

    print("Running MIPROv2 optimization...")
    optimizer = MIPROv2(
        metric=adjudicator_metric,
        num_candidates=5,
        init_temperature=0.7,
        verbose=True,
    )

    compiled = optimizer.compile(
        adjudicator,
        trainset=train_examples,
        valset=dev_examples,
        num_trials=10,
        max_bootstrapped_demos=2,
        max_labeled_demos=3,
    )

    compiled.save(COMPILED_PATH)
    print(f"\nSaved compiled adjudicator → {COMPILED_PATH}")
    return compiled


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate compiled vs baseline
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_compiled():
    from config import client
    import os

    if not os.path.exists(COMPILED_PATH):
        print(f"{COMPILED_PATH} not found — run with --compile first")
        return

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    lm = dspy.LM(
        model="openai/gpt-4o",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        temperature=0.0,
        max_tokens=350,
    )
    dspy.configure(lm=lm)

    _, dev_examples = build_examples()

    baseline = DSPyAdjudicator()
    compiled = DSPyAdjudicator()
    compiled.load(COMPILED_PATH)

    evaluator = dspy.Evaluate(devset=dev_examples, metric=adjudicator_metric, display_progress=True)

    print("\nBaseline adjudicator:")
    base_score = evaluator(baseline)

    print("\nCompiled adjudicator:")
    comp_score = evaluator(compiled)

    print(f"\nBaseline F1: {base_score:.4f}")
    print(f"Compiled F1: {comp_score:.4f}")
    print(f"Delta:       {comp_score - base_score:+.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Run MIPROv2 optimization")
    parser.add_argument("--eval",    action="store_true", help="Evaluate compiled vs baseline")
    args = parser.parse_args()

    if args.compile:
        compile_adjudicator()
    elif args.eval:
        evaluate_compiled()
    else:
        print("Usage: python dspy_adjudicator.py --compile | --eval")

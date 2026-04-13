"""
GUARD-RAG — Threshold Sweep Script
====================================
Two modes:

  collect  — calls the API once, saves sweep_data.csv
  sweep    — offline analysis of sweep_data.csv, no API calls

Usage:
    python sweep_threshold.py --mode collect
    python sweep_threshold.py --mode sweep
    python sweep_threshold.py          # auto-selects based on whether sweep_data.csv exists
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH       = "data/tatqa_dataset_train.json"
SWEEP_DATA_PATH = "sweep_data.csv"
SWEEP_RESULTS_PATH = "sweep_results.csv"
SWEEP_PLOT_PATH = "sweep_plot.png"

MAX_SAMPLES     = 300
TRAIN_SIZE      = 200
VAL_SIZE        = 50            # samples[200:250]
TOP_K           = 6
THRESHOLD_MIN   = 0.05
THRESHOLD_MAX   = 0.95
THRESHOLD_STEP  = 0.05
ELBOW_RATE_JUMP = 0.15          # >15 pp trigger-rate increase = elbow
ELBOW_F1_GAIN   = 0.01          # <0.01 F1 gain despite rate jump = elbow

ALL_SIGNALS = [
    "heuristic_insufficient_info",
    "heuristic_missing_rows",
    "heuristic_too_long",
    "nli_low_grounding",
]


# ---------------------------------------------------------------------------
# Helpers shared between modes
# ---------------------------------------------------------------------------

def _thresholds():
    return np.round(np.arange(THRESHOLD_MIN, THRESHOLD_MAX + 1e-9, THRESHOLD_STEP), 2)


def _sweep_row(df, threshold, score_col="gatekeeper_score"):
    """Given a DataFrame with per-sample data, compute metrics at one threshold."""
    triggered = df[score_col] >= threshold
    effective_f1     = np.where(triggered, df["debated_f1"],   df["tier1_f1"])
    effective_tokens = np.where(triggered,
                                df["tier1_tokens"] + df["debate_tokens"],
                                df["tier1_tokens"])
    return {
        "threshold":     threshold,
        "mean_f1":       round(float(effective_f1.mean()), 4),
        "trigger_rate":  round(float(triggered.mean()), 4),
        "mean_tokens":   round(float(effective_tokens.mean()), 1),
    }


def _find_elbow(sweep_df):
    """
    Identify the highest threshold where:
      - lowering threshold by one step increases trigger rate > ELBOW_RATE_JUMP
      - but improves F1 by < ELBOW_F1_GAIN
    Returns the recommended threshold (the higher one before the elbow jump).
    Falls back to the threshold with max F1 if no elbow found.
    """
    rows = sweep_df.sort_values("threshold", ascending=False).reset_index(drop=True)
    for i in range(len(rows) - 1):
        rate_jump = rows.loc[i + 1, "trigger_rate"] - rows.loc[i, "trigger_rate"]
        f1_gain   = rows.loc[i + 1, "mean_f1"]     - rows.loc[i, "mean_f1"]
        if rate_jump > ELBOW_RATE_JUMP and f1_gain < ELBOW_F1_GAIN:
            return float(rows.loc[i, "threshold"])   # stay at higher threshold
    # fallback: max F1
    return float(sweep_df.loc[sweep_df["mean_f1"].idxmax(), "threshold"])


# ---------------------------------------------------------------------------
# Mode 1: collect
# ---------------------------------------------------------------------------

def run_collect():
    print("=" * 60)
    print("MODE: collect")
    print("=" * 60)

    # ── imports (API-touching code only loaded in collect mode) ──────────
    from config import client, embed_model, nli_model, BASE_MODEL, JUDGE_MODEL
    from data.loader import load_tatqa
    from indexing.vector_store import build_vector_store
    from retrieval.retriever import retrieve
    from tiers.baseline import baseline_rag
    from tiers.guardrag import gatekeeper_v4
    from tiers.llm_utils import ask_llm, strip_refinement_prefix, format_retrieved_context
    from evaluation.metrics import compute_f1

    # ── load & split ─────────────────────────────────────────────────────
    print(f"\nLoading {MAX_SAMPLES} samples from {DATA_PATH} ...")
    all_samples = load_tatqa(DATA_PATH, max_samples=MAX_SAMPLES)
    val_samples = all_samples[TRAIN_SIZE: TRAIN_SIZE + VAL_SIZE]
    print(f"  train={TRAIN_SIZE}  val={len(val_samples)}  (using val only)")

    # ── build improved index ──────────────────────────────────────────────
    print("\nBuilding table-aware vector store (improved) ...")
    index, chunks, metadata = build_vector_store(
        all_samples[:TRAIN_SIZE + VAL_SIZE],   # index over train+val docs
        embed_model,
        use_table_aware_chunking=True,
    )

    # ── debate prompt (mirrors tier3_selective_debate exactly) ────────────
    def _force_debate(question, retrieved, draft_answer, signals):
        from tiers.llm_utils import format_retrieved_context
        context = format_retrieved_context(retrieved, max_chars=4500)
        prompt = f"""
You are a careful verifier answering ONLY from the evidence below.

Question:
{question}

Evidence:
{context}

Draft answer:
{draft_answer}

Concerns raised:
{chr(10).join("- " + s for s in signals)}

Instructions:
- Use only the evidence.
- Prefer precision over verbosity.
- If the question is asking for components, list the supported components explicitly.
- If the evidence is partial, give the supported partial answer.
- Do NOT mention the concerns or reasoning.
- Return ONLY the final answer.

Final answer:
""".strip()
        out = ask_llm(prompt, client=client, model=JUDGE_MODEL,
                      temperature=0.0, max_tokens=260)
        return strip_refinement_prefix(out["text"]), out["tokens"]

    # ── collect loop ──────────────────────────────────────────────────────
    records = []
    print(f"\nCollecting {len(val_samples)} validation samples ...\n")

    for i, sample in enumerate(val_samples):
        question   = sample["question"]
        gold       = sample["gold_answer"]

        retrieved  = retrieve(question, index, chunks, metadata, embed_model, top_k=TOP_K)
        t1         = baseline_rag(question, retrieved, client, BASE_MODEL)
        tier1_ans  = t1["answer"]
        tier1_tok  = t1["tokens"]

        # gatekeeper at threshold=0.0 → always computes score, trigger always True
        # but we use it only for the score, not the trigger decision
        gate = gatekeeper_v4(
            question, retrieved, tier1_ans,
            nli_model=nli_model,
            threshold=0.0,
        )

        # force-run debate regardless of gatekeeper decision
        # pass all fired signal names (or a placeholder if none fired)
        signals_for_prompt = gate["signals"] if gate["signals"] else ["Score-based review requested"]
        debated_ans, debate_tok = _force_debate(question, retrieved, tier1_ans, signals_for_prompt)

        tier1_f1   = compute_f1(tier1_ans,   gold)
        debated_f1 = compute_f1(debated_ans, gold)

        rec = {
            "question":         question,
            "gold_answer":      gold,
            "tier1_answer":     tier1_ans,
            "tier1_f1":         tier1_f1,
            "debated_answer":   debated_ans,
            "debated_f1":       debated_f1,
            "gatekeeper_score": gate["gatekeeper_score"],
            "tier1_tokens":     tier1_tok,
            "debate_tokens":    debate_tok,
        }

        # boolean columns for each signal
        for sig in ALL_SIGNALS:
            rec[f"sig_{sig}"] = sig in gate["signals"]

        records.append(rec)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(val_samples)}] done")

    df = pd.DataFrame(records)
    df.to_csv(SWEEP_DATA_PATH, index=False)
    print(f"\nSaved {len(df)} rows to {SWEEP_DATA_PATH}")

    # ── summary ───────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("COLLECT SUMMARY")
    print("─" * 50)
    print(f"  Samples collected : {len(df)}")
    print(f"  Mean Tier 1 F1    : {df['tier1_f1'].mean():.4f}")
    print(f"  Mean Debated F1   : {df['debated_f1'].mean():.4f}")
    print(f"  Mean GK score     : {df['gatekeeper_score'].mean():.4f}")
    print()
    print("  Signal fire counts:")
    for sig in ALL_SIGNALS:
        col = f"sig_{sig}"
        print(f"    {sig:<35s}: {df[col].sum()} / {len(df)}")


# ---------------------------------------------------------------------------
# Mode 2: sweep
# ---------------------------------------------------------------------------

def run_sweep():
    print("=" * 60)
    print("MODE: sweep  (offline, no API calls)")
    print("=" * 60)

    if not os.path.exists(SWEEP_DATA_PATH):
        sys.exit(f"ERROR: {SWEEP_DATA_PATH} not found. Run with --mode collect first.")

    df = pd.read_csv(SWEEP_DATA_PATH)
    print(f"Loaded {len(df)} rows from {SWEEP_DATA_PATH}\n")

    thresholds = _thresholds()

    # ── main sweep ────────────────────────────────────────────────────────
    sweep_rows = [_sweep_row(df, t) for t in thresholds]
    sweep_df   = pd.DataFrame(sweep_rows)

    print(f"{'threshold':>10} | {'mean_f1':>8} | {'trigger_rate':>13} | {'mean_tokens':>12}")
    print("-" * 52)
    for _, row in sweep_df.iterrows():
        print(f"  {row['threshold']:>8.2f} | {row['mean_f1']:>8.4f} | "
              f"{row['trigger_rate']:>12.1%} | {row['mean_tokens']:>12.1f}")

    # ── elbow detection ───────────────────────────────────────────────────
    rec_threshold = _find_elbow(sweep_df)
    print(f"\nRecommended threshold: {rec_threshold:.2f}")

    sweep_df.to_csv(SWEEP_RESULTS_PATH, index=False)
    print(f"Sweep results saved to {SWEEP_RESULTS_PATH}")

    # ── signal ablation ───────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("SIGNAL ABLATION  (at recommended threshold)")
    print("─" * 60)

    from tiers.guardrag import SIGNAL_WEIGHTS

    # baseline metrics at recommended threshold
    baseline_row = _sweep_row(df, rec_threshold)
    baseline_f1  = baseline_row["mean_f1"]
    baseline_rate = baseline_row["trigger_rate"]

    print(f"\n  Baseline (all signals) @ {rec_threshold:.2f}:")
    print(f"    mean_f1={baseline_f1:.4f}  trigger_rate={baseline_rate:.1%}")
    print()
    print(f"  {'signal_removed':<35} | {'new_f1':>8} | {'new_trigger_rate':>16} | {'f1_delta':>9}")
    print("  " + "-" * 77)

    ablation_rows = []
    for sig in ALL_SIGNALS:
        sig_col = f"sig_{sig}"
        sig_weight = SIGNAL_WEIGHTS.get(sig, 0.0)

        # recompute gatekeeper_score with this signal zeroed out
        df_abl = df.copy()
        df_abl["abl_score"] = df_abl["gatekeeper_score"] - (
            df_abl[sig_col].astype(float) * sig_weight
        )
        df_abl["abl_score"] = df_abl["abl_score"].clip(lower=0.0).round(4)

        abl_row   = _sweep_row(df_abl, rec_threshold, score_col="abl_score")
        f1_delta  = abl_row["mean_f1"] - baseline_f1

        print(f"  {sig:<35} | {abl_row['mean_f1']:>8.4f} | "
              f"{abl_row['trigger_rate']:>15.1%} | {f1_delta:>+9.4f}")

        ablation_rows.append({
            "signal_removed":  sig,
            "new_f1":          abl_row["mean_f1"],
            "new_trigger_rate": abl_row["trigger_rate"],
            "f1_delta":        round(f1_delta, 4),
        })

    print()

    # ── plot ──────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.plot(sweep_df["threshold"], sweep_df["mean_f1"],
                 color="blue", linewidth=2, label="Mean F1")
        ax1.set_xlabel("Threshold")
        ax1.set_ylabel("Mean F1", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.set_ylim(bottom=0)

        ax2 = ax1.twinx()
        ax2.plot(sweep_df["threshold"], sweep_df["trigger_rate"] * 100,
                 color="red", linewidth=2, linestyle="--", label="Trigger rate %")
        ax2.set_ylabel("Debate trigger rate (%)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.set_ylim(0, 100)

        # elbow marker
        rec_f1 = float(sweep_df.loc[
            sweep_df["threshold"] == rec_threshold, "mean_f1"
        ].values[0]) if rec_threshold in sweep_df["threshold"].values else baseline_f1

        ax1.axvline(rec_threshold, color="gray", linestyle="--", linewidth=1.5)
        ax1.annotate(
            f"Recommended\n{rec_threshold:.2f}",
            xy=(rec_threshold, rec_f1),
            xytext=(rec_threshold + 0.05, rec_f1 + 0.02),
            fontsize=9,
            color="gray",
            arrowprops=dict(arrowstyle="->", color="gray"),
        )

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        plt.title("GUARD-RAG Gatekeeper Threshold Sweep\n(Mean F1 vs Debate Trigger Rate)")
        plt.tight_layout()
        plt.savefig(SWEEP_PLOT_PATH, dpi=150)
        plt.close()
        print(f"Plot saved to {SWEEP_PLOT_PATH}")

    except ImportError:
        print("matplotlib not installed — skipping plot.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GUARD-RAG threshold sweep")
    parser.add_argument(
        "--mode",
        choices=["collect", "sweep"],
        default=None,
        help="collect: run API and save data. sweep: offline analysis. "
             "Auto-selects sweep if sweep_data.csv exists.",
    )
    args = parser.parse_args()

    if args.mode is None:
        args.mode = "sweep" if os.path.exists(SWEEP_DATA_PATH) else "collect"
        print(f"Auto-selected mode: {args.mode}")

    if args.mode == "collect":
        run_collect()
    else:
        run_sweep()


if __name__ == "__main__":
    main()

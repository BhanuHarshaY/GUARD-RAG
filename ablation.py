"""
GUARD-RAG Ablation Study
========================
Phase 1 (offline): Threshold sweep + debate-impact analysis from existing CSV.
Phase 2 (online):  Signal ablation — remove one signal at a time, re-run 30 samples.
Phase 3 (online):  Threshold sweep for T < 0.35 (force-debate currently-bypassed samples).

Usage:
    python ablation.py --phase 1          # offline only, no API calls
    python ablation.py --phase 2          # signal ablation  (~18 min)
    python ablation.py --phase 3          # low-threshold sweep (~12 min)
    python ablation.py --phase all        # everything
"""

import argparse
import os
import random
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_CSV   = "evaluation_results.csv"
ABLATION_DIR  = "ablation_results"
ABLATION_SAMPLES = 15   # keep fast; use same seed as run.py
RANDOM_SEED      = 42

ALL_SIGNALS = [
    "heuristic_arithmetic",
    "heuristic_multispan",
    "heuristic_insufficient_info",
    "heuristic_missing_rows",
    "heuristic_too_long",
    "nli_low_grounding",
]

os.makedirs(ABLATION_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bar_color(val, baseline):
    if val > baseline + 0.005:
        return "#2ecc71"   # green  — improvement
    if val < baseline - 0.005:
        return "#e74c3c"   # red    — hurt
    return "#95a5a6"       # grey   — neutral


def _save(fig, name):
    path = os.path.join(ABLATION_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — offline analysis
# ─────────────────────────────────────────────────────────────────────────────

def phase1_offline():
    print("\n" + "=" * 60)
    print("PHASE 1 — Offline Analysis")
    print("=" * 60)

    if not os.path.exists(RESULTS_CSV):
        print(f"  {RESULTS_CSV} not found — run run.py first.")
        return

    df = pd.read_csv(RESULTS_CSV)
    print(f"  Loaded {len(df)} samples from {RESULTS_CSV}")

    # ── 1a. Tier comparison summary ───────────────────────────────────────────
    print("\n--- Tier Comparison ---")
    for tier, col in [("Baseline", "baseline"), ("Refinement", "refinement"), ("GUARD-RAG", "guardrag")]:
        f1 = df[f"{col}_f1"].mean()
        em = df[f"{col}_em"].mean()
        tok = df[f"{col}_tokens"].mean()
        print(f"  {tier:12s}  F1={f1:.4f}  EM={em:.4f}  Tokens={tok:.0f}")

    # ── 1b. Threshold sweep (T = 0.35 → 1.0) ─────────────────────────────────
    print("\n--- Offline Threshold Sweep (T ≥ 0.35) ---")
    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 1.00]
    sweep_rows = []
    for t in thresholds:
        # At threshold T: samples whose score < T revert to baseline_answer
        # (GUARD-RAG debates Baseline, so bypass → return Baseline)
        f1_vals = []
        em_vals = []
        debate_rate = 0
        for _, row in df.iterrows():
            if row["guardrag_gatekeeper_score"] >= t:
                f1_vals.append(row["guardrag_f1"])
                em_vals.append(row["guardrag_em"])
                debate_rate += 1
            else:
                f1_vals.append(row["baseline_f1"])
                em_vals.append(row["baseline_em"])
        f1 = np.mean(f1_vals)
        em = np.mean(em_vals)
        dr = debate_rate / len(df)
        sweep_rows.append({"threshold": t, "f1": f1, "em": em, "debate_rate": dr})
        print(f"  T={t:.2f}  F1={f1:.4f}  EM={em:.4f}  Debate%={dr:.0%}")

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(os.path.join(ABLATION_DIR, "threshold_sweep_offline.csv"), index=False)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ts = sweep_df["threshold"]
    ax1.plot(ts, sweep_df["f1"], "o-", color="#1abc9c", linewidth=2, label="F1")
    ax1.axhline(df["baseline_f1"].mean(), color="#3498db", linestyle="--", alpha=0.7, label="Baseline F1")
    ax1.axhline(df["refinement_f1"].mean(), color="#e67e22", linestyle="--", alpha=0.7, label="Refinement F1")
    ax2.bar(ts, sweep_df["debate_rate"], width=0.03, alpha=0.25, color="#9b59b6", label="Debate %")
    ax1.set_xlabel("Gatekeeper Threshold")
    ax1.set_ylabel("F1 Score")
    ax2.set_ylabel("Debate Rate")
    ax1.set_ylim(0.45, 0.70)
    ax2.set_ylim(0, 1.0)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)
    ax1.set_title("F1 vs. Gatekeeper Threshold (Offline Simulation)")
    ax1.grid(alpha=0.3)
    _save(fig, "threshold_sweep_offline.png")

    # ── 1c. Debate impact: debated vs non-debated ─────────────────────────────
    print("\n--- Debate Impact (debated vs. non-debated samples) ---")
    debated   = df[df["guardrag_debate_triggered"] == True]
    bypassed  = df[df["guardrag_debate_triggered"] == False]

    print(f"  Debated   ({len(debated):2d} samples):  "
          f"Refinement F1={debated['refinement_f1'].mean():.4f}  "
          f"GUARD-RAG F1={debated['guardrag_f1'].mean():.4f}  "
          f"Delta={debated['guardrag_f1'].mean() - debated['refinement_f1'].mean():+.4f}")
    print(f"  Bypassed  ({len(bypassed):2d} samples):  "
          f"Refinement F1={bypassed['refinement_f1'].mean():.4f}  "
          f"GUARD-RAG F1={bypassed['guardrag_f1'].mean():.4f}  "
          f"Delta={bypassed['guardrag_f1'].mean() - bypassed['refinement_f1'].mean():+.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, subset, label in [
        (axes[0], debated,  f"Debated (n={len(debated)})"),
        (axes[1], bypassed, f"Bypassed (n={len(bypassed)})"),
    ]:
        ax.scatter(subset["refinement_f1"], subset["guardrag_f1"],
                   alpha=0.6, color="#1abc9c", edgecolors="white", s=50)
        lim = [0, 1.05]
        ax.plot(lim, lim, "k--", alpha=0.3)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Refinement F1"); ax.set_ylabel("GUARD-RAG F1")
        ax.set_title(label)
        ax.grid(alpha=0.3)
        wins  = (subset["guardrag_f1"] > subset["refinement_f1"]).sum()
        losses = (subset["guardrag_f1"] < subset["refinement_f1"]).sum()
        ties  = (subset["guardrag_f1"] == subset["refinement_f1"]).sum()
        ax.text(0.02, 0.95, f"Win/Tie/Loss: {wins}/{ties}/{losses}",
                transform=ax.transAxes, fontsize=8, va="top")
    fig.suptitle("GUARD-RAG vs. Refinement — Debate Impact")
    fig.tight_layout()
    _save(fig, "debate_impact.png")

    # ── 1d. Per-verdict F1 breakdown ─────────────────────────────────────────
    print("\n--- Per-Verdict F1 ---")
    for verdict in ["APPROVE", "REVISE", "ABSTAIN", None]:
        sub = df[df["guardrag_adjudicator_verdict"] == verdict] if verdict else \
              df[df["guardrag_adjudicator_verdict"].isna()]
        if len(sub) == 0:
            continue
        label = verdict if verdict else "No debate"
        print(f"  {label:10s} ({len(sub):2d} samples):  "
              f"GUARD-RAG F1={sub['guardrag_f1'].mean():.4f}  "
              f"Refinement F1={sub['refinement_f1'].mean():.4f}  "
              f"Delta={sub['guardrag_f1'].mean() - sub['refinement_f1'].mean():+.4f}")

    # ── 1e. Answer-type breakdown ─────────────────────────────────────────────
    def classify_question(q):
        q = q.lower()
        arith_kws = ["difference", "change", "increase", "decrease", "% change",
                     "ratio", "sum of", "average", "calculate", "grew", "fell", "more", "less"]
        multi_kws  = ["respectively", "both", "each year", "each of"]
        if any(k in q for k in arith_kws):
            return "arithmetic"
        if any(k in q for k in multi_kws):
            return "multispan"
        return "span"

    df["q_type"] = df["question"].apply(classify_question)
    print("\n--- Answer-Type Breakdown ---")
    type_rows = []
    for qtype in ["span", "arithmetic", "multispan"]:
        sub = df[df["q_type"] == qtype]
        if len(sub) == 0:
            continue
        row = {
            "type": qtype, "n": len(sub),
            "baseline_f1":   sub["baseline_f1"].mean(),
            "refinement_f1": sub["refinement_f1"].mean(),
            "guardrag_f1":   sub["guardrag_f1"].mean(),
        }
        type_rows.append(row)
        print(f"  {qtype:10s} (n={len(sub):2d})  "
              f"B={row['baseline_f1']:.3f}  R={row['refinement_f1']:.3f}  G={row['guardrag_f1']:.3f}")
    type_df = pd.DataFrame(type_rows)
    type_df.to_csv(os.path.join(ABLATION_DIR, "answer_type_breakdown.csv"), index=False)

    x = np.arange(len(type_df))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w, type_df["baseline_f1"],   w, label="Baseline",   color="#3498db", alpha=0.85)
    ax.bar(x,     type_df["refinement_f1"], w, label="Refinement", color="#e67e22", alpha=0.85)
    ax.bar(x + w, type_df["guardrag_f1"],   w, label="GUARD-RAG",  color="#1abc9c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['type']}\n(n={r['n']})" for _, r in type_df.iterrows()])
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.set_title("F1 by Answer Type")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "answer_type_f1.png")

    print("\nPhase 1 complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — online signal ablation
# ─────────────────────────────────────────────────────────────────────────────

def phase2_signal_ablation():
    print("\n" + "=" * 60)
    print("PHASE 2 — Signal Ablation (online, ~18 min)")
    print("=" * 60)

    from config import client, embed_model, nli_model, BASE_MODEL, JUDGE_MODEL
    from data.loader import load_tatqa
    from indexing.vector_store import build_vector_store
    from evaluation.evaluator import evaluate_all, summarize_results

    samples = load_tatqa("data/tatqa_dataset_train.json", max_samples=300)
    random.seed(RANDOM_SEED)
    random.shuffle(samples)
    eval_samples = samples[:ABLATION_SAMPLES]

    print(f"Building index for {ABLATION_SAMPLES} samples...")
    idx, chunks, meta = build_vector_store(samples, embed_model, use_table_aware_chunking=True)

    configs = [("all_signals", None)] + [(f"no_{s}", {s}) for s in ALL_SIGNALS]

    rows = []
    for config_name, disabled in configs:
        print(f"\n  Running: {config_name} ...")
        df = evaluate_all(
            eval_samples,
            idx, chunks, meta,
            idx, chunks, meta,
            embed_model, client, BASE_MODEL, JUDGE_MODEL,
            mode="improved", max_samples=ABLATION_SAMPLES,
            top_k=6, nli_model=nli_model,
            disabled_signals=disabled,
        )
        f1  = df["guardrag_f1"].mean()
        em  = df["guardrag_em"].mean()
        dr  = df["guardrag_debate_triggered"].mean()
        tok = df["guardrag_tokens"].mean()
        rows.append({"config": config_name, "guardrag_f1": f1, "guardrag_em": em,
                     "debate_rate": dr, "avg_tokens": tok})
        print(f"    F1={f1:.4f}  EM={em:.4f}  Debate={dr:.0%}  Tokens={tok:.0f}")

    abl_df = pd.DataFrame(rows)
    abl_df.to_csv(os.path.join(ABLATION_DIR, "signal_ablation.csv"), index=False)

    # Plot
    baseline_f1 = abl_df.loc[abl_df["config"] == "all_signals", "guardrag_f1"].values[0]
    signal_rows = abl_df[abl_df["config"] != "all_signals"].copy()
    signal_rows["delta"] = signal_rows["guardrag_f1"] - baseline_f1
    signal_rows = signal_rows.sort_values("delta")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [_bar_color(r["guardrag_f1"], baseline_f1) for _, r in signal_rows.iterrows()]
    bars = ax.barh(signal_rows["config"].str.replace("no_", ""), signal_rows["delta"],
                   color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("F1 Delta (vs. all signals)")
    ax.set_title(f"Signal Ablation — F1 Impact\n(Baseline with all signals: {baseline_f1:.4f})")
    for bar, (_, row) in zip(bars, signal_rows.iterrows()):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{row['delta']:+.4f}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    patches = [mpatches.Patch(color="#e74c3c", label="Signal hurts when removed (important)"),
               mpatches.Patch(color="#2ecc71", label="Signal helps when removed (harmful)"),
               mpatches.Patch(color="#95a5a6", label="Neutral")]
    ax.legend(handles=patches, fontsize=7, loc="lower right")
    _save(fig, "signal_ablation.png")

    print("\nPhase 2 complete.")
    return abl_df


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — low-threshold sweep (force debate on bypassed samples)
# ─────────────────────────────────────────────────────────────────────────────

def phase3_threshold_sweep():
    print("\n" + "=" * 60)
    print("PHASE 3 — Low-Threshold Sweep (online, ~12 min)")
    print("=" * 60)

    from config import client, embed_model, nli_model, BASE_MODEL, JUDGE_MODEL
    from data.loader import load_tatqa
    from indexing.vector_store import build_vector_store
    from evaluation.evaluator import evaluate_all

    samples = load_tatqa("data/tatqa_dataset_train.json", max_samples=300)
    random.seed(RANDOM_SEED)
    random.shuffle(samples)
    eval_samples = samples[:ABLATION_SAMPLES]

    print(f"Building index...")
    idx, chunks, meta = build_vector_store(samples, embed_model, use_table_aware_chunking=True)

    thresholds_online = [0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
    sweep_rows = []
    for t in thresholds_online:
        print(f"\n  Running threshold={t:.2f} ...")
        df = evaluate_all(
            eval_samples,
            idx, chunks, meta,
            idx, chunks, meta,
            embed_model, client, BASE_MODEL, JUDGE_MODEL,
            mode="improved", max_samples=ABLATION_SAMPLES,
            top_k=6, nli_model=nli_model,
            guardrag_threshold=t,
        )
        f1  = df["guardrag_f1"].mean()
        em  = df["guardrag_em"].mean()
        dr  = df["guardrag_debate_triggered"].mean()
        tok = df["guardrag_tokens"].mean()
        sweep_rows.append({"threshold": t, "f1": f1, "em": em, "debate_rate": dr, "avg_tokens": tok})
        print(f"    F1={f1:.4f}  EM={em:.4f}  Debate={dr:.0%}  Tokens={tok:.0f}")

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(os.path.join(ABLATION_DIR, "threshold_sweep_online.csv"), index=False)

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()
    ax1.plot(sweep_df["threshold"], sweep_df["f1"], "o-", color="#1abc9c", linewidth=2, label="F1")
    ax1.plot(sweep_df["threshold"], sweep_df["em"], "s--", color="#3498db", linewidth=1.5, label="EM", alpha=0.7)
    ax2.fill_between(sweep_df["threshold"], sweep_df["debate_rate"], alpha=0.15, color="#9b59b6")
    ax2.plot(sweep_df["threshold"], sweep_df["debate_rate"], "^:", color="#9b59b6", linewidth=1.5, label="Debate %")
    ax1.set_xlabel("Gatekeeper Threshold")
    ax1.set_ylabel("Score")
    ax2.set_ylabel("Debate Rate")
    ax1.set_ylim(0.40, 0.75)
    ax2.set_ylim(0, 1.2)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax1.set_title("GUARD-RAG F1 + Debate Rate vs. Threshold (Online Sweep)")
    ax1.grid(alpha=0.3)

    # Annotate optimal
    best = sweep_df.loc[sweep_df["f1"].idxmax()]
    ax1.annotate(f"Best T={best['threshold']:.2f}\nF1={best['f1']:.4f}",
                 xy=(best["threshold"], best["f1"]),
                 xytext=(best["threshold"] + 0.05, best["f1"] - 0.03),
                 arrowprops=dict(arrowstyle="->", color="gray"),
                 fontsize=8)
    _save(fig, "threshold_sweep_online.png")

    print("\nPhase 3 complete.")
    return sweep_df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="1", choices=["1", "2", "3", "all"])
    args = parser.parse_args()

    if args.phase in ("1", "all"):
        phase1_offline()

    if args.phase in ("2", "all"):
        phase2_signal_ablation()

    if args.phase in ("3", "all"):
        phase3_threshold_sweep()

    print(f"\nAll results saved to ./{ABLATION_DIR}/")

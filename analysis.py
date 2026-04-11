"""
GUARD-RAG — Ablation Analysis
Reads evaluation_results.csv, classifies questions, runs offline ablations,
prints full tables, and saves plots.

Usage:
    python analysis.py
"""

import re
import numpy as np
import pandas as pd

# ── Optional matplotlib ────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not found — plots will be skipped.\n")


# ── Question-type classifier ───────────────────────────────────────────────────

_ARITH_PAT = re.compile(
    r"\b(difference|percentage change|percent change|% change|how much more|"
    r"how much less|change in|ratio of|sum of|increase from|decrease from|"
    r"grew by|fell by|average of|average \w+|what is the increase|"
    r"what is the decrease|what is the change|what was the change)\b",
    re.IGNORECASE,
)
_MULTISPAN_PAT = re.compile(
    r"\b(respectively|in \d{4} and \d{4})\b|\b(19\d{2}|20\d{2})\b.*\b(19\d{2}|20\d{2})\b",
    re.IGNORECASE,
)
_REASON_PAT = re.compile(r"\b(why|what led|what caused|how did|explain)\b", re.IGNORECASE)


def classify_question(q):
    q = str(q)
    if _REASON_PAT.search(q):
        return "reasoning"
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", q)
    if len(set(years)) >= 2 or "respectively" in q.lower():
        return "multispan"
    if _ARITH_PAT.search(q):
        return "arithmetic"
    return "span_lookup"


# ── Load data ──────────────────────────────────────────────────────────────────

df = pd.read_csv("evaluation_results.csv")
df["q_type"] = df["question"].apply(classify_question)

SEP = "=" * 72


# ── 1. Overall summary ─────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("1. OVERALL SUMMARY  (N={})".format(len(df)))
print(SEP)

overall = pd.DataFrame([{
    "Tier":        t,
    "F1":          round(df[f"{t}_f1"].mean(), 4),
    "EM":          round(df[f"{t}_em"].mean(), 4),
    "Halluc.":     round(df[f"{t}_hallucination_rate"].mean(), 4),
    "Avg Tokens":  round(df[f"{t}_tokens"].mean(), 1),
    "Avg Latency": round(df[f"{t}_latency"].mean(), 3),
} for t in ["baseline", "refinement", "guardrag"]])

print(overall.to_string(index=False))


# ── 2. Per question-type breakdown ─────────────────────────────────────────────

print(f"\n{SEP}")
print("2. PER QUESTION-TYPE BREAKDOWN")
print(SEP)

rows = []
for qtype in ["arithmetic", "multispan", "span_lookup", "reasoning"]:
    sub = df[df["q_type"] == qtype]
    if sub.empty:
        continue
    for tier in ["baseline", "refinement", "guardrag"]:
        rows.append({
            "q_type":  qtype,
            "n":       len(sub),
            "Tier":    tier,
            "F1":      round(sub[f"{tier}_f1"].mean(), 4),
            "EM":      round(sub[f"{tier}_em"].mean(), 4),
        })

qtype_df = pd.DataFrame(rows)
pivot_f1 = qtype_df.pivot_table(index=["q_type", "n"], columns="Tier", values="F1").reset_index()
pivot_em = qtype_df.pivot_table(index=["q_type", "n"], columns="Tier", values="EM").reset_index()

print("\nF1 by question type:")
print(pivot_f1.to_string(index=False))
print("\nEM by question type:")
print(pivot_em.to_string(index=False))


# ── 3. Per-sample verdict analysis ─────────────────────────────────────────────

print(f"\n{SEP}")
print("3. PER-SAMPLE VERDICT & TIER COMPARISON")
print(SEP)

sample_rows = []
for _, r in df.iterrows():
    b, ref, g = r["baseline_f1"], r["refinement_f1"], r["guardrag_f1"]
    winner = max([("baseline", b), ("refinement", ref), ("guardrag", g)], key=lambda x: x[1])[0]
    sample_rows.append({
        "Q": r["question"][:55],
        "type":      r["q_type"],
        "gold":      str(r["gold_answer"])[:25],
        "B_F1":      round(b, 2),
        "R_F1":      round(ref, 2),
        "G_F1":      round(g, 2),
        "debate":    "Y" if r["guardrag_debate_triggered"] else "N",
        "verdict":   str(r["guardrag_adjudicator_verdict"])[:6],
        "gk_score":  round(r["guardrag_gatekeeper_score"], 2),
        "winner":    winner,
    })

sample_df = pd.DataFrame(sample_rows)
print(sample_df.to_string(index=False))

# Win/loss counts
wins = sample_df["winner"].value_counts()
print(f"\nQuestion-level wins:")
for t in ["baseline", "refinement", "guardrag"]:
    print(f"  {t:<12}: {wins.get(t, 0)} / {len(df)}")


# ── 4. Offline gatekeeper threshold sweep ─────────────────────────────────────

print(f"\n{SEP}")
print("4. OFFLINE GATEKEEPER THRESHOLD SWEEP")
print(SEP)

# For each threshold: if gk_score >= threshold → use guardrag_f1, else use refinement_f1
thresholds = np.round(np.arange(0.0, 1.05, 0.05), 2)
sweep_rows = []
for t in thresholds:
    triggered   = df["guardrag_gatekeeper_score"] >= t
    eff_f1      = np.where(triggered, df["guardrag_f1"], df["refinement_f1"])
    eff_em      = np.where(triggered, df["guardrag_em"], df["refinement_em"])
    debate_rate = triggered.mean()
    sweep_rows.append({
        "threshold":    t,
        "eff_F1":       round(float(eff_f1.mean()), 4),
        "eff_EM":       round(float(eff_em.mean()), 4),
        "debate_rate":  round(float(debate_rate), 3),
    })

sweep_df = pd.DataFrame(sweep_rows)
print(sweep_df.to_string(index=False))
best = sweep_df.loc[sweep_df["eff_F1"].idxmax()]
print(f"\nBest threshold by F1: {best['threshold']} → F1={best['eff_F1']}, EM={best['eff_EM']}, debate_rate={best['debate_rate']}")


# ── 5. Signal ablation (offline) ──────────────────────────────────────────────

print(f"\n{SEP}")
print("5. SIGNAL ABLATION  (what-if each gatekeeper signal were removed)")
print(SEP)

SIGNAL_WEIGHTS = {
    "heuristic_arithmetic":        0.6,
    "heuristic_multispan":         0.5,
    "nli_low_grounding":           0.3,
    "heuristic_insufficient_info": 0.4,
    "heuristic_missing_rows":      0.3,
    "heuristic_too_long":          0.2,
}
THRESHOLD = 0.35

# Reconstruct per-signal contributions from gatekeeper scores
# We know which signals fired by looking at score composition:
# For arithmetic Qs, heuristic_arithmetic=0.6 was definitely in the score
# For multispan Qs, heuristic_multispan=0.5 was in the score
# We can't fully reconstruct all signals without re-running, but we can simulate
# removing the strongest signals based on question type

baseline_f1 = df["baseline_f1"].mean()
refinement_f1 = df["refinement_f1"].mean()
guardrag_f1 = df["guardrag_f1"].mean()

abl_rows = []
for sig, wt in SIGNAL_WEIGHTS.items():
    # Simulate score without this signal:
    # For arithmetic signal: affects arithmetic questions
    # For multispan: affects multispan questions
    # For others: rough estimate — subtract weight from scores where it could fire
    if sig == "heuristic_arithmetic":
        mask = df["q_type"] == "arithmetic"
    elif sig == "heuristic_multispan":
        mask = df["q_type"] == "multispan"
    else:
        # Conservative: could fire on any debated sample
        mask = df["guardrag_debate_triggered"]

    adj_scores = df["guardrag_gatekeeper_score"].copy()
    adj_scores[mask] = (adj_scores[mask] - wt).clip(lower=0)

    triggered  = adj_scores >= THRESHOLD
    eff_f1     = np.where(triggered, df["guardrag_f1"], df["refinement_f1"])
    abl_rows.append({
        "signal_removed":  sig,
        "weight":          wt,
        "new_debate_rate": round(float(triggered.mean()), 3),
        "new_eff_F1":      round(float(eff_f1.mean()), 4),
        "f1_delta":        round(float(eff_f1.mean()) - guardrag_f1, 4),
    })

abl_df = pd.DataFrame(abl_rows).sort_values("f1_delta", ascending=False)
print(f"Baseline at threshold={THRESHOLD}: F1={guardrag_f1:.4f}, debate_rate={df['guardrag_debate_triggered'].mean():.3f}")
print()
print(abl_df.to_string(index=False))


# ── 6. Tier-vs-Tier head-to-head ──────────────────────────────────────────────

print(f"\n{SEP}")
print("6. HEAD-TO-HEAD: WHERE DOES EACH TIER WIN / LOSE?")
print(SEP)

def h2h(tier_a, tier_b, label):
    a_wins  = (df[f"{tier_a}_f1"] > df[f"{tier_b}_f1"]).sum()
    b_wins  = (df[f"{tier_b}_f1"] > df[f"{tier_a}_f1"]).sum()
    ties    = (df[f"{tier_a}_f1"] == df[f"{tier_b}_f1"]).sum()
    print(f"  {label}: {tier_a} wins={a_wins}  {tier_b} wins={b_wins}  ties={ties}")

h2h("refinement", "baseline",   "Refinement vs Baseline")
h2h("guardrag",   "refinement", "GUARD-RAG  vs Refinement")
h2h("guardrag",   "baseline",   "GUARD-RAG  vs Baseline")


# ── 7. Design drift & CoT analysis ────────────────────────────────────────────

print(f"\n{SEP}")
print("7. DESIGN DRIFT & CoT IMPACT ANALYSIS")
print(SEP)

arith   = df[df["q_type"] == "arithmetic"]
nonarith = df[df["q_type"] != "arithmetic"]

print(f"Arithmetic questions ({len(arith)}):")
print(f"  Baseline F1={arith['baseline_f1'].mean():.3f}  EM={arith['baseline_em'].mean():.3f}")
print(f"  Refinement F1={arith['refinement_f1'].mean():.3f}  EM={arith['refinement_em'].mean():.3f}")
print(f"  GUARD-RAG F1={arith['guardrag_f1'].mean():.3f}  EM={arith['guardrag_em'].mean():.3f}")

print(f"\nNon-arithmetic questions ({len(nonarith)}):")
print(f"  Baseline F1={nonarith['baseline_f1'].mean():.3f}  EM={nonarith['baseline_em'].mean():.3f}")
print(f"  Refinement F1={nonarith['refinement_f1'].mean():.3f}  EM={nonarith['refinement_em'].mean():.3f}")
print(f"  GUARD-RAG F1={nonarith['guardrag_f1'].mean():.3f}  EM={nonarith['guardrag_em'].mean():.3f}")

print(f"""
Design drift summary:
  Original Tier 2: SC (k=3 candidates, temp=0.3) + refine — diversity through sampling
  Current  Tier 2: 1 refine call (temp=0.0)               — verification only
  Problem:  On 8B model, temp=0.3 SC gave near-identical outputs → removed.
            Single refine is verification, not self-consistency.

  CoT (chain-of-thought) currently:
    Baseline:   CoT ON  for arithmetic  (Step 1/2/3 scratchpad, final answer extracted)
    Refinement: CoT ON  for arithmetic  (verify+recompute)
    GUARD-RAG:  CoT ON  in adjudicator  for arithmetic  (recompute from evidence)

  CoT impact estimate from results:
    Arithmetic EM={arith['baseline_em'].mean():.2f} (with CoT) vs Non-arithmetic EM={nonarith['baseline_em'].mean():.2f}
    CoT is HELPING on arithmetic but 8B model still fails complex multi-step calcs.

  CoT for ALL question types (span/reasoning):
    Risk: produces longer answers → precision drops → F1 drops → EM drops
    Span questions already have EM={nonarith['baseline_em'].mean():.2f} — CoT would likely hurt these.

  Recommendation: Keep CoT for arithmetic only. The bottleneck is model capability, not prompting.
""")


# ── 8. Plots ──────────────────────────────────────────────────────────────────

if HAS_MPL:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("GUARD-RAG Ablation Analysis", fontsize=14, fontweight="bold")

    TIERS  = ["baseline", "refinement", "guardrag"]
    COLORS = ["#4C72B0", "#DD8452", "#55A868"]

    # Plot 1: Overall F1 / EM
    ax = axes[0, 0]
    x = np.arange(3)
    f1_vals = [df[f"{t}_f1"].mean() for t in TIERS]
    em_vals = [df[f"{t}_em"].mean() for t in TIERS]
    ax.bar(x - 0.2, f1_vals, 0.35, label="F1",  color=COLORS, alpha=0.85)
    ax.bar(x + 0.2, em_vals, 0.35, label="EM",  color=COLORS, alpha=0.45, hatch="//")
    ax.set_xticks(x); ax.set_xticklabels(["Baseline", "Refinement", "GUARD-RAG"], fontsize=9)
    ax.set_ylim(0, 1); ax.set_title("Overall F1 & EM"); ax.legend(fontsize=8)
    for i, (f, e) in enumerate(zip(f1_vals, em_vals)):
        ax.text(i - 0.2, f + 0.01, f"{f:.3f}", ha="center", fontsize=7)
        ax.text(i + 0.2, e + 0.01, f"{e:.3f}", ha="center", fontsize=7)

    # Plot 2: F1 by question type
    ax = axes[0, 1]
    qtypes = ["arithmetic", "multispan", "span_lookup", "reasoning"]
    n_types = len(qtypes)
    bw = 0.25
    for j, tier in enumerate(TIERS):
        vals = [df[df["q_type"] == qt][f"{tier}_f1"].mean() if len(df[df["q_type"] == qt]) > 0 else 0
                for qt in qtypes]
        ax.bar(np.arange(n_types) + j * bw, vals, bw, label=tier.capitalize(), color=COLORS[j], alpha=0.85)
    ax.set_xticks(np.arange(n_types) + bw); ax.set_xticklabels(qtypes, fontsize=8, rotation=15)
    ax.set_ylim(0, 1.15); ax.set_title("F1 by Question Type"); ax.legend(fontsize=8)

    # Plot 3: Threshold sweep
    ax = axes[0, 2]
    ax.plot(sweep_df["threshold"], sweep_df["eff_F1"], "b-o", ms=4, label="Eff. F1")
    ax.plot(sweep_df["threshold"], sweep_df["eff_EM"],  "g-s", ms=4, label="Eff. EM")
    ax2 = ax.twinx()
    ax2.plot(sweep_df["threshold"], sweep_df["debate_rate"] * 100, "r--^", ms=4, label="Debate %")
    ax2.set_ylabel("Debate rate (%)", color="red", fontsize=8)
    ax.axvline(0.35, color="gray", linestyle="--", lw=1.5, label="Current=0.35")
    ax.set_xlabel("Threshold"); ax.set_title("GK Threshold Sweep")
    ax.legend(loc="upper left", fontsize=7); ax2.legend(loc="upper right", fontsize=7)
    ax.set_ylim(0, 1)

    # Plot 4: Token cost
    ax = axes[1, 0]
    tok_vals = [df[f"{t}_tokens"].mean() for t in TIERS]
    bars = ax.bar(TIERS, tok_vals, color=COLORS, alpha=0.85)
    ax.set_title("Average Token Cost"); ax.set_ylabel("Tokens")
    for bar, v in zip(bars, tok_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 30, f"{v:.0f}", ha="center", fontsize=8)

    # Plot 5: Signal ablation
    ax = axes[1, 1]
    abl_plot = abl_df.sort_values("f1_delta")
    colors_abl = ["#d62728" if v < 0 else "#2ca02c" for v in abl_plot["f1_delta"]]
    ax.barh(abl_plot["signal_removed"], abl_plot["f1_delta"], color=colors_abl, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Signal Ablation (F1 delta if removed)"); ax.set_xlabel("ΔF1")
    ax.tick_params(axis="y", labelsize=7)

    # Plot 6: Per-sample F1 comparison
    ax = axes[1, 2]
    idx = range(len(df))
    ax.plot(idx, df["baseline_f1"],   "o-", color=COLORS[0], ms=5, label="Baseline",   alpha=0.8)
    ax.plot(idx, df["refinement_f1"], "s-", color=COLORS[1], ms=5, label="Refinement", alpha=0.8)
    ax.plot(idx, df["guardrag_f1"],   "^-", color=COLORS[2], ms=5, label="GUARD-RAG",  alpha=0.8)
    ax.set_title("Per-sample F1"); ax.set_xlabel("Sample"); ax.set_ylabel("F1")
    ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.1)
    # Mark debate-triggered samples
    debated = df[df["guardrag_debate_triggered"]].index
    ax.scatter(debated, [1.05] * len(debated), marker="v", color="red", s=20, label="Debate↓", zorder=5)

    plt.tight_layout()
    plt.savefig("ablation_analysis.png", dpi=150, bbox_inches="tight")
    print(f"\n{SEP}")
    print("Plot saved → ablation_analysis.png")
    plt.close()

print(f"\n{SEP}")
print("DONE")
print(SEP)

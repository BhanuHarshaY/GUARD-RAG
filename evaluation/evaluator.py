import pandas as pd
from tqdm import tqdm


def evaluate_all(samples, baseline_index, baseline_chunks, baseline_metadata,
                 improved_index, improved_chunks, improved_metadata,
                 embed_model, client, BASE_MODEL, JUDGE_MODEL,
                 mode="improved", max_samples=15, top_k=6, nli_model=None):
    from retrieval.retriever import retrieve
    from tiers.tier1 import tier1_basic_rag
    from tiers.tier2 import tier2_self_refine
    from tiers.tier3 import tier3_selective_debate
    from evaluation.metrics import compute_f1, compute_em

    if mode == "baseline":
        current_index = baseline_index
        current_chunks = baseline_chunks
        current_metadata = baseline_metadata
    else:
        current_index = improved_index
        current_chunks = improved_chunks
        current_metadata = improved_metadata

    records = []
    for sample in tqdm(samples[:max_samples]):
        question = sample["question"]
        gold = sample["gold_answer"]

        retrieved = retrieve(question, current_index, current_chunks, current_metadata, embed_model, top_k=top_k)

        t1 = tier1_basic_rag(question, retrieved, client, BASE_MODEL)
        t2 = tier2_self_refine(question, retrieved, client, BASE_MODEL, tier1_result=t1)
        t3 = tier3_selective_debate(question, retrieved, client, BASE_MODEL, JUDGE_MODEL, tier1_result=t1, nli_model=nli_model)

        records.append({
            "question": question,
            "gold_answer": gold,
            "tier1_answer": t1["answer"],
            "tier2_answer": t2["answer"],
            "tier3_answer": t3["answer"],
            "tier1_f1": compute_f1(t1["answer"], gold),
            "tier2_f1": compute_f1(t2["answer"], gold),
            "tier3_f1": compute_f1(t3["answer"], gold),
            "tier1_em": compute_em(t1["answer"], gold),
            "tier2_em": compute_em(t2["answer"], gold),
            "tier3_em": compute_em(t3["answer"], gold),
            "tier1_latency": t1["latency"],
            "tier2_latency": t2["latency"],
            "tier3_latency": t3["latency"],
            "tier1_tokens": t1["tokens"],
            "tier2_tokens": t2["tokens"],
            "tier3_tokens": t3["tokens"],
            "tier3_debate_triggered":    t3.get("debate_triggered", False),
            "tier3_gatekeeper_score":    t3.get("gatekeeper_score", 0.0),
            "tier3_threshold":           t3.get("threshold", 0.20),
            "tier3_adjudicator_verdict": t3.get("adjudicator_verdict", None),
            "mode": mode,
        })

    return pd.DataFrame(records)


def summarize_results(df, label):
    return pd.DataFrame([{
        "system": label,
        "tier1_f1": df["tier1_f1"].mean(),
        "tier2_f1": df["tier2_f1"].mean(),
        "tier3_f1": df["tier3_f1"].mean(),
        "tier1_em": df["tier1_em"].mean(),
        "tier2_em": df["tier2_em"].mean(),
        "tier3_em": df["tier3_em"].mean(),
        "tier1_avg_tokens": df["tier1_tokens"].mean(),
        "tier2_avg_tokens": df["tier2_tokens"].mean(),
        "tier3_avg_tokens": df["tier3_tokens"].mean(),
        "tier1_avg_latency": df["tier1_latency"].mean(),
        "tier2_avg_latency": df["tier2_latency"].mean(),
        "tier3_avg_latency": df["tier3_latency"].mean(),
        "tier3_debate_rate":            df["tier3_debate_triggered"].mean(),
        "tier3_avg_gatekeeper_score":   df["tier3_gatekeeper_score"].mean(),
        "tier3_abstention_rate":        (df["tier3_adjudicator_verdict"] == "ABSTAIN").mean(),
        "tier3_approve_rate":           (df["tier3_adjudicator_verdict"] == "APPROVE").mean(),
        "tier3_revise_rate":            (df["tier3_adjudicator_verdict"] == "REVISE").mean(),
    }])

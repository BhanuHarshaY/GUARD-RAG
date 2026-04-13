import pandas as pd
from tqdm import tqdm


def evaluate_all(samples, baseline_index, baseline_chunks, baseline_metadata,
                 improved_index, improved_chunks, improved_metadata,
                 embed_model, client, BASE_MODEL, JUDGE_MODEL,
                 mode="improved", max_samples=15, top_k=6, nli_model=None,
                 guardrag_threshold=0.35, disabled_signals=None):
    from retrieval.retriever import retrieve
    from tiers.baseline import baseline_rag
    from tiers.refinement import refinement_rag
    from tiers.guardrag import guardrag_debate
    from evaluation.metrics import compute_f1, compute_em, compute_hallucination_rate

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

        retrieved    = retrieve(question, current_index, current_chunks, current_metadata, embed_model, top_k=top_k)
        # GUARD-RAG gets 2× the context — T1/T2 are unaffected
        retrieved_g  = retrieve(question, current_index, current_chunks, current_metadata, embed_model, top_k=top_k * 2)

        try:
            b = baseline_rag(question, retrieved, client, BASE_MODEL)
            r = refinement_rag(question, retrieved, client, BASE_MODEL, baseline_result=b)
            g = guardrag_debate(question, retrieved_g, client, BASE_MODEL, JUDGE_MODEL, baseline_result=r, nli_model=nli_model, threshold=guardrag_threshold, disabled_signals=disabled_signals)
        except RuntimeError as e:
            print(f"\n  SKIPPED (rate limit exhausted): {question[:60]}")
            continue

        records.append({
            "question":                     question,
            "gold_answer":                  gold,
            "baseline_answer":              b["answer"],
            "refinement_answer":            r["answer"],
            "guardrag_answer":              g["answer"],
            "baseline_f1":                  compute_f1(b["answer"], gold),
            "refinement_f1":                compute_f1(r["answer"], gold),
            "guardrag_f1":                  compute_f1(g["answer"], gold),
            "baseline_em":                  compute_em(b["answer"], gold),
            "refinement_em":                compute_em(r["answer"], gold),
            "guardrag_em":                  compute_em(g["answer"], gold),
            "baseline_hallucination_rate":  compute_hallucination_rate(b["answer"], retrieved, nli_model),
            "refinement_hallucination_rate":compute_hallucination_rate(r["answer"], retrieved, nli_model),
            "guardrag_hallucination_rate":  compute_hallucination_rate(g["answer"], retrieved, nli_model),
            "baseline_latency":             b["latency"],
            "refinement_latency":           r["latency"],
            "guardrag_latency":             g["latency"],
            "baseline_tokens":              b["tokens"],
            "refinement_tokens":            r["tokens"],
            "guardrag_tokens":              g["tokens"],
            "guardrag_debate_triggered":    g.get("debate_triggered", False),
            "guardrag_gatekeeper_score":    g.get("gatekeeper_score", 0.0),
            "guardrag_gatekeeper_signals":  "|".join(g.get("gatekeeper_signals", [])),
            "guardrag_threshold":           g.get("threshold", 0.35),
            "guardrag_adjudicator_verdict": g.get("adjudicator_verdict", None),
            "mode":                         mode,
        })

    return pd.DataFrame(records)


def summarize_results(df, label):
    return pd.DataFrame([{
        "system":                       label,
        "baseline_f1":                  df["baseline_f1"].mean(),
        "refinement_f1":                df["refinement_f1"].mean(),
        "guardrag_f1":                  df["guardrag_f1"].mean(),
        "baseline_em":                  df["baseline_em"].mean(),
        "refinement_em":                df["refinement_em"].mean(),
        "guardrag_em":                  df["guardrag_em"].mean(),
        "baseline_avg_tokens":          df["baseline_tokens"].mean(),
        "refinement_avg_tokens":        df["refinement_tokens"].mean(),
        "guardrag_avg_tokens":          df["guardrag_tokens"].mean(),
        "baseline_avg_latency":         df["baseline_latency"].mean(),
        "refinement_avg_latency":       df["refinement_latency"].mean(),
        "guardrag_avg_latency":         df["guardrag_latency"].mean(),
        "baseline_hallucination_rate":  df["baseline_hallucination_rate"].mean(),
        "refinement_hallucination_rate":df["refinement_hallucination_rate"].mean(),
        "guardrag_hallucination_rate":  df["guardrag_hallucination_rate"].mean(),
        "guardrag_debate_rate":         df["guardrag_debate_triggered"].mean(),
        "guardrag_avg_gatekeeper_score":df["guardrag_gatekeeper_score"].mean(),
        "guardrag_abstention_rate":     (df["guardrag_adjudicator_verdict"] == "ABSTAIN").mean(),
        "guardrag_approve_rate":        (df["guardrag_adjudicator_verdict"] == "APPROVE").mean(),
        "guardrag_revise_rate":         (df["guardrag_adjudicator_verdict"] == "REVISE").mean(),
    }])

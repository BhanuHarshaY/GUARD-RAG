from retrieval.retriever import retrieve
from tiers.tier1 import tier1_basic_rag
from tiers.tier2 import tier2_self_refine
from tiers.tier3 import tier3_selective_debate


def run_comparison(question, baseline_index, baseline_chunks, baseline_metadata,
                   improved_index, improved_chunks, improved_metadata,
                   embed_model, client, BASE_MODEL, JUDGE_MODEL,
                   mode="improved", top_k=10, nli_model=None):
    if mode == "baseline":
        current_index = baseline_index
        current_chunks = baseline_chunks
        current_metadata = baseline_metadata
        mode_name = "BASELINE"
    elif mode == "improved":
        current_index = improved_index
        current_chunks = improved_chunks
        current_metadata = improved_metadata
        mode_name = "IMPROVED"
    else:
        raise ValueError("mode must be either 'baseline' or 'improved'")

    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print(f"RETRIEVER MODE: {mode_name}")
    print("=" * 70)

    retrieved = retrieve(
        question,
        current_index,
        current_chunks,
        current_metadata,
        embed_model,
        top_k=top_k,
        candidate_k=40
    )

    print(f"\nRetrieved {len(retrieved)} chunks:")
    for doc in retrieved[:12]:
        print(
            f"  [{doc['chunk_id']}] "
            f"({doc['chunk_type']}, score={doc.get('rerank_score', doc['score']):.3f}): "
            f"{doc['text'][:160]}..."
        )

    print(f"\n--- TIER 1: Basic RAG ---")
    t1 = tier1_basic_rag(question, retrieved, client, BASE_MODEL)
    print(f"Answer: {t1['answer']}")
    print(f"Latency: {t1['latency']}s | Tokens: {t1['tokens']}")

    print(f"\n--- TIER 2: Self-Refine ---")
    t2 = tier2_self_refine(question, retrieved, client, BASE_MODEL, tier1_result=t1)
    print(f"Initial: {t2['initial_answer']}")
    print(f"Final: {t2['answer']}")
    print(f"Latency: {t2['latency']}s | Tokens: {t2['tokens']}")

    print(f"\n--- TIER 3: Selective Debate ---")
    t3 = tier3_selective_debate(question, retrieved, client, BASE_MODEL, JUDGE_MODEL, tier1_result=t1, nli_model=nli_model)
    print(f"Debate triggered: {t3['debate_triggered']}")
    if t3.get("debate_triggered"):
        print(f"Gatekeeper signals: {t3.get('gatekeeper_signals', [])}")
    print(f"Answer: {t3['answer']}")
    print(f"Latency: {t3['latency']}s | Tokens: {t3['tokens']}")

    return t1, t2, t3

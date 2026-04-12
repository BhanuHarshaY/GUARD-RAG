from retrieval.retriever import retrieve
from tiers.baseline import baseline_rag
from tiers.refinement import refinement_rag
from tiers.guardrag import guardrag_debate


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

    print(f"\n--- BASELINE: Basic RAG ---")
    b = baseline_rag(question, retrieved, client, BASE_MODEL)
    print(f"Answer: {b['answer']}")
    print(f"Latency: {b['latency']}s | Tokens: {b['tokens']}")

    print(f"\n--- REFINEMENT: Self-Consistency + Self-Refine ---")
    r = refinement_rag(question, retrieved, client, BASE_MODEL, baseline_result=b)
    print(f"Initial: {r['initial_answer']}")
    print(f"Final: {r['answer']}")
    print(f"Latency: {r['latency']}s | Tokens: {r['tokens']}")

    print(f"\n--- GUARD-RAG: Selective Debate ---")
    g = guardrag_debate(question, retrieved, client, BASE_MODEL, JUDGE_MODEL, baseline_result=r, nli_model=nli_model)
    print(f"Debate triggered: {g['debate_triggered']}")
    print(f"Gatekeeper score: {g.get('gatekeeper_score', 0.0):.3f} (threshold={g.get('threshold', 0.20)})")
    if g.get("debate_triggered"):
        print(f"Gatekeeper signals: {g.get('gatekeeper_signals', [])}")
        print(f"Adjudicator verdict: {g.get('adjudicator_verdict', 'N/A')}")
        transcript = g.get("debate_transcript") or {}
        if transcript.get("skeptic"):
            print(f"\n  [Skeptic]\n{transcript['skeptic']}")
        if transcript.get("grounder"):
            print(f"\n  [Grounder]\n{transcript['grounder']}")
    print(f"\nAnswer: {g['answer']}")
    print(f"Latency: {g['latency']}s | Tokens: {g['tokens']}")

    return b, r, g

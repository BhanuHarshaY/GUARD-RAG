import faiss
import numpy as np

from indexing.table_parser import normalize_text_for_match

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "for", "to", "and", "or",
    "is", "are", "was", "were", "what", "which", "does", "do", "did",
    "used", "use", "under", "by", "with", "from"
}


def tokenize_for_overlap(text):
    text = normalize_text_for_match(text)
    return [t for t in text.split() if t and t not in STOPWORDS]


def lexical_overlap_score(query, text):
    q = set(tokenize_for_overlap(query))
    d = set(tokenize_for_overlap(text))
    if not q or not d:
        return 0.0
    return len(q & d) / max(1, len(q))


def question_is_list_like(question):
    q = normalize_text_for_match(question)
    patterns = [
        "consist", "consists", "include", "includes", "what are", "which are",
        "list", "components", "composed of", "made up of"
    ]
    return any(p in q for p in patterns)


def rerank_candidates(query, candidates):
    reranked = []
    list_like = question_is_list_like(query)

    for cand in candidates:
        score = cand["score"]

        lex = lexical_overlap_score(query, cand["text"])
        score += 0.30 * lex

        chunk_type = cand.get("chunk_type", "text")
        if chunk_type in {"table_cell", "table_row", "table_section"}:
            score += 0.08

        if list_like and chunk_type in {"table_cell", "table_row"}:
            score += 0.10

        row_label = cand.get("row_label")
        section_label = cand.get("section_label")
        if row_label and lexical_overlap_score(query, row_label) > 0:
            score += 0.15
        if section_label and lexical_overlap_score(query, section_label) > 0:
            score += 0.15

        cand = dict(cand)
        cand["rerank_score"] = score
        reranked.append(cand)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked


def expand_same_section(results, metadata, max_extra=8):
    if not results:
        return results

    existing_ids = {r["chunk_id"] for r in results}
    seed_sections = set()
    seed_doc_ids = set()

    for r in results:
        if r.get("section_label"):
            seed_sections.add((r["doc_id"], r["section_label"]))
            seed_doc_ids.add(r["doc_id"])

    if not seed_sections:
        return results

    extras = []
    for meta in metadata:
        key = (meta["doc_id"], meta.get("section_label"))
        if key in seed_sections and meta["chunk_type"] in {"table_cell", "table_row"}:
            if meta["chunk_id"] not in existing_ids:
                extras.append({
                    "text": meta["text"],
                    "score": 0.0,
                    "rerank_score": 0.0,
                    "doc_id": meta["doc_id"],
                    "chunk_id": meta["chunk_id"],
                    "chunk_type": meta["chunk_type"],
                    "section_label": meta.get("section_label"),
                    "row_label": meta.get("row_label"),
                    "column_header": meta.get("column_header"),
                    "value": meta.get("value"),
                })

    extras.sort(key=lambda x: (
        0 if x["chunk_type"] == "table_row" else 1,
        x.get("row_label") or "",
        x.get("column_header") or ""
    ))

    return results + extras[:max_extra]


def retrieve(query, index, chunks, metadata, embed_model, top_k=10, candidate_k=40):
    query_embedding = embed_model.encode([query])
    query_embedding = np.array(query_embedding, dtype="float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, min(candidate_k, len(chunks)))

    candidates = []
    for rank, idx in enumerate(indices[0]):
        if 0 <= idx < len(chunks):
            meta = metadata[idx]
            candidates.append({
                "text": chunks[idx],
                "score": float(scores[0][rank]),
                "doc_id": meta["doc_id"],
                "chunk_id": meta["chunk_id"],
                "chunk_type": meta.get("chunk_type", "text"),
                "section_label": meta.get("section_label"),
                "row_label": meta.get("row_label"),
                "column_header": meta.get("column_header"),
                "value": meta.get("value"),
            })

    reranked = rerank_candidates(query, candidates)
    selected = reranked[:top_k]
    expanded = expand_same_section(selected, metadata, max_extra=10)

    final_results = []
    seen = set()
    for r in expanded:
        if r["chunk_id"] not in seen:
            seen.add(r["chunk_id"])
            final_results.append(r)

    return final_results

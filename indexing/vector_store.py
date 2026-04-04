import re

import faiss
import numpy as np


def sentence_chunk_text(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p and len(p.strip()) > 10]


def build_vector_store(samples, embed_model, use_table_aware_chunking=False):
    chunks = []
    chunk_metadata = []
    seen = set()

    for sample in samples:
        doc_id = sample["doc_id"]

        if sample["context"]:
            for sent_idx, sent in enumerate(sentence_chunk_text(sample["context"])):
                key = ("text", doc_id, sent)
                if key in seen:
                    continue
                seen.add(key)

                chunks.append(sent)
                chunk_metadata.append({
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_text_{sent_idx}",
                    "chunk_type": "text",
                    "text": sent,
                    "section_label": None,
                    "row_label": None,
                    "column_header": None,
                    "value": None,
                })

        if use_table_aware_chunking:
            for table_idx, chunk in enumerate(sample.get("table_chunks", [])):
                text = chunk["chunk_text"]
                key = ("table", doc_id, text)
                if key in seen or len(text) < 3:
                    continue
                seen.add(key)

                chunks.append(text)
                chunk_metadata.append({
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_{chunk['chunk_type']}_{table_idx}",
                    "chunk_type": chunk["chunk_type"],
                    "text": text,
                    "section_label": chunk.get("section_label"),
                    "row_label": chunk.get("row_label"),
                    "column_header": chunk.get("column_header"),
                    "value": chunk.get("value"),
                })

    print(f"Created {len(chunks)} unique chunks from {len(samples)} samples")
    print("Embedding chunks...")
    embeddings = embed_model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    print(f"FAISS index built with {index.ntotal} vectors")
    return index, chunks, chunk_metadata

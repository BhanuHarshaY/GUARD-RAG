import json
import re

from indexing.table_parser import (
    clean_cell_text,
    normalize_text_for_match,
    is_year_or_period_header,
    choose_header_row,
    is_section_row,
    first_non_empty_cell,
    extract_table_chunks,
)


def load_tatqa(filepath, max_samples=100):
    with open(filepath, "r") as f:
        data = json.load(f)

    samples = []
    for entry in data:
        table = entry.get("table", {})
        paragraphs = entry.get("paragraphs", [])
        doc_id = entry.get("table", {}).get("uid", entry.get("uid", "unknown"))

        paragraph_parts = []
        for para in paragraphs:
            if para.get("text"):
                paragraph_parts.append(para["text"].strip())

        paragraph_context = "\n".join([p for p in paragraph_parts if p])
        table_chunks = extract_table_chunks(table, doc_id)

        for qa in entry.get("questions", []):
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            answer_type = qa.get("answer_type", "")

            if question and answer != "":
                samples.append({
                    "question": question,
                    "context": paragraph_context,
                    "gold_answer": answer,
                    "answer_type": answer_type,
                    "doc_id": doc_id,
                    "table_chunks": table_chunks,
                })

            if len(samples) >= max_samples:
                return samples

    return samples


def sentence_chunk_text(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p and len(p.strip()) > 10]

import re
import string
from collections import Counter

# NLI label index for cross-encoder/nli-deberta-v3-small:
# 0 = contradiction, 1 = entailment, 2 = neutral
_NLI_ENTAILMENT_IDX = 1


def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", str(text).strip())
    return [p.strip() for p in parts if len(p.strip()) > 1]


def _normalize_number(s):
    """Try to parse s as a number and return a canonical string, else return None."""
    try:
        # remove commas, spaces
        cleaned = re.sub(r'[,\s]', '', s)
        val = float(cleaned)
        # drop unnecessary trailing zeros: 42.0 → 42, 3.50 → 3.5
        if val == int(val):
            return str(int(val))
        return f"{val:.10g}"
    except (ValueError, OverflowError):
        return None


def normalize_answer(s):
    s = str(s).lower().strip()
    # Strip footnote superscripts attached to words: "inflation2" → "inflation"
    s = re.sub(r'([a-z])(\d+)\b', r'\1', s)
    # Remove articles and approximation words
    s = re.sub(r'\b(a|an|the|approximately|about|around|roughly|nearly)\b', ' ', s)
    # Remove unit words
    s = re.sub(r'\b(million|billion|thousand|usd|eur|gbp)\b', '', s)
    # Remove currency and percent symbols
    s = re.sub(r'[$%]', '', s)
    # Normalize commas in numbers: "4,369" → "4369"
    s = re.sub(r'(\d),(\d)', r'\1\2', s)
    # Normalize negative: "(4369)" accounting format → "-4369"
    s = re.sub(r'\((\d+\.?\d*)\)', r'-\1', s)
    # Strip trailing % unit text
    s = re.sub(r'\b(percent|pct)\b', '', s)
    # Remove remaining punctuation except hyphen (keep hyphen for negative numbers)
    preserve = {'-'}
    s = ''.join(ch for ch in s if ch not in string.punctuation or ch in preserve)
    s = re.sub(r'\s+', ' ', s).strip()
    # Canonicalize numeric strings: "42.0" → "42", "3.50" → "3.5"
    num = _normalize_number(s)
    if num is not None:
        return num
    return s


def compute_f1(prediction, ground_truth):
    if isinstance(ground_truth, list):
        # Try each element individually AND the full joined string
        scores = [compute_f1(prediction, g) for g in ground_truth]
        scores.append(compute_f1(prediction, " ".join(str(g) for g in ground_truth)))
        return max(scores, default=0.0)
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_em(prediction, ground_truth):
    if isinstance(ground_truth, list):
        # Try each element individually AND the full joined string
        if max((compute_em(prediction, g) for g in ground_truth), default=0.0):
            return 1.0
        return compute_em(prediction, " ".join(str(g) for g in ground_truth))
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def _chunk_to_natural_language(chunk):
    """
    Converts a retrieved chunk dict to a natural language sentence for NLI scoring.
    Pipe-delimited table chunks confuse DeBERTa; this renders them as prose.
    Falls back to raw chunk text on any missing field or parse error.
    """
    try:
        chunk_type = chunk.get("chunk_type", "text")

        if chunk_type == "table_cell":
            row   = chunk.get("row_label", "").strip()
            col   = chunk.get("column_header", "").strip()
            val   = chunk.get("value", "").strip()
            sec   = chunk.get("section_label", "").strip()
            if not (row and col and val):
                return chunk["text"]
            if sec:
                return f"The {row} for {col} under {sec} is {val}."
            return f"The {row} for {col} is {val}."

        if chunk_type == "table_row":
            row = chunk.get("row_label", "").strip()
            if not row:
                return chunk["text"]
            # text format: "Section: X | Row: Y | Col1 | Val1 ; Col2 | Val2"
            # pairs live after the last "|"-separated header segment, delimited by ";"
            text = chunk["text"]
            # strip leading "Section: X | Row: Y" prefix — everything before the first ";"
            pairs_part = text.split(";")
            col_vals = []
            for segment in pairs_part:
                parts = [p.strip() for p in segment.split("|")]
                # each segment is "Col | Val" (last two meaningful parts)
                # filter out "Section: ..." and "Row: ..." prefixes
                data_parts = [p for p in parts if not p.startswith("Section:") and not p.startswith("Row:")]
                if len(data_parts) >= 2:
                    col_name = data_parts[-2].replace("Column:", "").strip()
                    col_val  = data_parts[-1].replace("Value:", "").strip()
                    if col_name and col_val:
                        col_vals.append(f"{col_name}: {col_val}")
            if not col_vals:
                return chunk["text"]
            return f"{row} values: {', '.join(col_vals)}."

        if chunk_type == "table_section":
            sec = chunk.get("section_label", "").strip()
            if sec:
                return f"Section: {sec}."
            return chunk["text"]

        return chunk["text"]

    except Exception:
        return chunk.get("text", "")


def compute_hallucination_rate(answer, retrieved, nli_model):
    """
    Fraction of answer sentences NOT entailed by the retrieved context.
    Uses cross-encoder/nli-deberta-v3-small.

    Returns:
        float in [0, 1] — 0.0 means fully grounded, 1.0 means fully hallucinated.
        Returns 0.0 when nli_model is None or the answer has no scoreable sentences.
    """
    if nli_model is None:
        return 0.0

    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0

    context = " ".join(_chunk_to_natural_language(r) for r in retrieved)[:4500]
    pairs = [(context, s) for s in sentences]
    scores = nli_model.predict(pairs)   # shape: (n_sentences, 3)

    not_entailed = sum(1 for s in scores if s[_NLI_ENTAILMENT_IDX] < 0.5)
    return round(not_entailed / len(sentences), 4)

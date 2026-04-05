import re
import string
from collections import Counter

# NLI label index for cross-encoder/nli-deberta-v3-small:
# 0 = contradiction, 1 = entailment, 2 = neutral
_NLI_ENTAILMENT_IDX = 1


def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", str(text).strip())
    return [p.strip() for p in parts if len(p.strip()) > 1]


def normalize_answer(s):
    s = str(s).lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    preserve = {'.', ',', '-'}
    s = ''.join(ch for ch in s if ch not in string.punctuation or ch in preserve)
    s = re.sub(r'[$%]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def compute_f1(prediction, ground_truth):
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
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


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

    context = " ".join(r["text"] for r in retrieved)[:4500]
    pairs = [(context, s) for s in sentences]
    scores = nli_model.predict(pairs)   # shape: (n_sentences, 3)

    not_entailed = sum(1 for s in scores if s[_NLI_ENTAILMENT_IDX] < 0.5)
    return round(not_entailed / len(sentences), 4)

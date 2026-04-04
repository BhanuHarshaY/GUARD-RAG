import re
import string
from collections import Counter


def normalize_answer(s):
    s = str(s).lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
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

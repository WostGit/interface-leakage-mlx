"""Generic extraction logic for classification-like outputs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List

from models.linear import LinearStudent


Matrix = List[List[float]]


def response_to_soft_targets(resp: Dict[str, List], num_classes: int) -> Matrix:
    mode = resp["mode"][0]
    if mode == "probs":
        out = []
        for row in resp["probs"]:
            s = sum(max(v, 1e-12) for v in row)
            out.append([max(v, 1e-12) / s for v in row])
        return out

    if mode == "argmax":
        out = []
        for y in resp["labels"]:
            row = [0.0] * num_classes
            row[y] = 1.0
            out.append(row)
        return out

    if mode == "topk":
        out = []
        for idxs, vals in zip(resp["indices"], resp["values"]):
            row = [1e-6] * num_classes
            for i, v in zip(idxs, vals):
                row[i] = v
            s = sum(row)
            out.append([v / s for v in row])
        return out

    raise ValueError(f"Unknown response mode: {mode}")


def mean_kl(p: Matrix, q: Matrix) -> float:
    total = 0.0
    n = len(p)
    for pr, qr in zip(p, q):
        s = 0.0
        for pi, qi in zip(pr, qr):
            pi = max(pi, 1e-12)
            qi = max(qi, 1e-12)
            s += pi * (math.log(pi) - math.log(qi))
        total += s
    return total / n


@dataclass
class ExtractionResult:
    agreement: float
    kl_div: float


def evaluate_extraction(student: LinearStudent, x_test: Matrix, victim_test_probs: Matrix) -> ExtractionResult:
    p_student = student.predict_proba(x_test)
    agree = 0
    for ps, pv in zip(p_student, victim_test_probs):
        if max(range(len(ps)), key=lambda i: ps[i]) == max(range(len(pv)), key=lambda i: pv[i]):
            agree += 1
    return ExtractionResult(agreement=agree / len(x_test), kl_div=mean_kl(victim_test_probs, p_student))

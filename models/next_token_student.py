"""Simple student for next-token imitation based on last-token features."""

from __future__ import annotations

import math
import random
from typing import List


def _softmax(row: List[float]) -> List[float]:
    m = max(row)
    ex = [math.exp(v - m) for v in row]
    s = sum(ex)
    return [v / s for v in ex]


class NextTokenStudent:
    def __init__(self, vocab_size: int, seed: int = 0):
        self.vocab_size = vocab_size
        rng = random.Random(seed)
        self.transition_logits = [
            [rng.gauss(0, 0.01) for _ in range(vocab_size)] for _ in range(vocab_size)
        ]

    def predict_proba(self, contexts: List[List[int]]) -> List[List[float]]:
        return [_softmax(self.transition_logits[row[-1]]) for row in contexts]

    def fit_soft_targets(self, contexts: List[List[int]], y_soft: List[List[float]], epochs: int = 100, lr: float = 0.3):
        n = len(contexts)
        last = [row[-1] for row in contexts]
        for _ in range(epochs):
            preds = self.predict_proba(contexts)
            for i in range(n):
                t = last[i]
                for c in range(self.vocab_size):
                    grad = (preds[i][c] - y_soft[i][c]) / n
                    self.transition_logits[t][c] -= lr * grad

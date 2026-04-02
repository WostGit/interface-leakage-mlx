"""Toy linear classifier victim and student model (stdlib-only)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import List

from models.interfaces import InterfaceConfig, apply_interface


Vector = List[float]
Matrix = List[Vector]


def softmax(logits: Vector) -> Vector:
    m = max(logits)
    ex = [math.exp(v - m) for v in logits]
    s = sum(ex)
    return [v / s for v in ex]


def matvec(x: Vector, w: Matrix, b: Vector) -> Vector:
    out = [b[j] for j in range(len(b))]
    for i, xi in enumerate(x):
        for j in range(len(b)):
            out[j] += xi * w[i][j]
    return out


@dataclass
class ToyLinearVictim:
    weights: Matrix
    bias: Vector

    @classmethod
    def random(cls, input_dim: int, num_classes: int, seed: int = 0):
        rng = random.Random(seed)
        w = [[rng.gauss(0, 1.0) for _ in range(num_classes)] for _ in range(input_dim)]
        b = [rng.gauss(0, 0.5) for _ in range(num_classes)]
        return cls(weights=w, bias=b)

    def predict_proba(self, x: Matrix) -> Matrix:
        return [softmax(matvec(row, self.weights, self.bias)) for row in x]

    def query(self, x: Matrix, cfg: InterfaceConfig, seed: int = 0):
        return apply_interface(self.predict_proba(x), cfg, seed=seed)


@dataclass
class LinearStudent:
    input_dim: int
    num_classes: int
    seed: int = 0

    def __post_init__(self):
        rng = random.Random(self.seed)
        self.weights = [[rng.gauss(0, 0.01) for _ in range(self.num_classes)] for _ in range(self.input_dim)]
        self.bias = [0.0 for _ in range(self.num_classes)]

    def predict_proba(self, x: Matrix) -> Matrix:
        return [softmax(matvec(row, self.weights, self.bias)) for row in x]

    def fit_soft_targets(self, x: Matrix, y_soft: Matrix, epochs: int = 120, lr: float = 0.2):
        n = len(x)
        for _ in range(epochs):
            preds = self.predict_proba(x)
            grad_w = [[0.0 for _ in range(self.num_classes)] for _ in range(self.input_dim)]
            grad_b = [0.0 for _ in range(self.num_classes)]
            for i, row in enumerate(x):
                for c in range(self.num_classes):
                    g = (preds[i][c] - y_soft[i][c]) / n
                    grad_b[c] += g
                    for d, xv in enumerate(row):
                        grad_w[d][c] += xv * g
            for d in range(self.input_dim):
                for c in range(self.num_classes):
                    self.weights[d][c] -= lr * grad_w[d][c]
            for c in range(self.num_classes):
                self.bias[c] -= lr * grad_b[c]

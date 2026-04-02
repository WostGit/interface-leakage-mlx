from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.interfaces import softmax


@dataclass
class StudentLinearModel:
    input_dim: int
    num_classes: int
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w = rng.normal(0.0, 0.05, size=(self.input_dim, self.num_classes))
        self.b = np.zeros((self.num_classes,), dtype=np.float64)

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w + self.b

    def predict_probs(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.predict_logits(x), axis=1)

    def fit_distribution_targets(
        self,
        x: np.ndarray,
        target_probs: np.ndarray,
        epochs: int = 150,
        lr: float = 0.3,
    ) -> None:
        n = x.shape[0]
        for _ in range(epochs):
            pred = self.predict_probs(x)
            grad_logits = (pred - target_probs) / n
            grad_w = x.T @ grad_logits
            grad_b = grad_logits.sum(axis=0)
            self.w -= lr * grad_w
            self.b -= lr * grad_b

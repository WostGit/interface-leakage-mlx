from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.victim_interfaces import softmax


@dataclass
class LinearStudent:
    in_dim: int
    out_dim: int
    lr: float = 0.2
    seed: int = 123

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(0, 0.05, size=(self.in_dim, self.out_dim))
        self.b = np.zeros((self.out_dim,))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.W + self.b
        return softmax(logits)

    def train_step(self, X: np.ndarray, target_probs: np.ndarray) -> float:
        P = self.predict_proba(X)
        n = X.shape[0]
        grad_logits = (P - target_probs) / n
        grad_W = X.T @ grad_logits
        grad_b = grad_logits.sum(axis=0)
        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b
        loss = -np.mean(np.sum(target_probs * np.log(np.clip(P, 1e-9, 1.0)), axis=1))
        return float(loss)

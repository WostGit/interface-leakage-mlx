from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from models.interfaces import InterfaceConfig, softmax


@dataclass
class TokenStudent:
    vocab_size: int
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w = rng.normal(0.0, 0.02, size=(self.vocab_size, self.vocab_size))
        self.b = np.zeros((self.vocab_size,), dtype=np.float64)

    def _features(self, contexts: List[List[int]]) -> np.ndarray:
        last_tokens = np.array([c[-1] for c in contexts], dtype=np.int64)
        x = np.zeros((len(contexts), self.vocab_size), dtype=np.float64)
        x[np.arange(len(contexts)), last_tokens] = 1.0
        return x

    def predict_probs(self, contexts: List[List[int]]) -> np.ndarray:
        x = self._features(contexts)
        logits = x @ self.w + self.b
        return softmax(logits, axis=1)

    def fit(self, contexts: List[List[int]], targets: np.ndarray, epochs: int = 120, lr: float = 0.5) -> None:
        x = self._features(contexts)
        n = x.shape[0]
        for _ in range(epochs):
            pred = self.predict_probs(contexts)
            grad_logits = (pred - targets) / n
            self.w -= lr * (x.T @ grad_logits)
            self.b -= lr * grad_logits.sum(axis=0)


@dataclass
class TokenExtractionAttacker:
    vocab_size: int
    seed: int = 0

    def _targets(self, out: Dict[str, np.ndarray], interface: InterfaceConfig) -> np.ndarray:
        if interface.mode == "argmax":
            labels = out["labels"]
            t = np.zeros((labels.shape[0], self.vocab_size), dtype=np.float64)
            t[np.arange(labels.shape[0]), labels] = 1.0
            return t

        if interface.mode == "topk":
            idx = out["topk_indices"]
            p = out["topk_probs"]
            t = np.full((idx.shape[0], self.vocab_size), 1e-6, dtype=np.float64)
            for i in range(idx.shape[1]):
                t[np.arange(idx.shape[0]), idx[:, i]] = p[:, i]
            t /= t.sum(axis=1, keepdims=True)
            return t

        if interface.mode == "full":
            return out["probs"]

        raise ValueError(interface.mode)

    def extract(self, victim, contexts: List[List[int]], interface: InterfaceConfig) -> TokenStudent:
        out = victim.query_next_token(contexts, interface)
        t = self._targets(out, interface)
        student = TokenStudent(vocab_size=self.vocab_size, seed=self.seed)
        student.fit(contexts, t)
        return student

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .common import InterfaceResponse, softmax

InterfaceType = Literal["argmax", "topk", "probs"]


@dataclass
class VictimLinearModel:
    w: np.ndarray
    b: np.ndarray
    noise_std: float = 0.0

    def logits(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w + self.b

    def predict_probs(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.logits(x))

    def query(
        self,
        x: np.ndarray,
        interface: InterfaceType,
        topk: int = 3,
        rng: np.random.Generator | None = None,
    ) -> InterfaceResponse:
        probs = self.predict_probs(x)
        if interface == "argmax":
            labels = np.argmax(probs, axis=1)
            out = np.zeros_like(probs)
            out[np.arange(x.shape[0]), labels] = 1.0
            return InterfaceResponse(probs=out, labels=labels)

        if interface == "topk":
            idx = np.argpartition(-probs, kth=topk - 1, axis=1)[:, :topk]
            out = np.zeros_like(probs)
            row = np.arange(x.shape[0])[:, None]
            out[row, idx] = probs[row, idx]
            z = np.sum(out, axis=1, keepdims=True)
            out = out / np.clip(z, 1e-8, None)
            labels = np.argmax(out, axis=1)
            return InterfaceResponse(probs=out, labels=labels)

        if interface == "probs":
            out = probs.copy()
            if self.noise_std > 0.0:
                if rng is None:
                    rng = np.random.default_rng(0)
                out = out + rng.normal(0.0, self.noise_std, size=out.shape)
                out = np.clip(out, 1e-8, None)
                out = out / np.sum(out, axis=1, keepdims=True)
            labels = np.argmax(out, axis=1)
            return InterfaceResponse(probs=out, labels=labels)

        raise ValueError(f"Unknown interface: {interface}")


@dataclass
class StudentLinearModel:
    d_in: int
    n_classes: int
    lr: float = 0.2
    steps: int = 120
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w = rng.normal(0.0, 0.02, size=(self.d_in, self.n_classes))
        self.b = np.zeros((self.n_classes,), dtype=np.float64)

    def predict_probs(self, x: np.ndarray) -> np.ndarray:
        return softmax(x @ self.w + self.b)

    def fit(self, x: np.ndarray, target_probs: np.ndarray) -> None:
        n = x.shape[0]
        for _ in range(self.steps):
            preds = self.predict_probs(x)
            grad_logits = (preds - target_probs) / n
            grad_w = x.T @ grad_logits
            grad_b = np.sum(grad_logits, axis=0)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

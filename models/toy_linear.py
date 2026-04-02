from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.victim_interfaces import Interface, interface_projection, softmax


@dataclass
class ToyLinearVictim:
    n_features: int
    n_classes: int
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(0, 1.0, size=(self.n_features, self.n_classes))
        self.b = rng.normal(0, 0.3, size=(self.n_classes,))

    def logits(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b

    def probs(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.logits(x))

    def query(
        self,
        x: np.ndarray,
        interface: Interface,
        k: int = 3,
        noise_std: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        return interface_projection(self.probs(x), interface=interface, k=k, noise_std=noise_std, rng=rng)


def make_toy_dataset(seed: int, n: int, n_features: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, size=(n, n_features))

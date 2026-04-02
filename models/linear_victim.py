from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from models.interfaces import InterfaceConfig, package_interface_output, softmax


@dataclass
class LinearVictim:
    input_dim: int
    num_classes: int
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.w = self.rng.normal(0.0, 1.0, size=(self.input_dim, self.num_classes))
        self.b = self.rng.normal(0.0, 0.2, size=(self.num_classes,))

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w + self.b

    def predict_probs(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.predict_logits(x), axis=1)

    def query(self, x: np.ndarray, interface: InterfaceConfig) -> Dict[str, np.ndarray]:
        probs = self.predict_probs(x)
        return package_interface_output(probs, interface, self.rng)

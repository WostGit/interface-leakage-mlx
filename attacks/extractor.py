from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from attacks.student_linear import StudentLinearModel
from models.interfaces import InterfaceConfig


@dataclass
class ExtractionAttacker:
    input_dim: int
    num_classes: int
    seed: int = 0

    def _to_target_distribution(
        self,
        victim_output: Dict[str, np.ndarray],
        interface: InterfaceConfig,
    ) -> np.ndarray:
        if interface.mode == "argmax":
            labels = victim_output["labels"]
            target = np.zeros((labels.shape[0], self.num_classes), dtype=np.float64)
            target[np.arange(labels.shape[0]), labels] = 1.0
            return target

        if interface.mode == "topk":
            idx = victim_output["topk_indices"]
            p = victim_output["topk_probs"]
            target = np.full((idx.shape[0], self.num_classes), 1e-6, dtype=np.float64)
            for i in range(idx.shape[1]):
                target[np.arange(idx.shape[0]), idx[:, i]] = p[:, i]
            target /= target.sum(axis=1, keepdims=True)
            return target

        if interface.mode == "full":
            return victim_output["probs"]

        raise ValueError(f"Unsupported interface mode: {interface.mode}")

    def extract(
        self,
        victim,
        interface: InterfaceConfig,
        x_queries: np.ndarray,
        train_epochs: int = 150,
        lr: float = 0.3,
    ) -> StudentLinearModel:
        victim_output = victim.query(x_queries, interface)
        targets = self._to_target_distribution(victim_output, interface)

        student = StudentLinearModel(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            seed=self.seed,
        )
        student.fit_distribution_targets(
            x=x_queries,
            target_probs=targets,
            epochs=train_epochs,
            lr=lr,
        )
        return student

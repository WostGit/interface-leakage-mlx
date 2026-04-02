from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.common import kl_divergence, top1_agreement


@dataclass
class ExtractionMetrics:
    budget: int
    interface: str
    top1_agreement: float
    kl_div: float
    seed: int
    domain: str
    backend: str


def evaluate_extraction(victim_probs: np.ndarray, student_probs: np.ndarray) -> tuple[float, float]:
    return top1_agreement(victim_probs, student_probs), kl_divergence(victim_probs, student_probs)

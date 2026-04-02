from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class InterfaceResponse:
    """Container returned by victim interfaces."""

    probs: np.ndarray
    labels: np.ndarray


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    p2 = np.clip(p, eps, 1.0)
    q2 = np.clip(q, eps, 1.0)
    return float(np.mean(np.sum(p2 * (np.log(p2) - np.log(q2)), axis=1)))


def top1_agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.argmax(a, axis=1) == np.argmax(b, axis=1)))


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def train_test_split(x: np.ndarray, ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    k = int(n * ratio)
    return x[:k], x[k:]

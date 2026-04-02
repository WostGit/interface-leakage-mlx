from __future__ import annotations

import numpy as np


def top1_agreement(victim_probs: np.ndarray, student_probs: np.ndarray) -> float:
    v = np.argmax(victim_probs, axis=1)
    s = np.argmax(student_probs, axis=1)
    return float(np.mean(v == s))


def mean_kl_divergence(victim_probs: np.ndarray, student_probs: np.ndarray) -> float:
    eps = 1e-8
    v = np.clip(victim_probs, eps, 1.0)
    s = np.clip(student_probs, eps, 1.0)
    return float(np.mean(np.sum(v * (np.log(v) - np.log(s)), axis=1)))

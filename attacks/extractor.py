from __future__ import annotations

import numpy as np

from attacks.student_models import LinearStudent


def responses_to_targets(responses: list[dict], n_classes: int, interface: str) -> np.ndarray:
    Y = np.zeros((len(responses), n_classes), dtype=np.float64)
    for i, r in enumerate(responses):
        if interface == "argmax":
            Y[i, r["label"]] = 1.0
        elif interface == "topk":
            idx = r["topk_indices"]
            probs = r["topk_probs"]
            Y[i, idx] = probs
        else:
            Y[i] = r["probs"]
    Y = np.clip(Y, 1e-9, None)
    Y /= Y.sum(axis=1, keepdims=True)
    return Y


def extract_student(
    X_query: np.ndarray,
    responses: list[dict],
    interface: str,
    n_classes: int,
    seed: int,
    epochs: int = 120,
    lr: float = 0.2,
) -> LinearStudent:
    student = LinearStudent(in_dim=X_query.shape[1], out_dim=n_classes, lr=lr, seed=seed)
    Y = responses_to_targets(responses, n_classes=n_classes, interface=interface)
    for _ in range(epochs):
        student.train_step(X_query, Y)
    return student


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1.0)
    q = np.clip(q, 1e-9, 1.0)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=1)))

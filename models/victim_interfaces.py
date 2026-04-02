from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np

Interface = Literal["argmax", "topk", "probs"]


@dataclass
class VictimResponse:
    label: int
    topk_indices: np.ndarray
    topk_probs: np.ndarray
    probs: np.ndarray


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def interface_projection(
    probs: np.ndarray,
    interface: Interface,
    k: int = 3,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Dict[str, np.ndarray | int]:
    if rng is None:
        rng = np.random.default_rng(0)

    p = np.array(probs, copy=True)
    if noise_std > 0:
        p = p + rng.normal(0.0, noise_std, size=p.shape)
        p = np.clip(p, 1e-9, None)
        p = p / p.sum()

    label = int(np.argmax(p))
    if interface == "argmax":
        return {"label": label}

    idx = np.argsort(p)[::-1][:k]
    topk_probs = p[idx]
    topk_probs = topk_probs / topk_probs.sum()

    if interface == "topk":
        return {"label": label, "topk_indices": idx, "topk_probs": topk_probs}

    return {"label": label, "probs": p}

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class InterfaceConfig:
    mode: str
    top_k: int = 3
    noise_std: float = 0.0


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def add_prob_noise(probs: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    if noise_std <= 0.0:
        return probs
    noisy = probs + rng.normal(0.0, noise_std, size=probs.shape)
    noisy = np.clip(noisy, 1e-8, None)
    return noisy / noisy.sum(axis=-1, keepdims=True)


def package_interface_output(
    probs: np.ndarray,
    config: InterfaceConfig,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    if config.mode == "argmax":
        return {"labels": np.argmax(probs, axis=1).astype(np.int64)}

    if config.mode == "topk":
        k = min(config.top_k, probs.shape[1])
        idx = np.argpartition(-probs, k - 1, axis=1)[:, :k]
        sorted_idx = np.take_along_axis(
            idx,
            np.argsort(-np.take_along_axis(probs, idx, axis=1), axis=1),
            axis=1,
        )
        topk_vals = np.take_along_axis(probs, sorted_idx, axis=1)
        topk_vals = topk_vals / np.sum(topk_vals, axis=1, keepdims=True)
        return {"topk_indices": sorted_idx.astype(np.int64), "topk_probs": topk_vals}

    if config.mode == "full":
        out = add_prob_noise(probs, config.noise_std, rng)
        return {"probs": out}

    raise ValueError(f"Unknown interface mode: {config.mode}")

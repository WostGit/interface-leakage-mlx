"""Victim API interfaces used by all experiments."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List


@dataclass
class InterfaceConfig:
    mode: str  # argmax | topk | probs
    topk: int = 3
    noise_std: float = 0.0


def _normalize(row: List[float]) -> List[float]:
    s = sum(max(v, 1e-12) for v in row)
    return [max(v, 1e-12) / s for v in row]


def apply_interface(probs: List[List[float]], cfg: InterfaceConfig, seed: int = 0) -> Dict[str, List]:
    rng = random.Random(seed)
    p = [_normalize(row) for row in probs]

    if cfg.mode == "argmax":
        labels = [max(range(len(row)), key=lambda i: row[i]) for row in p]
        return {"mode": ["argmax"], "labels": labels}

    if cfg.mode == "topk":
        all_idx, all_vals = [], []
        for row in p:
            order = sorted(range(len(row)), key=lambda i: row[i], reverse=True)[: cfg.topk]
            all_idx.append(order)
            all_vals.append([row[i] for i in order])
        return {"mode": ["topk"], "indices": all_idx, "values": all_vals}

    if cfg.mode == "probs":
        if cfg.noise_std > 0:
            out = []
            for row in p:
                noisy = [v + rng.gauss(0, cfg.noise_std) for v in row]
                out.append(_normalize(noisy))
            p = out
        return {"mode": ["probs"], "probs": p}

    raise ValueError(f"Unknown interface mode: {cfg.mode}")

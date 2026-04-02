from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from models.interfaces import InterfaceConfig, package_interface_output


@dataclass
class SyntheticTokenVictim:
    vocab_size: int = 128
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.table = self.rng.normal(0.0, 1.0, size=(self.vocab_size, self.vocab_size))

    def _context_to_logits(self, context: List[int]) -> np.ndarray:
        last = context[-1]
        logits = self.table[last].copy()
        if len(context) > 1:
            logits += 0.2 * self.table[context[-2]]
        return logits

    def query_next_token(self, contexts: List[List[int]], interface: InterfaceConfig) -> Dict[str, np.ndarray]:
        logits = np.stack([self._context_to_logits(c) for c in contexts], axis=0)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        return package_interface_output(probs, interface, self.rng)

    def true_probs(self, contexts: List[List[int]]) -> np.ndarray:
        logits = np.stack([self._context_to_logits(c) for c in contexts], axis=0)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        return probs / probs.sum(axis=1, keepdims=True)


@dataclass
class MlxQwenVictim:
    model_name: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    seed: int = 0

    def __post_init__(self) -> None:
        self.available = False
        self.reason = "Not initialized"
        try:
            import mlx.core as mx  # type: ignore
            from mlx_lm import load  # type: ignore

            self.mx = mx
            self.model, self.tokenizer = load(self.model_name)
            self.available = True
            self.reason = "MLX model loaded"
            mx.random.seed(self.seed)
        except Exception as exc:  # noqa: BLE001
            self.available = False
            self.reason = f"MLX path unavailable: {exc}"

    def query_next_token(self, contexts: List[List[int]], interface: InterfaceConfig) -> Dict[str, np.ndarray]:
        if not self.available:
            raise RuntimeError(self.reason)

        probs = []
        for context in contexts:
            x = self.mx.array([context])
            logits = self.model(x)[:, -1, :]
            logits_np = np.array(logits)
            p = np.exp(logits_np - logits_np.max(axis=1, keepdims=True))
            p = p / p.sum(axis=1, keepdims=True)
            probs.append(p[0])
        probs_arr = np.stack(probs, axis=0)
        return package_interface_output(probs_arr, interface, np.random.default_rng(self.seed))

    def true_probs(self, contexts: List[List[int]]) -> np.ndarray:
        if not self.available:
            raise RuntimeError(self.reason)
        probs = []
        for context in contexts:
            x = self.mx.array([context])
            logits = self.model(x)[:, -1, :]
            logits_np = np.array(logits)
            p = np.exp(logits_np - logits_np.max(axis=1, keepdims=True))
            p = p / p.sum(axis=1, keepdims=True)
            probs.append(p[0])
        return np.stack(probs, axis=0)


def make_token_contexts(seed: int, n: int, vocab_size: int, length: int = 12) -> Tuple[List[List[int]], List[List[int]]]:
    rng = np.random.default_rng(seed)
    train = [rng.integers(0, vocab_size, size=length).tolist() for _ in range(n)]
    eval_c = [rng.integers(0, vocab_size, size=length).tolist() for _ in range(max(256, n // 2))]
    return train, eval_c

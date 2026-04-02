"""Next-token victim wrappers with optional MLX Qwen path and stdlib fallback."""

from __future__ import annotations

import math
import random
from typing import List

from models.interfaces import InterfaceConfig, apply_interface


def _softmax(row: List[float]) -> List[float]:
    m = max(row)
    ex = [math.exp(v - m) for v in row]
    s = sum(ex)
    return [v / s for v in ex]


class BaseNextTokenVictim:
    vocab_size: int

    def predict_proba(self, contexts: List[List[int]]) -> List[List[float]]:
        raise NotImplementedError

    def query(self, contexts: List[List[int]], cfg: InterfaceConfig, seed: int = 0):
        return apply_interface(self.predict_proba(contexts), cfg, seed=seed)


class ToyBigramVictim(BaseNextTokenVictim):
    def __init__(self, vocab_size: int, seed: int = 0):
        self.vocab_size = vocab_size
        rng = random.Random(seed)
        self.transition_logits = [
            [rng.gauss(0, 1.0) for _ in range(vocab_size)] for _ in range(vocab_size)
        ]

    def predict_proba(self, contexts: List[List[int]]) -> List[List[float]]:
        return [_softmax(self.transition_logits[row[-1]]) for row in contexts]


class QwenMLXVictim(BaseNextTokenVictim):
    """Optional real Qwen2-0.5B victim path.

    This only works on Apple Silicon with mlx + mlx-lm installed.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2-0.5B", max_context: int = 16):
        self.model_id = model_id
        self.max_context = max_context
        try:
            from mlx_lm import load  # type: ignore

            self.model, self.tokenizer = load(self.model_id)
            self.vocab_size = int(self.tokenizer.vocab_size)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Qwen MLX unavailable in this environment") from exc

    def predict_proba(self, contexts: List[List[int]]) -> List[List[float]]:
        import mlx.core as mx  # type: ignore

        out = []
        for row in contexts:
            txt = self.tokenizer.decode(row)
            toks = self.tokenizer.encode(txt)[-self.max_context :]
            logits = self.model(mx.array([toks]))[:, -1, :]
            probs = mx.softmax(logits, axis=-1)
            out.append(list(map(float, probs.tolist()[0])))
        return out


def build_next_token_victim(prefer_real_qwen: bool, vocab_size: int, seed: int):
    if prefer_real_qwen:
        try:
            return QwenMLXVictim()
        except RuntimeError:
            pass
    return ToyBigramVictim(vocab_size=vocab_size, seed=seed)

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np

from models.victim_interfaces import Interface, interface_projection


@dataclass
class LLMVictimConfig:
    model_id: str = "Qwen/Qwen2-0.5B-Instruct"
    vocab_size: int = 128
    topk_default: int = 5
    seed: int = 7


class LLMVictim:
    """Victim wrapper.

    - Tries MLX + Qwen2-0.5B on Apple Silicon unless FORCE_CPU=1.
    - Falls back to deterministic pseudo-logits on CPU-only environments.
    """

    def __init__(self, cfg: LLMVictimConfig):
        self.cfg = cfg
        self.backend = "fallback"
        self._rng = np.random.default_rng(cfg.seed)
        self._mlx_model = None
        self._tokenizer = None

        if os.getenv("FORCE_CPU", "0") != "1":
            self._try_init_mlx()

    def _try_init_mlx(self) -> None:
        try:
            import mlx.core as mx  # type: ignore
            from mlx_lm import load  # type: ignore

            self._mx = mx
            self._mlx_model, self._tokenizer = load(self.cfg.model_id)
            self.backend = "mlx-qwen"
        except Exception:
            self.backend = "fallback"

    def _fallback_probs(self, prompt: str) -> np.ndarray:
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little") ^ self.cfg.seed
        rng = np.random.default_rng(seed)
        logits = rng.normal(0, 1.0, size=(self.cfg.vocab_size,))
        # introduce low-rank structure so extraction is possible
        for i, ch in enumerate(prompt[:16]):
            logits[(ord(ch) + i) % self.cfg.vocab_size] += 0.7
        logits = logits - logits.max()
        p = np.exp(logits)
        return p / p.sum()

    def _mlx_probs(self, prompt: str) -> np.ndarray:
        # Best-effort MLX path; if runtime API mismatches, fallback safely.
        try:
            token_ids = self._tokenizer.encode(prompt)
            x = self._mx.array([token_ids])
            out = self._mlx_model(x)
            logits = np.array(out.logits[0, -1, : self.cfg.vocab_size])
            logits = logits - logits.max()
            p = np.exp(logits)
            return p / p.sum()
        except Exception:
            return self._fallback_probs(prompt)

    def probs(self, prompt: str) -> np.ndarray:
        if self.backend == "mlx-qwen":
            return self._mlx_probs(prompt)
        return self._fallback_probs(prompt)

    def query(
        self,
        prompt: str,
        interface: Interface,
        k: int | None = None,
        noise_std: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        k = k or self.cfg.topk_default
        return interface_projection(self.probs(prompt), interface=interface, k=k, noise_std=noise_std, rng=rng)

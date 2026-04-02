from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .common import InterfaceResponse, softmax

InterfaceType = Literal["argmax", "topk", "probs"]


@dataclass
class TinyFallbackLM:
    vocab_size: int = 96
    feature_dim: int = 64
    seed: int = 123

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w = rng.normal(0.0, 0.8, size=(self.feature_dim, self.vocab_size))
        self.b = rng.normal(0.0, 0.2, size=(self.vocab_size,))

    def featurize(self, prompts: list[str]) -> np.ndarray:
        x = np.zeros((len(prompts), self.feature_dim), dtype=np.float64)
        for i, p in enumerate(prompts):
            for j, ch in enumerate(p.encode("utf-8")):
                x[i, (ch + 17 * j) % self.feature_dim] += 1.0
        x /= np.maximum(1.0, np.linalg.norm(x, axis=1, keepdims=True))
        return x

    def next_token_probs(self, prompts: list[str]) -> np.ndarray:
        x = self.featurize(prompts)
        return softmax(x @ self.w + self.b)


class MLXQwenVictim:
    """Uses mlx-lm if available; otherwise falls back to TinyFallbackLM."""

    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B-Instruct", seed: int = 0) -> None:
        self.model_name = model_name
        self.seed = seed
        self.mode = "fallback"
        self.fallback = TinyFallbackLM(seed=seed + 11)
        self._mlx_model = None
        self._tokenizer = None
        try:
            from mlx_lm import load  # type: ignore
            import mlx.core as mx  # type: ignore

            self._mx = mx
            self._mlx_model, self._tokenizer = load(model_name)
            self.mode = "mlx_qwen"
        except Exception:
            self.mode = "fallback"

    def _mlx_next_token_probs(self, prompts: list[str], max_vocab: int = 256) -> np.ndarray:
        # Defensive implementation: if any model/tokenizer behavior differs, fallback is used.
        try:
            mx = self._mx
            all_rows = []
            for p in prompts:
                ids = self._tokenizer.encode(p)
                arr = mx.array([ids])
                out = self._mlx_model(arr)
                logits = np.array(out[0, -1, :])
                top = np.argpartition(-logits, kth=min(max_vocab, logits.size - 1))[:max_vocab]
                top_logits = logits[top]
                probs = np.zeros((max_vocab,), dtype=np.float64)
                top_probs = np.exp(top_logits - np.max(top_logits))
                top_probs /= np.sum(top_probs)
                probs[:] = top_probs
                all_rows.append(probs)
            return np.vstack(all_rows)
        except Exception:
            self.mode = "fallback"
            return self.fallback.next_token_probs(prompts)

    def next_token_probs(self, prompts: list[str]) -> np.ndarray:
        if self.mode == "mlx_qwen":
            return self._mlx_next_token_probs(prompts)
        return self.fallback.next_token_probs(prompts)

    def query(
        self,
        prompts: list[str],
        interface: InterfaceType,
        topk: int = 5,
        noise_std: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> InterfaceResponse:
        probs = self.next_token_probs(prompts)
        if interface == "argmax":
            labels = np.argmax(probs, axis=1)
            out = np.zeros_like(probs)
            out[np.arange(len(prompts)), labels] = 1.0
            return InterfaceResponse(probs=out, labels=labels)

        if interface == "topk":
            idx = np.argpartition(-probs, kth=topk - 1, axis=1)[:, :topk]
            out = np.zeros_like(probs)
            row = np.arange(len(prompts))[:, None]
            out[row, idx] = probs[row, idx]
            out /= np.clip(np.sum(out, axis=1, keepdims=True), 1e-8, None)
            labels = np.argmax(out, axis=1)
            return InterfaceResponse(probs=out, labels=labels)

        if interface == "probs":
            out = probs.copy()
            if noise_std > 0.0:
                if rng is None:
                    rng = np.random.default_rng(0)
                out += rng.normal(0.0, noise_std, size=out.shape)
                out = np.clip(out, 1e-8, None)
                out /= np.sum(out, axis=1, keepdims=True)
            labels = np.argmax(out, axis=1)
            return InterfaceResponse(probs=out, labels=labels)

        raise ValueError(interface)


@dataclass
class StudentLMHead:
    feature_dim: int
    vocab_size: int
    seed: int = 0
    lr: float = 0.3
    steps: int = 150

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w = rng.normal(0.0, 0.01, size=(self.feature_dim, self.vocab_size))
        self.b = np.zeros((self.vocab_size,), dtype=np.float64)

    def featurize(self, prompts: list[str]) -> np.ndarray:
        x = np.zeros((len(prompts), self.feature_dim), dtype=np.float64)
        for i, p in enumerate(prompts):
            for j, ch in enumerate(p.encode("utf-8")):
                x[i, (ch * 31 + j) % self.feature_dim] += 1.0
        x /= np.maximum(1.0, np.linalg.norm(x, axis=1, keepdims=True))
        return x

    def predict_probs(self, prompts: list[str]) -> np.ndarray:
        x = self.featurize(prompts)
        return softmax(x @ self.w + self.b)

    def fit(self, prompts: list[str], target_probs: np.ndarray) -> None:
        x = self.featurize(prompts)
        n = x.shape[0]
        for _ in range(self.steps):
            pred = softmax(x @ self.w + self.b)
            grad_logits = (pred - target_probs) / n
            self.w -= self.lr * (x.T @ grad_logits)
            self.b -= self.lr * np.sum(grad_logits, axis=0)

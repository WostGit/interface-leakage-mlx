from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from attacks.extractor import extract_student, kl_divergence
from models.llm_victim import LLMVictim, LLMVictimConfig


PROMPTS = [
    "The weather in San Francisco is",
    "A good unit test should",
    "Machine learning models are vulnerable because",
    "In a distant galaxy, the captain",
    "The quick brown fox",
    "Data privacy matters when",
    "A recipe for pancakes includes",
    "Debugging is easier when",
    "The meaning of reproducibility is",
    "Neural networks can",
]


def prompt_features(prompt: str, dim: int = 64) -> np.ndarray:
    x = np.zeros((dim,), dtype=np.float64)
    for i, ch in enumerate(prompt):
        x[(ord(ch) + 13 * i) % dim] += 1.0
    norm = np.linalg.norm(x) + 1e-9
    return x / norm


def sample_prompts(n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(PROMPTS), size=n)
    return [PROMPTS[i] + f" #{j}" for j, i in enumerate(idx)]


def run_llm_experiment(out_dir: Path, seed: int = 123) -> pd.DataFrame:
    cfg = LLMVictimConfig(seed=seed, vocab_size=128)
    victim = LLMVictim(cfg)
    interfaces = ["argmax", "topk", "probs"]
    budgets = [20, 50, 100, 180]

    eval_prompts = sample_prompts(120, seed + 1)
    X_eval = np.vstack([prompt_features(p) for p in eval_prompts])
    victim_eval_probs = np.vstack([victim.probs(p) for p in eval_prompts])
    victim_eval_labels = victim_eval_probs.argmax(axis=1)

    rows = []
    for interface in interfaces:
        for budget in budgets:
            qs = sample_prompts(budget, seed + budget)
            X_q = np.vstack([prompt_features(p) for p in qs])
            responses = [victim.query(p, interface=interface, k=5, noise_std=0.005) for p in qs]
            student = extract_student(
                X_query=X_q,
                responses=responses,
                interface=interface,
                n_classes=cfg.vocab_size,
                seed=seed,
                epochs=180 if interface == "probs" else 120,
                lr=0.35,
            )
            student_probs = student.predict_proba(X_eval)
            student_labels = student_probs.argmax(axis=1)
            rows.append(
                {
                    "experiment": f"llm_{victim.backend}",
                    "interface": interface,
                    "budget": budget,
                    "top1_agreement": float((student_labels == victim_eval_labels).mean()),
                    "kl_divergence": kl_divergence(victim_eval_probs, student_probs),
                    "backend": victim.backend,
                }
            )

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "llm_results.csv", index=False)
    return df

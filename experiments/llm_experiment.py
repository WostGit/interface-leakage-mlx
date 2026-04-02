from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from attacks.extraction import ExtractionMetrics, evaluate_extraction
from models.common import make_rng
from models.llm_models import MLXQwenVictim, StudentLMHead


def _make_prompts(n: int) -> list[str]:
    base = [
        "The capital of France is",
        "Machine learning models can",
        "In one sentence explain gravity:",
        "Write a short poem about rain",
        "Python function to reverse a list:",
        "A healthy breakfast includes",
        "What is the derivative of x^2?",
        "The quick brown fox",
    ]
    out: list[str] = []
    for i in range(n):
        out.append(f"{base[i % len(base)]} #{i}")
    return out


def run_llm_experiment(seed: int = 13) -> pd.DataFrame:
    rng = make_rng(seed)
    budgets = [16, 32, 64, 96, 128]
    interfaces = ["argmax", "topk", "probs"]
    pool = _make_prompts(180)
    test = _make_prompts(64)

    victim = MLXQwenVictim(seed=seed)
    victim_test = victim.next_token_probs(test)
    vocab_size = victim_test.shape[1]

    rows: list[dict] = []
    for interface in interfaces:
        for budget in budgets:
            q = pool[:budget]
            resp = victim.query(q, interface=interface, topk=5, noise_std=0.01, rng=rng)
            student = StudentLMHead(feature_dim=64, vocab_size=vocab_size, seed=seed + budget)
            student.fit(q, resp.probs)
            pred = student.predict_probs(test)
            top1, kl = evaluate_extraction(victim_test, pred)
            m = ExtractionMetrics(
                budget=budget,
                interface=interface,
                top1_agreement=top1,
                kl_div=kl,
                seed=seed,
                domain="tiny_llm_next_token",
                backend=victim.mode,
            )
            rows.append(asdict(m))
    return pd.DataFrame(rows)

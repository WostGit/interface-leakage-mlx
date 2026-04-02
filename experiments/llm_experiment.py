"""Next-token imitation experiment using Qwen2-0.5B via MLX when available."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import random
from typing import Dict, List

from attacks.extractor import mean_kl, response_to_soft_targets
from models.interfaces import InterfaceConfig
from models.llm_victim import QwenMLXVictim, build_next_token_victim
from models.next_token_student import NextTokenStudent


Row = Dict[str, object]


def _sample_contexts(rng: random.Random, n: int, seq_len: int, vocab_size: int) -> List[List[int]]:
    return [[rng.randrange(vocab_size) for _ in range(seq_len)] for _ in range(n)]


def _argmax(row: List[float]) -> int:
    return max(range(len(row)), key=lambda i: row[i])


def run_llm_experiment(seed: int = 7, output_dir: str = "outputs") -> List[Row]:
    rng = random.Random(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prefer_real = os.getenv("USE_REAL_QWEN", "0") == "1"
    vocab_size = 192
    victim = build_next_token_victim(prefer_real_qwen=prefer_real, vocab_size=vocab_size, seed=seed)
    is_real_qwen = isinstance(victim, QwenMLXVictim)
    if is_real_qwen:
        vocab_size = victim.vocab_size

    train_pool = _sample_contexts(rng, n=1200, seq_len=8, vocab_size=vocab_size)
    test_contexts = _sample_contexts(rng, n=300, seq_len=8, vocab_size=vocab_size)
    victim_test_probs = victim.predict_proba(test_contexts)

    budgets = [40, 80, 160, 320]
    interfaces = [
        InterfaceConfig(mode="argmax"),
        InterfaceConfig(mode="topk", topk=5),
        InterfaceConfig(mode="probs", noise_std=0.0),
    ]

    rows: List[Row] = []
    for cfg in interfaces:
        for b in budgets:
            xq = train_pool[:b]
            resp = victim.query(xq, cfg, seed=seed)
            y_soft = response_to_soft_targets(resp, num_classes=vocab_size)

            student = NextTokenStudent(vocab_size=vocab_size, seed=seed)
            student.fit_soft_targets(xq, y_soft, epochs=90, lr=0.4)

            p_student = student.predict_proba(test_contexts)
            agreement = sum(1 for ps, pv in zip(p_student, victim_test_probs) if _argmax(ps) == _argmax(pv)) / len(test_contexts)
            kl = mean_kl(victim_test_probs, p_student)
            rows.append(
                {
                    "experiment": "qwen2_next_token" if is_real_qwen else "fallback_next_token",
                    "interface": f"{cfg.mode}_k{cfg.topk}_noise{cfg.noise_std}",
                    "budget": b,
                    "top1_agreement": agreement,
                    "kl_divergence": kl,
                    "backend": "mlx_qwen2_0.5b" if is_real_qwen else "cpu_fallback_bigram",
                }
            )

    with (out / "llm_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows

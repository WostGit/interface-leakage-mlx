from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from attacks.token_attacker import TokenExtractionAttacker
from experiments.metrics import mean_kl_divergence, top1_agreement
from models.interfaces import InterfaceConfig
from models.qwen_victim import MlxQwenVictim, SyntheticTokenVictim, make_token_contexts


def run_llm_experiment(output_dir: Path, seed: int = 13) -> pd.DataFrame:
    budgets: List[int] = [32, 64, 128, 256]

    qwen = MlxQwenVictim(seed=seed)
    if qwen.available:
        victim = qwen
        victim_kind = "qwen2_0.5b_mlx"
        vocab_size = int(getattr(qwen.tokenizer, "vocab_size", 151936))
    else:
        vocab_size = 128
        victim = SyntheticTokenVictim(vocab_size=vocab_size, seed=seed)
        victim_kind = f"synthetic_fallback ({qwen.reason})"

    train_contexts_all, eval_contexts = make_token_contexts(
        seed=seed,
        n=max(budgets),
        vocab_size=vocab_size,
        length=10,
    )
    victim_eval_probs = victim.true_probs(eval_contexts)

    interfaces = [
        InterfaceConfig(mode="argmax"),
        InterfaceConfig(mode="topk", top_k=5),
        InterfaceConfig(mode="full", noise_std=0.0),
    ]

    rows = []
    for interface in interfaces:
        for budget in budgets:
            attacker = TokenExtractionAttacker(vocab_size=vocab_size, seed=seed + budget)
            student = attacker.extract(victim, train_contexts_all[:budget], interface)
            student_probs = student.predict_probs(eval_contexts)
            rows.append(
                {
                    "experiment": "tiny_llm_next_token",
                    "victim_backend": victim_kind,
                    "interface": interface.mode,
                    "budget": budget,
                    "top1_agreement": top1_agreement(victim_eval_probs, student_probs),
                    "kl_divergence": mean_kl_divergence(victim_eval_probs, student_probs),
                }
            )

    df = pd.DataFrame(rows)
    out_path = output_dir / "llm_results.csv"
    df.to_csv(out_path, index=False)
    return df

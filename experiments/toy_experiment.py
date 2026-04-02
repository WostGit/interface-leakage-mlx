from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from attacks.extractor import ExtractionAttacker
from experiments.metrics import mean_kl_divergence, top1_agreement
from models.interfaces import InterfaceConfig
from models.linear_victim import LinearVictim


def run_toy_experiment(output_dir: Path, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    input_dim = 20
    num_classes = 6
    budgets: List[int] = [50, 100, 200, 500, 1000]

    victim = LinearVictim(input_dim=input_dim, num_classes=num_classes, seed=seed)
    x_pool = rng.normal(size=(max(budgets), input_dim))
    x_eval = rng.normal(size=(1000, input_dim))
    victim_eval_probs = victim.predict_probs(x_eval)

    interfaces = [
        InterfaceConfig(mode="argmax"),
        InterfaceConfig(mode="topk", top_k=3),
        InterfaceConfig(mode="full", noise_std=0.0),
        InterfaceConfig(mode="full", noise_std=0.02),
    ]

    rows = []
    for interface in interfaces:
        for budget in budgets:
            attacker = ExtractionAttacker(input_dim=input_dim, num_classes=num_classes, seed=seed + budget)
            student = attacker.extract(
                victim=victim,
                interface=interface,
                x_queries=x_pool[:budget],
                train_epochs=180,
                lr=0.35,
            )
            student_probs = student.predict_probs(x_eval)

            rows.append(
                {
                    "experiment": "toy_linear",
                    "interface": interface.mode if interface.mode != "full" else f"full_noise_{interface.noise_std}",
                    "budget": budget,
                    "top1_agreement": top1_agreement(victim_eval_probs, student_probs),
                    "kl_divergence": mean_kl_divergence(victim_eval_probs, student_probs),
                }
            )

    df = pd.DataFrame(rows)
    out_path = output_dir / "toy_results.csv"
    df.to_csv(out_path, index=False)
    return df

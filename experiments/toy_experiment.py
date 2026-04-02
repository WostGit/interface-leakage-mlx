"""Toy linear extraction experiment across interface richness and query budget."""

from __future__ import annotations

import csv
from pathlib import Path
import random
from typing import Dict, List

from attacks.extractor import evaluate_extraction, response_to_soft_targets
from models.interfaces import InterfaceConfig
from models.linear import LinearStudent, ToyLinearVictim


Row = Dict[str, object]


def _rand_matrix(rng: random.Random, n: int, d: int) -> List[List[float]]:
    return [[rng.gauss(0, 1.0) for _ in range(d)] for _ in range(n)]


def run_toy_experiment(seed: int = 7, output_dir: str = "outputs") -> List[Row]:
    rng = random.Random(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    input_dim, num_classes = 8, 5
    victim = ToyLinearVictim.random(input_dim=input_dim, num_classes=num_classes, seed=seed)

    x_pool = _rand_matrix(rng, 3000, input_dim)
    x_test = _rand_matrix(rng, 600, input_dim)
    victim_test_probs = victim.predict_proba(x_test)

    budgets = [50, 100, 200, 400, 800]
    interfaces = [
        InterfaceConfig(mode="argmax"),
        InterfaceConfig(mode="topk", topk=3),
        InterfaceConfig(mode="probs", noise_std=0.0),
        InterfaceConfig(mode="probs", noise_std=0.02),
    ]

    rows: List[Row] = []
    for cfg in interfaces:
        for b in budgets:
            xq = x_pool[:b]
            resp = victim.query(xq, cfg, seed=seed)
            y_soft = response_to_soft_targets(resp, num_classes=num_classes)
            student = LinearStudent(input_dim=input_dim, num_classes=num_classes, seed=seed)
            student.fit_soft_targets(xq, y_soft, epochs=130, lr=0.25)
            metrics = evaluate_extraction(student, x_test, victim_test_probs)
            rows.append(
                {
                    "experiment": "toy_linear",
                    "interface": f"{cfg.mode}_k{cfg.topk}_noise{cfg.noise_std}",
                    "budget": b,
                    "top1_agreement": metrics.agreement,
                    "kl_divergence": metrics.kl_div,
                    "backend": "stdlib",
                }
            )

    with (out / "toy_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows

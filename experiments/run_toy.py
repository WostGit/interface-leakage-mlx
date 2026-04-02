from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from attacks.extractor import extract_student, kl_divergence
from models.toy_linear import ToyLinearVictim, make_toy_dataset


def run_toy_experiment(out_dir: Path, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_features, n_classes = 20, 6
    victim = ToyLinearVictim(n_features=n_features, n_classes=n_classes, seed=seed)

    X_eval = make_toy_dataset(seed + 1, 600, n_features)
    victim_eval_probs = np.array([victim.probs(x) for x in X_eval])
    victim_eval_labels = victim_eval_probs.argmax(axis=1)

    interfaces = ["argmax", "topk", "probs"]
    budgets = [20, 50, 100, 200, 400]
    rows = []

    for interface in interfaces:
        for budget in budgets:
            X_q = make_toy_dataset(seed + budget, budget, n_features)
            responses = [victim.query(x, interface=interface, k=3, noise_std=0.01, rng=rng) for x in X_q]
            student = extract_student(
                X_query=X_q,
                responses=responses,
                interface=interface,
                n_classes=n_classes,
                seed=seed,
                epochs=140 if interface != "argmax" else 100,
                lr=0.2,
            )
            student_probs = student.predict_proba(X_eval)
            student_labels = student_probs.argmax(axis=1)
            rows.append(
                {
                    "experiment": "toy",
                    "interface": interface,
                    "budget": budget,
                    "top1_agreement": float((student_labels == victim_eval_labels).mean()),
                    "kl_divergence": kl_divergence(victim_eval_probs, student_probs),
                }
            )

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "toy_results.csv", index=False)
    return df

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from attacks.extraction import ExtractionMetrics, evaluate_extraction
from models.common import make_rng
from models.linear_models import StudentLinearModel, VictimLinearModel


def run_toy_experiment(seed: int = 7) -> pd.DataFrame:
    rng = make_rng(seed)
    d_in = 16
    n_classes = 6
    n_pool = 1500
    n_test = 400
    budgets = [40, 80, 160, 320, 640]
    interfaces = ["argmax", "topk", "probs"]

    w = rng.normal(0.0, 1.0, size=(d_in, n_classes))
    b = rng.normal(0.0, 0.3, size=(n_classes,))
    victim = VictimLinearModel(w=w, b=b, noise_std=0.02)

    x_pool = rng.normal(size=(n_pool, d_in))
    x_test = rng.normal(size=(n_test, d_in))
    victim_test = victim.predict_probs(x_test)

    rows: list[dict] = []
    for interface in interfaces:
        for budget in budgets:
            xq = x_pool[:budget]
            resp = victim.query(xq, interface=interface, topk=3, rng=rng)
            student = StudentLinearModel(d_in=d_in, n_classes=n_classes, seed=seed + budget)
            student.fit(xq, resp.probs)
            pred = student.predict_probs(x_test)
            top1, kl = evaluate_extraction(victim_test, pred)
            m = ExtractionMetrics(
                budget=budget,
                interface=interface,
                top1_agreement=top1,
                kl_div=kl,
                seed=seed,
                domain="toy_linear",
                backend="numpy",
            )
            rows.append(asdict(m))
    return pd.DataFrame(rows)

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _plot_metric(df: pd.DataFrame, metric: str, out_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 5))
    for interface, g in df.groupby("interface"):
        g = g.sort_values("budget")
        plt.plot(g["budget"], g[metric], marker="o", label=interface)
    plt.xlabel("Query budget")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def generate_plots(results_dir: Path) -> None:
    toy = pd.read_csv(results_dir / "toy_results.csv")
    llm = pd.read_csv(results_dir / "llm_results.csv")

    _plot_metric(toy, "top1_agreement", results_dir / "toy_fidelity_vs_budget.png", "Toy: Fidelity vs Budget")
    _plot_metric(toy, "kl_divergence", results_dir / "toy_kl_vs_budget.png", "Toy: KL vs Budget")
    _plot_metric(llm, "top1_agreement", results_dir / "llm_fidelity_vs_budget.png", "LLM: Fidelity vs Budget")
    _plot_metric(llm, "kl_divergence", results_dir / "llm_kl_vs_budget.png", "LLM: KL vs Budget")

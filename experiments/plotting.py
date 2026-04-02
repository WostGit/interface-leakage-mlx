from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _plot_metric(df: pd.DataFrame, metric: str, out_file: Path, title: str) -> None:
    plt.figure(figsize=(7, 4.5))
    for label, sub in df.groupby("interface"):
        sub = sub.sort_values("budget")
        plt.plot(sub["budget"], sub[metric], marker="o", label=label)
    plt.xlabel("Query budget")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=140)
    plt.close()


def make_plots(toy_df: pd.DataFrame, llm_df: pd.DataFrame, output_dir: Path) -> None:
    _plot_metric(
        toy_df,
        metric="top1_agreement",
        out_file=output_dir / "toy_fidelity_vs_budget.png",
        title="Toy linear extraction: fidelity vs budget",
    )
    _plot_metric(
        toy_df,
        metric="kl_divergence",
        out_file=output_dir / "toy_kl_vs_budget.png",
        title="Toy linear extraction: KL vs budget",
    )
    _plot_metric(
        llm_df,
        metric="top1_agreement",
        out_file=output_dir / "llm_fidelity_vs_budget.png",
        title="LLM next-token extraction: fidelity vs budget",
    )
    _plot_metric(
        llm_df,
        metric="kl_divergence",
        out_file=output_dir / "llm_kl_vs_budget.png",
        title="LLM next-token extraction: KL vs budget",
    )

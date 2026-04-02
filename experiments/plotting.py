from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric(df: pd.DataFrame, metric: str, out_path: Path, title: str) -> None:
    plt.figure(figsize=(7, 4.5))
    for iface, g in df.groupby("interface"):
        g2 = g.sort_values("budget")
        plt.plot(g2["budget"], g2[metric], marker="o", label=iface)
    plt.xlabel("Query budget")
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()

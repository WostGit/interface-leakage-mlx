from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.llm_experiment import run_llm_experiment
from experiments.plotting import plot_metric
from experiments.toy_experiment import run_toy_experiment


def main() -> None:
    out = Path("results")
    out.mkdir(exist_ok=True)

    toy_df = run_toy_experiment(seed=7)
    llm_df = run_llm_experiment(seed=13)
    all_df = pd.concat([toy_df, llm_df], ignore_index=True)

    toy_csv = out / "toy_results.csv"
    llm_csv = out / "llm_results.csv"
    all_csv = out / "all_results.csv"

    toy_df.to_csv(toy_csv, index=False)
    llm_df.to_csv(llm_csv, index=False)
    all_df.to_csv(all_csv, index=False)

    for domain, g in all_df.groupby("domain"):
        plot_metric(
            g,
            metric="top1_agreement",
            out_path=out / f"{domain}_fidelity_vs_budget.png",
            title=f"{domain}: top-1 agreement vs budget",
        )
        plot_metric(
            g,
            metric="kl_div",
            out_path=out / f"{domain}_kl_vs_budget.png",
            title=f"{domain}: KL divergence vs budget",
        )

    summary = all_df.groupby(["domain", "interface"], as_index=False).agg(
        top1_agreement=("top1_agreement", "mean"),
        kl_div=("kl_div", "mean"),
    )
    summary.to_csv(out / "summary.csv", index=False)
    print("Wrote experiment artifacts to results/")
    print(summary)


if __name__ == "__main__":
    main()

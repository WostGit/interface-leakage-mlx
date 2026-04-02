from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.plot_results import generate_plots
from experiments.run_llm import run_llm_experiment
from experiments.run_toy import run_toy_experiment


def main() -> None:
    np.random.seed(0)
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    toy_df = run_toy_experiment(out_dir)
    llm_df = run_llm_experiment(out_dir)
    generate_plots(out_dir)

    all_df = toy_df.copy()
    all_df = all_df.assign(source="toy")
    llm_df2 = llm_df.copy().assign(source="llm")
    all_df = all_df.reindex(columns=sorted(set(all_df.columns) | set(llm_df2.columns))).fillna("")
    llm_df2 = llm_df2.reindex(columns=all_df.columns).fillna("")
    all_df = all_df._append(llm_df2, ignore_index=True)
    all_df.to_csv(out_dir / "all_results.csv", index=False)

    print("Wrote results to", out_dir.resolve())
    print(all_df.groupby(["source", "interface"])[["top1_agreement", "kl_divergence"]].mean())


if __name__ == "__main__":
    main()

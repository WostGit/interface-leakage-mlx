from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.llm_experiment import run_llm_experiment
from experiments.plotting import make_plots
from experiments.toy_experiment import run_toy_experiment


def main() -> None:
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    toy_df = run_toy_experiment(out_dir, seed=7)
    llm_df = run_llm_experiment(out_dir, seed=13)

    combined = pd.concat([toy_df, llm_df], ignore_index=True)
    combined.to_csv(out_dir / "all_results.csv", index=False)

    make_plots(toy_df=toy_df, llm_df=llm_df, output_dir=out_dir)

    print("Saved results to artifacts/")
    print(combined.groupby(["experiment", "interface"])[["top1_agreement", "kl_divergence"]].mean())


if __name__ == "__main__":
    main()

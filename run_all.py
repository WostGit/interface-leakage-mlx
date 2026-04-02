"""Entrypoint to run all extraction experiments and produce artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

from experiments.llm_experiment import run_llm_experiment
from experiments.plotting import plot_metrics
from experiments.toy_experiment import run_toy_experiment


def _write_csv(rows, path: Path):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    toy_rows = run_toy_experiment(seed=7, output_dir=str(output_dir))
    llm_rows = run_llm_experiment(seed=7, output_dir=str(output_dir))
    all_rows = toy_rows + llm_rows
    _write_csv(all_rows, output_dir / "all_results.csv")

    plot_metrics(toy_rows, "Toy linear", output_dir / "toy")
    plot_metrics(llm_rows, "LLM next-token", output_dir / "llm")

    print(f"Wrote {len(all_rows)} result rows to {output_dir}/all_results.csv")


if __name__ == "__main__":
    main()

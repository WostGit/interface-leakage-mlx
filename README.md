# interface-leakage-mlx

Minimal reproducible ML extraction experiments showing how richer prediction interfaces (argmax, top-k, full probabilities) reduce extraction difficulty.

## What is included

- **Toy linear model extraction** with controlled interfaces and query budgets.
- **Tiny LLM-style next-token extraction** using **Qwen2-0.5B through MLX** when available, with a deterministic CPU fallback for CI/non-Apple runners.
- Metrics: **Top-1 agreement** and **KL divergence**.
- Outputs: CSVs + plots.
- CI: GitHub Actions runs all experiments and uploads artifacts.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

Outputs are written to `results/`.

## Repository structure

- `models/` victim models and API interfaces
- `attacks/` student models and extraction logic
- `experiments/` experiment drivers
- `ci/` CI helper script
- `run_all.py` single entrypoint

## Notes

- Deterministic seeds are used throughout.
- CI sets `FORCE_CPU=1` to ensure the fallback path runs on Linux and remains under ~10 minutes.
- On Apple Silicon with MLX installed, the LLM experiment attempts to load `Qwen/Qwen2-0.5B-Instruct` via `mlx-lm`.

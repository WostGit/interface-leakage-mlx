# Interface Leakage with MLX: Minimal Model Extraction Repro

This repo provides a compact, reproducible set of experiments showing how API interface richness affects model extraction.

## What it includes

- **Toy linear classifier extraction** with three victim interfaces:
  - argmax-only labels
  - top-k probabilities
  - full probabilities (with optional noise)
- **Tiny LLM next-token imitation experiment**:
  - Uses **Qwen2-0.5B via MLX** when available on Apple Silicon.
  - Uses a deterministic **CPU fallback tiny LM** in CI/non-MLX environments.
- Query-budget sweeps reporting:
  - Top-1 agreement (student vs victim)
  - KL divergence (victim probs vs student probs)
- Outputs:
  - CSV results
  - Plots (fidelity vs budget, KL vs budget)
- GitHub Actions workflow that runs all experiments and uploads artifacts.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python run_all.py
```

Artifacts are written to `results/`.

## Reproducibility

All experiments use fixed seeds and deterministic NumPy operations.

## Notes on MLX + Qwen

The script attempts to load `Qwen/Qwen2-0.5B-Instruct` with `mlx-lm`.
If MLX/model loading is unavailable, it automatically falls back to a deterministic CPU tiny-LM so CI remains runnable.

# Interface Leakage with MLX: Minimal Reproducible Model Extraction

This repository provides a compact, reproducible ML extraction benchmark showing that **richer output interfaces reduce extraction difficulty**.

## Experiments

### 1) Toy linear victim
A fixed linear classifier is exposed through three interfaces:
- `argmax` (label-only)
- `topk` (top-k probabilities)
- `full` (full probability vector, optional Gaussian noise)

An attacker queries the victim under a budget and trains a student linear model to imitate it.

### 2) Tiny LLM-style next-token victim
- Preferred path (Apple Silicon): `Qwen2-0.5B` via `mlx-lm`.
- CI/Linux fallback: deterministic synthetic token-distribution victim with the same interface types.

The attacker learns a next-token student model under the same interfaces and budgets.

## Metrics
- Top-1 agreement
- KL divergence

Metrics are measured versus query budget and written as CSV plus plots.

## Project structure

- `models/` victim models + interface wrappers
- `attacks/` extraction attacker + student models
- `experiments/` experiment drivers, metrics, plotting
- `ci/` CI runner scripts
- `run_all.py` one-shot entrypoint

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

Outputs are generated under `artifacts/`:
- `toy_results.csv`
- `llm_results.csv`
- `all_results.csv`
- `*_fidelity_vs_budget.png`
- `*_kl_vs_budget.png`

## MLX/Qwen notes

If you want the real Qwen path, install MLX packages on Apple Silicon and optionally set up a Qwen2 0.5B MLX checkpoint (e.g., mlx-community variants). If unavailable, experiments still run in CPU CI through the fallback victim.

## Determinism

All experiments use fixed seeds.

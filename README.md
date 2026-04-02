# interface-leakage-mlx

Minimal reproducible ML extraction experiments showing that richer API outputs (argmax vs top-k vs probabilities) reduce extraction difficulty.

## What this repo demonstrates

1. **Victim interfaces**
   - argmax-only labels
   - top-k probabilities
   - full probabilities (optionally noisy)
2. **Attacker extraction**
   - query victim under budget constraints
   - train student models to imitate victim outputs
3. **Measured extraction quality**
   - top-1 agreement
   - KL divergence
4. **Two experiment tracks**
   - toy linear classifier
   - next-token imitation with an optional `Qwen/Qwen2-0.5B` MLX backend and deterministic CPU fallback for CI

## Project layout

- `models/` – victim/student model code and interface adapters
- `attacks/` – extraction target conversion and evaluation metrics
- `experiments/` – experiment runners and plotting helpers
- `ci/` – CI execution script
- `run_all.py` – single command entrypoint

## Determinism and CI runtime

- All experiments use fixed seeds.
- GitHub Actions runs CPU-only and avoids heavyweight dependencies.
- Runtime is designed to stay below ~10 minutes.

## Run locally

```bash
python run_all.py
```

Generated artifacts in `outputs/`:

- `toy_results.csv`
- `llm_results.csv`
- `all_results.csv`
- `toy_fidelity.svg`, `toy_kl.svg`
- `llm_fidelity.svg`, `llm_kl.svg`

## Optional: try real Qwen2-0.5B via MLX (Apple Silicon)

Set this env variable before running:

```bash
USE_REAL_QWEN=1 python run_all.py
```

If MLX/Qwen loading fails, the code falls back to a deterministic bigram victim so CI and Linux hosts remain reproducible.

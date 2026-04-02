#!/usr/bin/env bash
set -euo pipefail

python run_all.py

python - <<'PY'
import csv

with open('outputs/all_results.csv', newline='') as f:
    rows = list(csv.DictReader(f))

assert rows, 'results CSV is empty'
required = {'budget', 'top1_agreement', 'kl_divergence', 'interface', 'experiment'}
assert required.issubset(rows[0].keys()), rows[0].keys()
print('rows:', len(rows))
PY

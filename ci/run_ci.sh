#!/usr/bin/env bash
set -euo pipefail

python run_all.py

echo "Artifacts generated:"
ls -1 artifacts

#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-standard}"
CONFIG="configs/${MODE}.yaml"

if [[ ! -f "$CONFIG" ]]; then
  echo "Unknown mode '$MODE'. Expected quick, standard, or full." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ is required")
PY

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
pytest -q
python -u run_experiments.py --config "$CONFIG"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${ACTIVETOUCHGS_EVALUATION_CONFIG:-$REPO_ROOT/configs/evaluation_paths.env}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing evaluation config: $CONFIG" >&2
  echo "Copy configs/evaluation_paths.example.env and fill in its paths." >&2
  exit 2
fi

set -a
source "$CONFIG"
set +a

"${PYTHON_BIN}" \
  "$REPO_ROOT/src/evaluation/final/evaluate_fixed_stage1_to_stage2_factorial.py" \
  --selection-csv "${STAGE1_SELECTION_CSV}" \
  --stage2-oof-root "${STAGE2_OOF_ROOT}" \
  --out-dir "${FINAL_CASCADE_OUT}" \
  --threshold 0.5

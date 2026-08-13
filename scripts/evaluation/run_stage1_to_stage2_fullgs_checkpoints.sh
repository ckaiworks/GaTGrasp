#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${ACTIVETOUCHGS_CASCADE_CONFIG:-$REPO_ROOT/configs/cascade_checkpoint_paths.env}"

if [[ ! -f "$CONFIG" ]]; then
  echo "MISSING_CASCADE_CONFIG=$CONFIG" >&2
  echo "Copy configs/cascade_checkpoint_paths.example.env and fill in its paths." >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$CONFIG"

: "${PYTHON_BIN:?missing PYTHON_BIN}"
: "${CUDA_DEVICE:?missing CUDA_DEVICE}"
: "${STAGE1_DATA_ROOT:?missing STAGE1_DATA_ROOT}"
: "${STAGE2_DATA_ROOT:?missing STAGE2_DATA_ROOT}"
: "${STAGE1_RUN_ROOT:?missing STAGE1_RUN_ROOT}"
: "${STAGE2_FULLGS_RUN_ROOT:?missing STAGE2_FULLGS_RUN_ROOT}"
: "${CASCADE_CHECKPOINT_OUT:?missing CASCADE_CHECKPOINT_OUT}"

EVALUATOR="$REPO_ROOT/src/evaluation/final/run_cascade.py"
REFERENCE_SELECTION="$REPO_ROOT/reports/final_stage1_selection/selected_stage1_current_repro17of19_withGS_top1.csv"

for path in "$EVALUATOR" "$REFERENCE_SELECTION"; do
  [[ -f "$path" ]] || { echo "MISSING=$path" >&2; exit 3; }
done
[[ ! -e "$CASCADE_CHECKPOINT_OUT" ]] || {
  echo "REFUSE_OUTPUT_EXISTS=$CASCADE_CHECKPOINT_OUT" >&2
  exit 4
}

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

REFERENCE_ARGS=()
if [[ "${CASCADE_REQUIRE_REFERENCE_PROBABILITIES:-1}" == "1" ]]; then
  REFERENCE_ARGS+=(--require-reference-probabilities)
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" "$EVALUATOR" \
  --stage1-data-root "$STAGE1_DATA_ROOT" \
  --stage1-run-root "$STAGE1_RUN_ROOT" \
  --stage2-data-root "$STAGE2_DATA_ROOT" \
  --fullgs-run-root "$STAGE2_FULLGS_RUN_ROOT" \
  --out-dir "$CASCADE_CHECKPOINT_OUT" \
  --stage2-variants fullgs \
  --device cuda \
  --batch-size "${CASCADE_BATCH_SIZE:-32}" \
  --num-workers "${CASCADE_NUM_WORKERS:-0}" \
  --image-size 224 \
  --threshold 0.5 \
  --expected-checkpoint-epoch 50 \
  --reference-selection-csv "$REFERENCE_SELECTION" \
  "${REFERENCE_ARGS[@]}"

echo "STAGE1_TO_STAGE2_FULLGS_CHECKPOINT_STATUS=0"
echo "OUT=$CASCADE_CHECKPOINT_OUT"

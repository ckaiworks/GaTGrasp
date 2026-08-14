#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${ACTIVETOUCHGS_TRAINING_CONFIG:-$PROJECT_ROOT/configs/training_paths.env}"

if [[ ! -f "$CONFIG" ]]; then
  echo "MISSING_TRAINING_CONFIG=$CONFIG"
  exit 2
fi
# shellcheck disable=SC1090
source "$CONFIG"

: "${PYTHON_BIN:?missing PYTHON_BIN}"
: "${CUDA_DEVICE:?missing CUDA_DEVICE}"
: "${DATASET_ROOT:?missing DATASET_ROOT}"
: "${STAGE2_RUN_ROOT:?missing STAGE2_RUN_ROOT}"

TRAINER="$PROJECT_ROOT/src/stage2/final/train_stage2_fullgs_touch.py"
DATA_ROOT="$DATASET_ROOT/metadata/stage2"
MAX_PARALLEL="${STAGE2_MAX_PARALLEL:-5}"

[[ -f "$DATASET_ROOT/metadata/dataset_contract.json" ]] || {
  echo "INVALID_DATASET_ROOT=$DATASET_ROOT"
  exit 5
}

if (( MAX_PARALLEL < 1 || MAX_PARALLEL > 5 )); then
  echo "INVALID_STAGE2_MAX_PARALLEL=$MAX_PARALLEL (expected 1..5)"
  exit 3
fi

[[ -f "$TRAINER" ]] || { echo "MISSING=$TRAINER"; exit 3; }
if [[ -e "$STAGE2_RUN_ROOT" ]]; then
  echo "REFUSE_RUN_ROOT_EXISTS=$STAGE2_RUN_ROOT"
  exit 4
fi
for fold in 0 1 2 3 4; do
  for split in train test; do
    path="$DATA_ROOT/folds/fold_${fold}/${split}.csv"
    [[ -f "$path" ]] || { echo "MISSING=$path"; exit 5; }
  done
done

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "$STAGE2_RUN_ROOT/logs" "$STAGE2_RUN_ROOT/status"
echo "STAGE2_VARIANT=exact Stage-1 old12 L/R/C GS backbone + bilateral Touch"
echo "STAGE2_FOLDS_TOTAL=5"
echo "STAGE2_MAX_PARALLEL=$MAX_PARALLEL"

run_one() {
  local fold="$1"
  local out="$STAGE2_RUN_ROOT/fold_${fold}"
  local log="$STAGE2_RUN_ROOT/logs/fold_${fold}.log"
  local status_file="$STAGE2_RUN_ROOT/status/fold_${fold}.status"

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" "$TRAINER" \
    --train_csv "$DATA_ROOT/folds/fold_${fold}/train.csv" \
    --test_csv "$DATA_ROOT/folds/fold_${fold}/test.csv" \
    --run_dir "$out" \
    --epochs 50 \
    --fixed_epoch 50 \
    --batch_size 32 \
    --lr 0.0003 \
    --weight_decay 0.0001 \
    --num_workers 0 \
    --seed 42 \
    --image_size 224 \
    >"$log" 2>&1
  local rc=$?
  echo "$rc" >"$status_file"
  return "$rc"
}

status=0
pids=()
for fold in 0 1 2 3 4; do
  run_one "$fold" &
  pids+=("$!")
  if [[ "${#pids[@]}" -ge "$MAX_PARALLEL" ]]; then
    wait "${pids[0]}" || status=1
    pids=("${pids[@]:1}")
  fi
done
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done

for fold in 0 1 2 3 4; do
  [[ "$(cat "$STAGE2_RUN_ROOT/status/fold_${fold}.status")" == "0" ]] || status=1
  [[ -f "$STAGE2_RUN_ROOT/fold_${fold}/checkpoint_epoch_050.pt" ]] || status=1
  [[ -f "$STAGE2_RUN_ROOT/fold_${fold}/test_metrics.json" ]] || status=1
done

echo "STAGE2_SELECTED_STAGE1BACKBONE_OLD12_TOUCH_STATUS=$status"
echo "RUN_ROOT=$STAGE2_RUN_ROOT"
exit "$status"

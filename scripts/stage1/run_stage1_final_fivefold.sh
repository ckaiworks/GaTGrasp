#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${ACTIVETOUCHGS_TRAINING_CONFIG:-$PROJECT_ROOT/configs/training_paths.env}"

if [[ ! -f "$CONFIG" ]]; then
  echo "MISSING_TRAINING_CONFIG=$CONFIG"
  exit 2
fi
source "$CONFIG"

: "${PYTHON_BIN:?missing PYTHON_BIN}"
: "${CUDA_DEVICE:?missing CUDA_DEVICE}"
: "${DATASET_ROOT:?missing DATASET_ROOT}"
: "${STAGE1_RUN_ROOT:?missing STAGE1_RUN_ROOT}"

STAGE1_DATA_ROOT="$DATASET_ROOT/metadata/stage1"
[[ -f "$DATASET_ROOT/metadata/dataset_contract.json" ]] || {
  echo "INVALID_DATASET_ROOT=$DATASET_ROOT"
  exit 5
}

CODE="$PROJECT_ROOT/src/stage1/final"
ENTRY="$CODE/train_stage1_old12_fullcorridor_annealedSelect_v1.py"
WRAPPER="$CODE/deterministic_run.py"

if [[ -e "$STAGE1_RUN_ROOT" ]]; then
  echo "REFUSE_RUN_ROOT_EXISTS=$STAGE1_RUN_ROOT"
  exit 3
fi
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "MISSING_PYTHON_COMMAND=$PYTHON_BIN"
  echo "Activate the atgs environment first: conda activate atgs"
  exit 4
}
for path in "$ENTRY" "$WRAPPER"; do
  [[ -e "$path" ]] || { echo "MISSING=$path"; exit 4; }
done
for fold in 0 1 2 3 4; do
  for split in train test; do
    path="$STAGE1_DATA_ROOT/folds/fold_${fold}/${split}.csv"
    [[ -f "$path" ]] || { echo "MISSING=$path"; exit 5; }
  done
done

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$STAGE1_RUN_ROOT/logs" "$STAGE1_RUN_ROOT/status"

run_fold() {
  local fold="$1"
  local out="$STAGE1_RUN_ROOT/fold_${fold}"
  local log="$STAGE1_RUN_ROOT/logs/fold_${fold}.log"
  local status="$STAGE1_RUN_ROOT/status/fold_${fold}.status"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" "$WRAPPER" \
    --entry "$ENTRY" \
    --deterministic_seed 42 \
    --train_csv "$STAGE1_DATA_ROOT/folds/fold_${fold}/train.csv" \
    --test_csv "$STAGE1_DATA_ROOT/folds/fold_${fold}/test.csv" \
    --run_dir "$out" \
    --epochs 50 \
    --fixed_epoch 50 \
    --batch_size 32 \
    --lr 0.0003 \
    --weight_decay 0.0001 \
    --num_workers 4 \
    --seed 42 \
    --image_size 224 \
    --soft_top1_weight 1.0 \
    --select_temperature_start 1.0 \
    --select_temperature_end 0.1 \
    --soft_top1_warmup_epochs 5 \
    >"$log" 2>&1
  local rc=$?
  echo "$rc" >"$status"
  return "$rc"
}

pids=()
for fold in 0 1 2 3 4; do
  run_fold "$fold" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done

for fold in 0 1 2 3 4; do
  [[ "$(cat "$STAGE1_RUN_ROOT/status/fold_${fold}.status")" == "0" ]] || status=1
  [[ -f "$STAGE1_RUN_ROOT/fold_${fold}/best_fullobject_softtop1.pt" ]] || status=1
  [[ -f "$STAGE1_RUN_ROOT/fold_${fold}/test_metrics.json" ]] || status=1
done

echo "STAGE1_FINAL_FIVEFOLD_STATUS=$status"
echo "RUN_ROOT=$STAGE1_RUN_ROOT"
exit "$status"

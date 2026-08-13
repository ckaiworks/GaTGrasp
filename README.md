# ActiveTouchGS

ActiveTouchGS is the compact training and inference release for the final
two-stage visuo-tactile grasp-stability model. This repository intentionally
contains **no raw reconstruction pipeline and no simulator runtime**. Prepared
RGB, old12 Gaussian features, tactile observations and labels are distributed
as one external dataset archive.

## What is included

- final reproducible Stage 1: RGB + Action13 + old12 left/right/corridor GS;
- selected Stage 2: the same visual/action/GS backbone plus bilateral Touch;
- five-fold concurrent launchers for both stages;
- standalone Stage-1 inference;
- exact Stage-1 -> Stage-2 checkpoint-level cascade inference;
- clean Stage-1 ablation sources and frozen paper-table results;
- portable dataset validation and runtime-index generation.

## External downloads

Download these two archives separately and replace the placeholder URLs after
uploading them to the selected data host:

| Archive | Purpose | SHA-256 |
|---|---|---|
| `ActiveTouchGS_dataset_v2.tar.gz` | Prepared RGB, old12 GS, bilateral Touch, labels and fixed folds | `8f8fc15492a37fa4ea392b6e119e593c9d58e0e6c8bd6b76717c538e96209a05` |
| `ActiveTouchGS_final_checkpoints_v1.tar.gz` | Selected Stage-1 and Stage-2 five-fold checkpoints/statistics | `0ac592b0cbb366338d85919bb8ca654c6d8a3ae9323a77a98ff9a631ee40621a` |

The dataset contains 19 objects and 2,850 actions (1,779 positive and 1,071
negative); 2,769 actions have tactile observations. Stage 1 and Stage 2 use
the same action identities and the same old12 GS features. Stage 1 ignores the
tactile-only columns.

Expected dataset contents:

- prepared RGB observations;
- Action13 candidate parameters and binary success labels;
- old12 left-contact, right-contact and closing-corridor GS features;
- bilateral after-close tactile observations;
- `raw_pair_id` identities and the fixed five-fold object-disjoint splits.

Expected checkpoint layout:

```text
stage1_corridor_withgs/fold_0 ... fold_4
stage2_fullgs_touch/fold_0 ... fold_4
```

Each Stage-1 fold contains `best_fullobject_softtop1.pt` and
`fullobject_feature_stats.npz`. Each Stage-2 fold contains
`checkpoint_epoch_050.pt` and `fullobject_feature_stats.npz`. Metrics and run
contracts are included with the checkpoint archive for identity checks.

## Environment

Only one Python environment is needed for prepared-data training and
inference. The environment name is arbitrary.

```bash
conda create -n activetouchgs python=3.9.25 pip -y
conda activate activetouchgs
pip install -r requirements.txt
python tests/verify_runtime_imports.py
```

PyBullet/TAXIM sampling, VGGT and 3DGS are not installed because their outputs
are already in the dataset archive.

## 1. Validate and materialize the dataset

```bash
tar -xzf ActiveTouchGS_dataset_v2.tar.gz -C /your_path

python scripts/dataset/validate_final_dataset.py \
  --dataset-root /your_path/ActiveTouchGS_dataset_v2 \
  --verify-sha256

python scripts/dataset/materialize_runtime_indexes.py \
  --dataset-root /your_path/ActiveTouchGS_dataset_v2 \
  --out /your_path/ActiveTouchGS_runtime_indexes
```

The generated `stage1/` and `stage2/` roots each directly contain
`folds/fold_0` through `folds/fold_4`.

## 2. Train Stage 1 and Stage 2

```bash
cp configs/training_paths.example.env configs/training_paths.env
# Replace every /your_path value in the private file.
export ACTIVETOUCHGS_TRAINING_CONFIG="$PWD/configs/training_paths.env"

bash scripts/stage1/run_stage1_final_fivefold.sh
bash scripts/stage2/run_stage2_selected_fullgs_fivefold.sh
```

Both launchers run all five folds concurrently. Scheduling does not change the
model, data, loss or per-fold hyperparameters.

## 3. Stage-1-only inference

```bash
python src/evaluation/final/infer_stage1_fivefold.py \
  --data-root /your_path/ActiveTouchGS_runtime_indexes/stage1 \
  --run-root /your_path/checkpoints/stage1_corridor_withgs \
  --out-dir /your_path/results/stage1 \
  --device cuda
```

This ranks all 150 candidates for every held-out object and reports S@1,
H@5, H@10 and MRR.

## 4. Connected Stage-1 -> Stage-2 inference

```bash
python src/evaluation/final/run_cascade.py \
  --stage1-data-root /your_path/ActiveTouchGS_runtime_indexes/stage1 \
  --stage1-run-root /your_path/checkpoints/stage1_corridor_withgs \
  --stage2-data-root /your_path/ActiveTouchGS_runtime_indexes/stage2 \
  --fullgs-run-root /your_path/checkpoints/stage2_fullgs_touch \
  --out-dir /your_path/results/cascade \
  --stage2-variants fullgs \
  --device cuda
```

Stage 1 selects exactly one action for each of the 19 objects. The connector
uses `raw_pair_id` to locate that exact same action in Stage 2; Stage 2 predicts
its success probability and never reranks the remaining candidates.

## Repository map

```text
configs/                     portable path templates
scripts/dataset/             dataset validation and runtime indexes
scripts/stage1/              final five-fold Stage-1 launcher
scripts/stage2/              final five-fold Stage-2 launcher
src/stage1/final/            selected Stage-1 implementation
src/stage2/final/            selected full-GS/corridor Stage-2 implementation
experiments/table2_*/        Table-2 selector methods and results
experiments/table3_*/        Table-3 Stage-2 controls and results
experiments/table4_*/        Table-4 Stage-1 ablations and results
src/evaluation/final/        Stage-1 and cascade inference
reports/                     frozen final paper-table results
tests/                       runtime and release checks
```

## Reproducibility boundary

This compact repository reproduces training and inference from the prepared
dataset. It does not claim to regenerate the dataset from raw object meshes.
Sampling/reconstruction source can be released separately after confirming
the redistribution terms of all ObjectFolder assets.

The repository deliberately excludes:

- raw PyBullet/TAXIM sampling and physical-data collection programs;
- VGGT inference and 3D Gaussian Splatting reconstruction/training trees;
- copied ObjectFolder and other third-party repositories;
- historical experiments, failed builds and machine-specific absolute paths;
- raw dataset binaries and trained checkpoint binaries.

## Third-party provenance and redistribution

The prepared dataset was produced with upstream systems including
ObjectFolder assets, PyBullet/TAXIM, VGGT and 3D Gaussian Splatting tooling.
This repository does not vendor those source trees. Public redistribution of
the prepared observations must comply with the corresponding upstream terms.
Archive SHA-256 values establish file identity; they do not grant a separate
redistribution license. The paper and public release should cite and link the
exact upstream repositories or commits used to create the prepared data.

## License

A project-level license has not yet been selected. Add an approved `LICENSE`
before making the repository public. Dataset and checkpoint redistribution
terms should be stated separately.

<div align="center">

# ActiveTouchGS

**Gaussian-guided active visuo-tactile grasp stability prediction**

[Installation](#installation) · [Data and checkpoints](#data-and-checkpoints) · [Quick start](#quick-start) · [Training](#training) · [Experiments](#reproducing-the-paper-tables)

</div>

ActiveTouchGS is a two-stage framework for robot grasp selection and tactile
verification. Given 150 candidate actions for an unseen object, Stage 1 ranks
the candidates before tactile sensing. After the selected action is executed,
Stage 2 combines visual, action, local 3D Gaussian and bilateral tactile
features to predict whether that grasp is stable.

This repository contains the final training, inference and ablation code. The
prepared dataset and selected five-fold checkpoints are distributed separately
because of their size.

## Highlights

- **Active selection before touch:** Stage 1 chooses one action from a
  150-action candidate pool without using tactile observations.
- **Action-conditioned 3D context:** local 3D Gaussian features describe the
  two contact regions and the closing corridor associated with each action.
- **Post-contact verification:** Stage 2 evaluates the selected action after
  receiving one bilateral tactile observation.
- **Object-disjoint evaluation:** all reported results use a fixed five-fold
  split over 19 objects, with no object shared between training and testing in
  a fold.
- **Reproducible release:** code, prepared data, fold definitions, final
  checkpoints and table-specific experiment files are provided.

## Installation

The released training and inference pipeline uses one Python environment. The
Conda environment name is arbitrary.

```bash
git clone https://github.com/964486029/ATGS.git
cd ATGS

conda create -n activetouchgs python=3.9.25 pip -y
conda activate activetouchgs
pip install -r requirements.txt
```

The prepared release does not require users to rerun PyBullet/TAXIM sampling,
VGGT inference or 3D Gaussian reconstruction.

## Data and checkpoints

Download the prepared dataset and final checkpoints:

- **File:** `ActiveTouchGS_external_files.zip`
- **Baidu Netdisk:** [download link](https://pan.baidu.com/s/1QukccPOYJeF56yExBHPq1Q?pwd=bsx8)
- **Extraction code:** `bsx8`
- **Size:** 2,045,441,684 bytes (approximately 1.91 GiB)
- **SHA-256:** `71196b84a1006e7fdd6ca209b0dc101749bfd2854b10dbadac72e8699a0873ac`

Extract the archive beside the Git repository:

```bash
unzip ActiveTouchGS_external_files.zip -d /your_path
```

The extracted files should have the following layout:

```text
ActiveTouchGS_external_files/
├── README.txt
├── ActiveTouchGS_dataset/
│   ├── data/
│   └── metadata/
└── ActiveTouchGS_final_checkpoints/
    ├── stage1_corridor_withgs/
    │   ├── fold_0/
    │   ├── ...
    │   └── fold_4/
    └── stage2_fullgs_touch/
        ├── fold_0/
        ├── ...
        └── fold_4/
```

Validate the extracted dataset before training or inference:

```bash
python scripts/dataset/validate_final_dataset.py \
  --dataset-root /your_path/ActiveTouchGS_external_files/ActiveTouchGS_dataset \
  --verify-sha256
```

Expected dataset summary:

| Item | Value |
|---|---:|
| Objects | 19 |
| Candidate actions per object | 150 |
| Total candidate actions | 2,850 |
| Positive / negative labels | 1,779 / 1,071 |
| Actions with tactile observations | 2,769 |
| Evaluation protocol | Five-fold object-disjoint |

## Quick start

### 1. Configure paths

```bash
cp configs/training_paths.example.env configs/training_paths.env
```

Replace the `/your_path` placeholders in `configs/training_paths.env`, then
load the configuration:

```bash
export ACTIVETOUCHGS_TRAINING_CONFIG="$PWD/configs/training_paths.env"
```

### 2. Run Stage-1 inference

This command ranks all 150 candidates for each held-out object and reports
S@1, H@5, H@10 and MRR.

```bash
python src/evaluation/final/infer_stage1_fivefold.py \
  --data-root /your_path/ActiveTouchGS_external_files/ActiveTouchGS_dataset/metadata/stage1 \
  --run-root /your_path/ActiveTouchGS_external_files/ActiveTouchGS_final_checkpoints/stage1_corridor_withgs \
  --out-dir /your_path/results/stage1 \
  --device cuda
```

### 3. Run connected Stage-1 to Stage-2 inference

Stage 1 first selects one action for each object. The connector then uses
`raw_pair_id` to locate that exact action in the Stage-2 data and predicts its
stability probability without changing the Stage-1 ranking.

```bash
python src/evaluation/final/run_cascade.py \
  --stage1-data-root /your_path/ActiveTouchGS_external_files/ActiveTouchGS_dataset/metadata/stage1 \
  --stage1-run-root /your_path/ActiveTouchGS_external_files/ActiveTouchGS_final_checkpoints/stage1_corridor_withgs \
  --stage2-data-root /your_path/ActiveTouchGS_external_files/ActiveTouchGS_dataset/metadata/stage2 \
  --fullgs-run-root /your_path/ActiveTouchGS_external_files/ActiveTouchGS_final_checkpoints/stage2_fullgs_touch \
  --out-dir /your_path/results/cascade \
  --stage2-variants fullgs \
  --device cuda
```

## Training

### Stage 1: active grasp selection

```bash
bash scripts/stage1/run_stage1_final_fivefold.sh
```

The selected Stage-1 model uses RGB, a 13-D action vector and local old12 3D
Gaussian features from the two contact regions and closing corridor. The
launcher trains all five object-disjoint folds.

### Stage 2: tactile verification

```bash
bash scripts/stage2/run_stage2_selected_fullgs_fivefold.sh
```

The selected Stage-2 model adds a shared bilateral ResNet18 tactile encoder to
the RGB, Action13 and local-GS representation. The launcher trains all five
folds.

## Reported results

The selected Stage-1 model achieves **17/19 successful Top-1 selections**
(S@1 89.47%), H@5 94.74%, H@10 100.00% and MRR 0.9175.

On the same 19 Stage-1-selected actions, the selected Stage-2 model obtains:

| Input | Accuracy | Macro-F1 | AUROC | AUPRC | ECE-5 ↓ |
|---|---:|---:|---:|---:|---:|
| RGB + Action + GS + Touch | **0.9474** | **0.8190** | **0.9118** | **0.9896** | **0.0499** |

Complete comparison and ablation tables are kept with their corresponding
implementations under `experiments/` rather than duplicated on the project
front page.

## Reproducing the paper tables

```text
experiments/
├── table2_primary_selector/
├── table3_tactile_verification/
└── table4_selector_ablations/
```

Each table directory contains:

- `results.csv`: the values reported in the corresponding paper table;
- `methods.csv`: the mapping from table rows to executable implementations;
- `methods/`: method-specific training or evaluation code;
- `README.md`: commands and notes for reproducing that table.

The non-neural Random, Geometry, Gaussian-Quality and Oracle baselines are
implemented together in
`experiments/table2_primary_selector/methods/non_neural_selectors.py`.

## Repository structure

```text
ATGS/
├── configs/       path configuration templates
├── experiments/   implementations and results for Tables 2–4
├── manifests/     release file inventory and checksums
├── reports/       frozen evaluation outputs
├── scripts/       dataset validation and training launchers
├── src/           final Stage-1, Stage-2 and inference code
├── README.md
└── requirements.txt
```

## Reproducibility scope

This release reproduces training, five-fold evaluation and connected inference
from the prepared dataset. Raw simulator/physical collection code, VGGT
inference, 3D Gaussian reconstruction repositories and obsolete development
experiments are not included in this compact release.

## Acknowledgements

The prepared data and pipeline build on ObjectFolder, PyBullet/TAXIM, VGGT and
3D Gaussian Splatting. Please follow the licenses and terms of the respective
upstream projects when using or redistributing derived assets.

## Citation

Citation metadata will be added when the final paper record is available.

<div align="center">

# GaTGrasp

### Gaussian-guided Active Visuo-Tactile Grasp Stability Prediction

[![Python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Data](https://img.shields.io/badge/Data-Baidu_Netdisk-06A7FF)](https://pan.baidu.com/s/1QukccPOYJeF56yExBHPq1Q?pwd=bsx8)
[![Paper](https://img.shields.io/badge/Paper-OpenReview-8C1B13)](https://openreview.net/forum?id=BZsC7zDWBO)

**[Paper](https://openreview.net/forum?id=BZsC7zDWBO) · [Installation](#-environment-preparation) · [Data](#-data-and-checkpoints) · [Evaluation](#-quick-evaluation) · [Training](#-model-training) · [Results](#-reported-results)**

</div>

> GaTGrasp is a two-stage framework for active robot grasp selection and post-contact tactile verification.

Given 150 candidate actions for an unseen object, **Stage 1** ranks the candidates before tactile sensing. After the selected action is executed, **Stage 2** combines visual, action, local 3D Gaussian, and bilateral tactile features to predict whether the grasp is stable.

This repository provides the final training, inference, and ablation code. The prepared dataset and selected five-fold checkpoints are distributed separately because of their size.

---

## ✨ Highlights

- **Active selection before touch:** Stage 1 selects one action from a 150-action candidate pool without tactile observations.
- **Action-conditioned 3D context:** local 3D Gaussian features describe the two contact regions and the closing corridor associated with each action.
- **Post-contact verification:** Stage 2 evaluates the selected action after receiving one bilateral tactile observation.
- **Object-disjoint evaluation:** all reported results use a fixed five-fold split over 19 objects, with no object shared between training and testing in a fold.
- **Reproducible release:** code, prepared data, fold definitions, final checkpoints, and table-specific experiment files are provided.

## 🧭 Framework Overview

| Stage | When | Inputs | Output |
|:---:|---|---|---|
| **Stage 1** | Before contact | RGB + Action13 + local 3D Gaussian features | Ranking of 150 candidate grasps |
| **Stage 2** | After contact | RGB + Action13 + local 3D Gaussian features + bilateral touch | Grasp-stability probability |

## ⚙️ Environment Preparation

The released training and inference pipeline uses **Python 3.9**, **PyTorch 2.7.0**, and **CUDA 12.8**. The Conda environment name is arbitrary.

```bash
git clone https://github.com/ckaiworks/GaTGrasp.git
cd GaTGrasp

conda create -n gatgrasp python=3.9.25 pip -y
conda activate gatgrasp
pip install -r requirements.txt
```

The prepared release does not require users to rerun PyBullet/TAXIM sampling, VGGT inference, or 3D Gaussian reconstruction.

## 📦 Data and Checkpoints

Download the prepared dataset and final checkpoints:

| Item | Details |
|---|---|
| **Archive** | `ActiveTouchGS_external_files.zip` |
| **Download** | [Baidu Netdisk](https://pan.baidu.com/s/1QukccPOYJeF56yExBHPq1Q?pwd=bsx8) |
| **Extraction code** | `bsx8` |
| **Size** | 2,045,441,684 bytes (approximately 1.91 GiB) |
| **SHA-256** | `71196b84a1006e7fdd6ca209b0dc101749bfd2854b10dbadac72e8699a0873ac` |

Extract the archive beside the Git repository:

```bash
unzip ActiveTouchGS_external_files.zip -d /your_path
```

The extracted files should have the following layout:

```text
ActiveTouchGS_external_files/
|-- README.txt
|-- ActiveTouchGS_dataset/
|   |-- data/
|   `-- metadata/
`-- ActiveTouchGS_final_checkpoints/
    |-- stage1_corridor_withgs/
    |   |-- fold_0/
    |   |-- ...
    |   `-- fold_4/
    `-- stage2_fullgs_touch/
        |-- fold_0/
        |-- ...
        `-- fold_4/
```

Validate the extracted dataset before training or inference:

```bash
python scripts/dataset/validate_final_dataset.py \
  --dataset-root /your_path/ActiveTouchGS_external_files/ActiveTouchGS_dataset \
  --verify-sha256
```

### Dataset summary

| Item | Value |
|---|---:|
| Objects | 19 |
| Candidate actions per object | 150 |
| Total candidate actions | 2,850 |
| Positive / negative labels | 1,779 / 1,071 |
| Actions with tactile observations | 2,769 |
| Evaluation protocol | Five-fold object-disjoint |

## 🚀 Quick Evaluation

### 1. Configure paths

```bash
cp configs/training_paths.example.env configs/training_paths.env
```

Replace the `/your_path` placeholders in `configs/training_paths.env`, then load the configuration:

```bash
export ACTIVETOUCHGS_TRAINING_CONFIG="$PWD/configs/training_paths.env"
```

### 2. Run Stage-1 inference

This command ranks all 150 candidates for each held-out object and reports S@1, H@5, H@10, and MRR.

```bash
python src/evaluation/final/infer_stage1_fivefold.py \
  --data-root /your_path/ActiveTouchGS_external_files/ActiveTouchGS_dataset/metadata/stage1 \
  --run-root /your_path/ActiveTouchGS_external_files/ActiveTouchGS_final_checkpoints/stage1_corridor_withgs \
  --out-dir /your_path/results/stage1 \
  --device cuda
```

### 3. Run connected Stage-1-to-Stage-2 inference

Stage 1 first selects one action for each object. The connector then uses `raw_pair_id` to locate that exact action in the Stage-2 data and predicts its stability probability without changing the Stage-1 ranking.

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

## 🏋️ Model Training

### Stage 1: active grasp selection

```bash
bash scripts/stage1/run_stage1_final_fivefold.sh
```

The selected Stage-1 model uses RGB, a 13-D action vector, and local old12 3D Gaussian features from the two contact regions and closing corridor. The launcher trains all five object-disjoint folds.

### Stage 2: tactile verification

```bash
bash scripts/stage2/run_stage2_selected_fullgs_fivefold.sh
```

The selected Stage-2 model adds a shared bilateral ResNet18 tactile encoder to the RGB, Action13, and local-GS representation. The launcher trains all five folds.

## 📊 Reported Results

### Stage 1: active selection

| S@1 | H@5 | H@10 | MRR |
|---:|---:|---:|---:|
| **89.47% (17/19)** | **94.74% (18/19)** | **100.00% (19/19)** | **0.9175** |

### Stage 2: tactile verification

The following result is measured on the same 19 Stage-1-selected actions.

| Input | Accuracy | Macro-F1 | AUROC | AUPRC | ECE-5 ↓ |
|---|---:|---:|---:|---:|---:|
| RGB + Action + GS + Touch | **0.9474** | **0.8190** | **0.9118** | **0.9896** | **0.0499** |

Complete comparisons and ablations are stored with their corresponding implementations under `experiments/`.

## 🧪 Reproducing the Paper Tables

```text
experiments/
|-- table2_primary_selector/
|-- table3_tactile_verification/
`-- table4_selector_ablations/
```

Each table directory contains:

- `results.csv`: values reported in the corresponding paper table;
- `methods.csv`: mapping from table rows to executable implementations;
- `methods/`: method-specific training or evaluation code;
- `README.md`: commands and notes for reproducing that table.

The non-neural Random, Geometry, Gaussian-Quality, and Oracle baselines are implemented together in `experiments/table2_primary_selector/methods/non_neural_selectors.py`.

## 🗂️ Repository Structure

```text
GaTGrasp/
|-- configs/       # Path configuration templates
|-- experiments/   # Implementations and results for Tables 2-4
|-- manifests/     # Release file inventory and checksums
|-- reports/       # Frozen evaluation outputs
|-- scripts/       # Dataset validation and training launchers
|-- src/           # Stage-1, Stage-2, and inference code
|-- README.md
`-- requirements.txt
```

## 🔁 Reproducibility Scope

This release reproduces training, five-fold evaluation, and connected inference from the prepared dataset. Raw simulator/physical collection code, VGGT inference, 3D Gaussian reconstruction repositories, and obsolete development experiments are not included in this compact release.

## 🙏 Acknowledgements

The prepared data and pipeline build on ObjectFolder, PyBullet/TAXIM, VGGT, and 3D Gaussian Splatting. Please follow the licenses and terms of the respective upstream projects when using or redistributing derived assets.

## 📜 Citation

The paper is available on [OpenReview](https://openreview.net/forum?id=BZsC7zDWBO).
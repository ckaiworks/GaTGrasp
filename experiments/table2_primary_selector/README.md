# Table 2: Primary selector comparison

All learned methods use the same object-disjoint five folds, the same 150
candidate actions per object, seed 42, epoch 50, weighted BCE plus selection
loss with lambda 1, cosine temperature 1.0 to 0.1, and five warm-up epochs.

## Row-to-code mapping

| Paper row | Runnable source | Notes |
|---|---|---|
| Random Selector | `methods/non_neural_selectors.py --method random` | Exact expectation over a uniform random permutation; no training. |
| Geometry Selector | `methods/non_neural_selectors.py --method geometry` | Equal mean of the three candidate-pool geometry cosines; no training. |
| Gaussian-Quality Selector | `methods/non_neural_selectors.py --method gaussian_quality` | Frozen Gaussian-quality heuristic; no training. |
| Action Only | `methods/action_only/train_stage1_actionOnly_annealedSelect_v1.py` | Action13 only; RGB and GS are structurally absent. |
| RGB+Action | `methods/rgb_action/train_stage1_clean_structural_nogs_annealedSelect_v1.py` | RGB and Action13; all GS branches are structurally absent. |
| RGB+Action+Depth | `methods/rgb_action_depth/train_stage1_RGBActionDepth_currentAnnealedSelect_v2.py` | Fixed-view depth baseline. |
| RGB+Action+Point Cloud | `methods/rgb_action_pointcloud/train_stage1_RGBActionPointCloud_currentAnnealedSelect_v1.py` | Fixed-view PointNet baseline. |
| Proposed Selector | `../../src/stage1/final/train_stage1_old12_fullcorridor_annealedSelect_v1.py` | Selected RGB+Action13+left/right/corridor GS model. |
| Oracle Ranking | `methods/non_neural_selectors.py --method oracle` | Labels are used only to compute the upper bound; no training. |

`methods/non_neural_selectors.py` reproduces Random, Geometry and Oracle from
the released candidate table.  The historical Gaussian-quality result is
reported in `results.csv`; exact recomputation additionally requires the
frozen per-candidate Gaussian-quality score column used by that experiment.

Depth and point-cloud rows require the corresponding precomputed asset-path
column in the dataset index.  They are alternative-modality baselines rather
than the selected model.


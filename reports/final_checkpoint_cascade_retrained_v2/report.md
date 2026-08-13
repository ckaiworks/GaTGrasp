# Final checkpoint-level Stage-1 -> Stage-2 cascade

Stage 1 runs once. Every Stage-2 row uses exactly the same selected actions.

| Stage-2 input | Acc. | Macro-F1 | AUROC | AUPRC | Brier | ECE-5 |
|---|---:|---:|---:|---:|---:|---:|
| Touch Only | 0.6842 | 0.5929 | 0.9706 | 0.9967 | 0.2793 | 0.4322 |
| RGB+Action13+Touch | 0.8947 | 0.7206 | 0.8088 | 0.9740 | 0.1046 | 0.0943 |
| RGB+Action13+ContactGS+CorridorGS+Touch | 0.9474 | 0.8190 | 0.9118 | 0.9896 | 0.0524 | 0.0499 |

CASCADE_CHECKPOINT_INFERENCE_PASS

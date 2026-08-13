# Table 4. Current clean Stage 1 selector ablations

所有变体均在每个物体相同的 150 个候选池上评价。每个消融块只与该块中的 full 行比较。

| Ablation block | Variant | S@1 (%) | H@5 (%) | H@10 (%) | MRR |
|---|---|---:|---:|---:|---:|
| Modality | Action Only | 84.21 (16/19) | 94.74 (18/19) | 100.00 (19/19) | 0.8760 |
| Modality | RGB+Action | 78.95 (15/19) | 94.74 (18/19) | 100.00 (19/19) | 0.8342 |
| Modality | GS+Action | 63.16 (12/19) | 89.47 (17/19) | 94.74 (18/19) | 0.7574 |
| Modality | RGB+Action+GS (full) | **89.47 (17/19)** | **94.74 (18/19)** | **100.00 (19/19)** | **0.9175** |
| Local GS region | Left/right contact patches only | 84.21 (16/19) | 94.74 (18/19) | 100.00 (19/19) | 0.8711 |
| Local GS region | Closing corridor only | 63.16 (12/19) | 94.74 (18/19) | 94.74 (18/19) | 0.7509 |
| Local GS region | Contact patches + corridor (full) | **89.47 (17/19)** | **94.74 (18/19)** | **100.00 (19/19)** | **0.9175** |
| GS node pooling | Mean Only | 73.68 (14/19) | 94.74 (18/19) | 100.00 (19/19) | 0.8277 |
| GS node pooling | Max Only | 78.95 (15/19) | 94.74 (18/19) | 100.00 (19/19) | 0.8461 |
| GS node pooling | Mean + Max (full) | **89.47 (17/19)** | **94.74 (18/19)** | **100.00 (19/19)** | **0.9175** |
| Selection-loss weight | Weighted BCE only (λ=0) | 84.21 (16/19) | 94.74 (18/19) | 100.00 (19/19) | 0.8947 |
| Selection-loss weight | Weighted BCE + λ=0.5 | 84.21 (16/19) | 94.74 (18/19) | 100.00 (19/19) | 0.8912 |
| Selection-loss weight | Weighted BCE + λ=1.0 (selected) | **89.47 (17/19)** | 94.74 (18/19) | 100.00 (19/19) | 0.9175 |
| Selection-loss weight | Weighted BCE + λ=1.5 | 84.21 (16/19) | 94.74 (18/19) | 100.00 (19/19) | 0.8912 |
| Temperature / warm-up | Fixed T=1.0, warm-up 5 | 78.95 (15/19) | 94.74 (18/19) | 100.00 (19/19) | 0.8649 |
| Temperature / warm-up | Cosine T 1.0→0.1, warm-up 10 | 84.21 (16/19) | 94.74 (18/19) | 100.00 (19/19) | 0.8912 |
| Temperature / warm-up | Cosine T 1.0→0.1, warm-up 5 (full) | **89.47 (17/19)** | **94.74 (18/19)** | **100.00 (19/19)** | **0.9175** |

说明：旧版 Mass-only/Rank-only、旧版 surface-channel、Depth 和 Point Cloud 结果没有混入本表，因为它们不是在当前确定性 Stage 1 代码与当前退火选择损失下完成的干净单变量对照。

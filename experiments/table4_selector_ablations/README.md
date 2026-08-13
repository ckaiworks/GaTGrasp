# Table 4: Selector ablations

Every row is a Stage-1 experiment on the same five folds and the same 150
actions per object.  The selected full implementation is shared from
`../../src/stage1/final/`; it is not copied into each block.

## Modality

- Action Only: reuse Table 2 `../table2_primary_selector/methods/action_only/`.
- RGB+Action: reuse Table 2 `../table2_primary_selector/methods/rgb_action/`.
- GS+Action: reuse Table 2 `../table2_primary_selector/methods/gs_action/`.
- RGB+Action+GS (full): selected Stage-1 source in `../../src/stage1/final/`.

## Local GS region

- Left/right contact patches only:
  `methods/local_region/contact_patches_only/train_stage1_clean_structural_no_corridor_annealedSelect_v1.py`.
- Closing corridor only:
  `methods/local_region/corridor_only/train_stage1_corridorOnly_annealedSelect_v1.py`.
- Contact patches + corridor: selected full Stage-1 source.

## GS node pooling

- Mean Only: `methods/pooling/mean_only/train_stage1_meanOnly_annealedSelect_v1.py`.
- Max Only: `methods/pooling/max_only/train_stage1_maxOnly_annealedSelect_v1.py`.
- Mean + Max: selected full Stage-1 source.

## Selection-loss weight

- lambda=0 uses the true BCE-only entry
  `methods/loss/bce_only/train_stage1_old12_fullcorridor_bceOnly_v1.py`.
- lambda=0.5 and lambda=1 use the selected full trainer with
  `--soft_top1_weight 0.5` and `1`.
- lambda=1.5 uses the selected full trainer with
  `--soft_top1_weight 1.5`.

## Temperature and warm-up

All schedule rows use the selected full trainer.  Only
`--select_temperature_start`, `--select_temperature_end`, and
`--soft_top1_warmup_epochs` change; see `methods.csv`.


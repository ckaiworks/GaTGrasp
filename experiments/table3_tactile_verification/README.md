# Table 3: Tactile verification

Table 3 freezes one common set of 19 Top-1 actions selected by the selected
Stage-1 model.  Only the Stage-2 input changes.  These rows must therefore be
evaluated with the same `raw_pair_id` list.

| Paper row | Runnable source | Actual input |
|---|---|---|
| Touch Only | `methods/touch_only/train.py` | bilateral touch only |
| RGB+Action+Touch | `methods/rgb_action_touch/train.py` | RGB, Action13 and bilateral touch |
| RGB+Action+GS+Touch | `methods/contact_corridor_gs_touch/train.py` | RGB, Action13, left/right contact GS, corridor GS and touch |

All three are deterministic single-output Stage-2 controls trained with
weighted BCE. `methods/_shared/` contains shared dataset/touch code and is not
an additional paper row. `results.csv` contains the exact same-action
connected inference values used in the current table. The repository's final
deployment Stage-2 is the full-GS/corridor implementation at
`../../src/stage2/final/train_stage2_fullgs_touch.py`.

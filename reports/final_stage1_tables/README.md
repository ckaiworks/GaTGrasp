# Final Stage-1 tables

This directory contains the paper-facing Stage-1 results under the selected,
deterministic five-fold protocol:

- 19 objects and 150 candidates per object;
- object-disjoint five-fold evaluation;
- seed 42 and the fixed epoch-50 checkpoint;
- weighted BCE plus annealed selection loss with lambda 1;
- cosine temperature schedule from 1.0 to 0.1 and five warm-up epochs.

`table2_primary_selector_comparison.*` contains the primary selector
comparison. `table4_stage1_clean_ablations.*` contains clean Stage-1
ablations. `sources/` contains the frozen result summaries used by
`validate_tables.py` to verify Table 4 and the shared Table-2 rows.

Stage-2 and connected Stage-1-to-Stage-2 results are stored separately in
`reports/final_stage2_oof/` and
`reports/final_checkpoint_cascade_retrained_v2/`.

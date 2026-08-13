# Paper experiments

The experiment code is organized by the paper table that it reproduces.

- `table2_primary_selector/`: primary Stage-1 selector comparison.
- `table3_tactile_verification/`: Stage-2 verification on the same frozen
  Stage-1 selected actions.
- `table4_selector_ablations/`: controlled Stage-1 ablations.

The selected Stage-1 implementation remains in `src/stage1/final/` and is
shared by Tables 2 and 4.  It is not duplicated inside the experiment
directories, so there is only one authoritative full-model source tree.

Each table directory contains:

- `README.md`: row-by-row explanation;
- `methods.csv`: paper row to runnable source/parameters mapping;
- `results.csv`: values used in the paper;
- `methods/`: method-specific code when the row needs a different model.


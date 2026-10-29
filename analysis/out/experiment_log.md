# Experiment log

One row per model version. The "test uses" column is the number of times the local test
split (`data/splits/test.csv`, 132 responses / 44 respondents) has been evaluated for that
version. The project allows this number to reach 1 once; it must never grow silently.

| Version | Date | Hyperparameters | Text fields | Features | Trained on | Selection rule | CV accuracy | Validation accuracy | **Test uses** | Test accuracy | Report role | Artifacts |
|---|---|---|---|---|--:|---|---|---|---|--:|---|---|---|
| v1 | 2026-10-29 | 300 trees, `max_depth=6`, `min_samples_leaf=3`, `max_features=log2`, gini, bootstrap, seed 311 | all four | 208 | 738 responses / 246 respondents (train + val) | 150-config grid, grouped 5-fold x 3 fold seeds, 1-SE (basis SE), simplest inside the band | 0.9043 (frontier 0.9082) | 0.9091, CI [0.8561, 0.9545] | **0** | not evaluated | candidate for the final report | `analysis/out/forest_*_v1/` |

## Rules

1. Update this table **before** any test evaluation, not after.
2. A version's test number belongs to the exact artifact listed in its row. If that artifact
   is retrained, the number no longer describes it.
3. Decisions about v2 must be justified by training-only cross-validation. Do not design a
   change by reading v1's test errors.
4. If a second test evaluation happens, record it explicitly as "test uses = 2" and state in
   the report that the final number is no longer an independent estimate.
5. Prefer changing code without changing predictions (verify with an array equality check on
   all 738 training responses). Such a version keeps `test uses = 1` inherited from v1.

## Open decision for v2

The user is considering changing preprocessing while keeping the architecture
(300 trees, depth 6, `log2`, leaf 3) fixed. Consequences to remember:

- Fixing hyperparameters while changing the feature space makes the choice a
  sensitivity analysis, not a tuned model; the report must say so.
- `search` and `diagnostics` always read `data/splits/train.csv` (606 responses, 202
  respondents), so v2's cross-validation stays comparable with v1's.
- Validation is already inside v1's training set, so v2 has **no** independent validation
  set. Its honest performance estimate is the grouped repeated CV over 606 responses, and
  the test split is unavailable for it.
- Any edit inside `data_preprocessing/encode_all.py` changes a file shared with v1.
  After such an edit, v1 is reproducible only from the frozen artifacts and the hashes in
  `V1_ARCHIVE.md`.

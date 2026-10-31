# Experiment log

One row per model version. The "test uses" column is the number of times the local test
split (`data/splits/test.csv`, 132 responses / 44 respondents) has been evaluated for that
version. The project allows this number to reach 1 once; it must never grow silently.

| Version | Date | Hyperparameters | Text fields | Features | Trained on | Selection rule | CV accuracy | Validation accuracy | **Test uses** | Test accuracy | Report role | Artifacts |
|---|---|---|---|---|--:|---|---|---|---|--:|---|---|---|
| v1 | 2026-10-29 | 300 trees, `max_depth=6`, `min_samples_leaf=3`, `max_features=log2`, gini, bootstrap, seed 311 | all four | 208 | 738 responses / 246 respondents (train + val) | 150-config grid, grouped 5-fold x 3 fold seeds, 1-SE (basis SE), simplest inside the band | 0.9043 (frontier 0.9082) | 0.9091, CI [0.8561, 0.9545] | **1** | 0.9015, CI [0.8485, 0.9470] | selected for the final report | `random_forest_deliverables/forest_*_v1/` |

## Pre-registration: v1 is the version going to test

Written before the test split is opened, as rule 1 requires.

- **Version selected:** v1. The choice rests on training-side evidence only.
- **Artifact to be evaluated:** `random_forest_deliverables/forest_final_v1/final_model.joblib`,
  sha256 `c08f5777706ad9a87740e0f6d732a80ac1a711d02a44d0983747a54fab765772`,
  1,089,420 bytes, written 2026-10-29 15:43:47.
- **Configuration:** 300 trees, gini, bootstrap, seed 311, `max_depth=6`,
  `min_samples_leaf=3`, `max_features=log2`, all four text fields, 208 features, fitted on
  738 responses from 246 respondents (train + validation).
- **Reproduction check before opening test:** refitting this exact configuration on the
  same 738 rows gave an identical column set and identical predictions on all 738 rows.
- **Test uses after this evaluation:** 1. This is the model's only test reading.

The comparison behind the choice, all of it on the 606 training rows and the 132
validation rows:

| model | CV, fifteen folds | overfitting gap | validation accuracy | validation errors |
|---|--:|--:|--:|--:|
| report | 0.9082 | +0.0852 | 0.9318 | 9 |
| v1 | 0.9043 | **+0.0217** | 0.9091 | 12 |
| v2 | 0.9043 | +0.0371 | 0.9242 | 10 |
| v3 | 0.9054 | +0.0385 | 0.9091 | 12 |

The four cannot be separated on accuracy: paired t test p = 0.13 / 0.25 / 0.41 on the folds,
McNemar p = 0.25 / 1.00 / 0.38 on the 132 validation rows. v1 was taken for the smallest
overfitting gap and the widest short-answer coverage, not for a higher point estimate. The
`report` model leads both CV and validation, but it fits 99.34% of its own training rows.

Inference cost was measured separately and is not a differentiator: a hand-written NumPy
traversal of v1's trees plus preprocessing takes about 140 ms for 132 rows against a 1 s
budget (`train/measure_inference.py`, `random_forest_deliverables/inference_time.json`).

## Test result

Opened once, on the artifact named above, with `--i-accept-using-the-test-set`.

    accuracy   0.9015   119 of 132
    macro F1   0.9022
    95% respondent-bootstrap CI   accuracy [0.8485, 0.9470], macro F1 [0.8500, 0.9473]

    precision / recall / F1
    Soccer       0.8864 / 0.8864 / 0.8864
    Baseball     0.8511 / 0.9091 / 0.8791
    Basketball   0.9756 / 0.9091 / 0.9412

    confusion, rows are actual
                 Soccer   Baseball  Basketball
    Soccer           39          5           0
    Baseball          3         40           1
    Basketball        2          2          40

Thirteen of the 132 rows are wrong, against twelve on validation, and the cross-validated
error rate of 0.0957 implies 12.6 errors on a batch this size. The test reading is
therefore 0.0076 below validation and 0.0028 below cross-validation, both well inside a
standard error of 0.026 on 132 rows, and the two respondent-bootstrap intervals overlap
almost completely.

The error shape moved but the total did not. On validation v1's confusion was
`[[37,3,4],[1,40,3],[0,1,43]]` and the single largest pair was Soccer called Basketball, at
four rows. On test that pair is empty and the losses spread out instead:

    recalled correctly     validation   test
    Soccer                 37 / 44      39 / 44      improves by two
    Baseball               40 / 44      40 / 44      unchanged
    Basketball             43 / 44      40 / 44      loses three

Basketball is where the difference went: it was called Soccer twice and Baseball twice,
where on validation it was called Baseball once. Baseball precision falls from 0.9091 to
0.8511 for the same reason, seven rows now arriving from the other two classes against four
before. Soccer recall is the one figure that improves, 0.8409 to 0.8864.

The lock `random_forest_deliverables/forest_final_v1/test_accessed.json` now exists. Any later change to
the model will need a fresh artifact and an explicit "test uses = 2" entry, and the number
above will stop being an independent estimate for it.

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

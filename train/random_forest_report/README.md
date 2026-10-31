# report: the random forest in `analysis/report/proposal.tex`, reproduced

`analysis/report/proposal.tex` already contains a completed three-family comparison and
it selects a random forest. Its row of that table is a specific model:

    300 bootstrap trees, gini impurity, no class weights, seed 311, one CPU thread
    max_depth = None, max_features = 'sqrt', min_samples_leaf = 1
    208 predictors
    CV    91.27 +- 2.14   over five grouped folds, fold seed 311
    val   93.18          123 of 132 responses, macro F1 0.9317
    train 99.34          602 of 606 responses, full-training refit

This directory exists so that the model has a name and a script, and so that the number in
the report can be checked rather than trusted. It is `report` in the same sense that `v1`,
`v2` and `v3` are versions: a fixed configuration, a set of artifacts, and a provenance
record.

It re-runs the report's own code instead of reimplementing it. `reproduce.py` imports
`SurveyPreprocessor`, `fit_model`, `metrics` and `CLASSES` from `analysis/model_experiments.py`,
and builds folds with `GroupKFold(n_splits=5, shuffle=True, random_state=311)` exactly as
that module's `run()` does. `data_preprocessing/encode_all.py` supplies the feature space.
Neither file is copied here, so the reproduction cannot drift away from what the report
ran.

## Reproduction result

Every recorded number is reproduced, and the validation predictions are identical row for
row:

| reading | reproduced | report |
|---|--:|--:|
| cv5 accuracy | 0.9127 +- 0.0214 | 0.9127 +- 2.14 |
| cv5 macro F1 | 0.9125 | 0.9125 |
| cv15 accuracy | 0.9082 +- 0.0201 | 0.9082 (v1's search, same configuration) |
| fit606 train accuracy | 0.9934 (602/606) | 99.34 |
| validation accuracy | 0.9318 (123/132) | 93.18 |
| validation macro F1 | 0.9317 | 0.9317 |
| validation predictions | 0 of 132 differ | - |

`analysis/model_experiments.py` no longer hashes to what the report's manifest recorded
(the report ran `4a7e8f26...`, the file is now `7c098ced...`). The reproduction still
matches to the last digit, including all 132 predictions, so whatever changed in that file
does not touch the forest's path. `reproduce.py` prints the hash comparison on every run
and stores it in `reproduction.json`, so this cannot silently become false later.

## Two things the reproduction makes visible

**The report's CV number is one fold seed, and it is the favourable one.** Reading the same
configuration over the fifteen folds v1, v2 and v3 use:

    fold seed 311    0.9127
    fold seed 312    0.9076
    fold seed 313    0.9041
    all fifteen      0.9082 +- 0.0201

The difference between the report's 91.27 and this version's 90.82 is not a difference in
models. It is the choice of fold seed, and seed 311 is the luckiest of the three by
0.0045. v1, v2 and v3 report fifteen folds for exactly this reason.

**The model memorises most of the training split.** In-fold train accuracy is 0.9934 with
the full fit, and 0.9917 to 0.9959 across the five folds. Unlimited depth with one sample
per leaf leaves almost nothing unfitted, which is why v1's one-standard-error rule kept the
same feature space and replaced these hyperparameters with `depth=6 / log2 / leaf=3`.

## How this version differs from v1, v2 and v3

| | report | v1 | v2 | v3 |
|---|---|---|---|---|
| predictors | 208 | 208 | 183 | 173 |
| hyperparameters | depth None, sqrt, leaf 1 | depth 6, log2, leaf 3 | depth 8, log2, leaf 2 | depth 8, sqrt, leaf 2 |
| folds | 5 (one fold seed) | 15 | 15 | 15 |
| grid | none, the configuration is given | 150 | 150 | 150 |
| selection | CV, then **validation**, then family preference | CV, one-standard-error | CV, one-standard-error | CV, one-standard-error |

The last row matters for reading the numbers. The report chose between softmax, forest and
MLP using validation accuracy, so its validation figure is not an independent estimate of
the forest. v1, v2 and v3 never let validation take part in a selection decision.

## Running it

    python reproduce.py

Writes to `random_forest_deliverables/report_forest/` unless `--out` says otherwise. The run takes about
twenty seconds: twenty-one forest fits, being fifteen for the fifteen-fold reading, five
for the report's own protocol and one for the full refit.

`reproduce.py` will exit non-zero on an assertion if a respondent straddles two folds, and
it prints `reproduced` or `NOT reproduced` at the end. The script reads the training split
and the validation split; it never reads the test split, and every JSON it writes records
`test_accessed: false`.

## Artifacts

    cv5_records.csv             five folds, fold seed 311, one row per fold
    cv15_records.csv            fifteen folds, one row per fold with its fold seed
    validation_predictions.csv  id, actual, predicted for the 132 validation rows
    validation_confusion.csv    rows actual, columns predicted
    reproduction.json           every reproduced number beside the recorded one
    run_manifest.json           parameters, splits read, source hashes, versions

The confusion matrix is the report's:

                 Soccer   Baseball  Basketball
    Soccer           39          2           3
    Baseball          1         41           2
    Basketball        0          1          43

## Files

    reproduce.py     the whole version: fold construction, three readings, the check
    README.md        this file

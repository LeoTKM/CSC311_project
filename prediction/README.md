# Predictions without scikit-learn

The course requires `pred.py` to run on Python's standard library, NumPy and pandas alone.
This directory is that predictor for the frozen v1 random forest, plus the two data files it
needs and the scripts that produced and checked them.

```
pred.py               the predictor - the only file the grader needs to call
model_data.npz        the 300 trees as flat arrays, the IDF vectors, the imputation values
preprocessing.json    column order, TF-IDF vocabularies, column names, class order
export_model.py       writes the two data files from the trained artifact (needs scikit-learn)
verify_pred.py        checks pred.py against the original pipeline (needs scikit-learn)
```

Submission set is `pred.py` + `model_data.npz` + `preprocessing.json`: **159 KB**, against the
10 MB limit.

## Using it

```python
from pred import predict_all

labels = predict_all('held_out_responses.csv')     # one label per row, in row order
```

Or from the command line:

```
python pred.py held_out_responses.csv                 # prints index<TAB>label
python pred.py held_out_responses.csv predictions.csv  # writes a one-column CSV
```

The input CSV must carry the twelve survey columns. It may use either the original long
question headers or the short names used by `data/splits/*.csv`; both are accepted. An `id`
column and a `label` column are optional and are ignored. Row order is preserved. Columns the
model does not use are ignored.

## What it does, in order

1. `sanitize` - blank out structured answers that are off the scale, unknown agreement
   labels, and malformed multi-select answers, exactly as the training pipeline did.
2. `build_features` - Q1 divided by 10, Q2-Q4 mapped to 0/0.25/0.5/0.75/1, 21 multi-hot
   indicators, four per-field TF-IDF blocks (sublinear term frequency, the recorded IDF, L2
   rows), payment parsed then mean-filled then `log1p`, seven question-level missing flags.
3. Reorder the 208 columns to the recorded order and fill any remaining gap with the value
   recorded at training time.
4. Descend all 300 trees with NumPy, average the leaf class distributions, take the argmax.

The class order stored inside the artifact is alphabetical, `Baseball`, `Basketball`, `Soccer`.
The label is read from the recorded `classes` list, so nothing depends on a hard-coded index.

## How it was checked

`verify_pred.py` compares this predictor against the original artifact on three sets of rows,
comparing **the whole 208-column matrix element by element**, not just the final labels,
because identical predictions can hide a feature that differs without crossing a split
threshold.

```
train.csv                 606 rows    max |diff| 2.22e-16   prediction disagreements 0
val.csv                   132 rows    max |diff| 2.22e-16   prediction disagreements 0
synthetic edge cases       12 rows    max |diff| 0.00e+00   prediction disagreements 0
```

The synthetic sheet is the awkward part: out-of-range ease values (`99`, `-2`, `abc`),
an unknown agreement label, unknown multi-select options, refusals (`n/a`, `none`), currency
words and symbols (`CAD 200`, `$150`), ranges (`50-100`, `between 100 and 200`), `1k`,
thousands separators (`1,000`, `200, 500`), hyphens and apostrophes in the text
(`hot-dog`, `don't`), digits inside titles (`1999`), inconsistent case and stray whitespace.

The residual 2.22e-16 is one unit in the last place, from a different summation order inside
the L2 row normalisation. Both matrix comparisons are exact to 0.00e+00 on the synthetic rows.

The check also covers short-header renaming, and CSVs with and without a `label` column.

`export_model.py` runs the same comparison on the training rows before writing the files, so
the exported arrays cannot silently diverge from the model.

## Timing

132 rows in 83 ms, about 0.63 ms per row. The course asks for roughly 60 predictions within a
minute: this does 60 in 38 ms. Cost scales with the number of trees and their depth, not with
the number of rows.

## Rebuilding the export

Only needed if the model itself changes.

```powershell
python export_model.py     # needs the csc311 conda environment: joblib + scikit-learn
python verify_pred.py      # same environment
```

`export_model.py` reads `random_forest_deliverables/forest_final_v1/final_model.joblib` and
refits nothing. If the model is retrained, the artifact must be replaced first, and the test
evaluation it carries would no longer describe the new model.

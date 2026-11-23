# Random forest on the CSC311 survey

The training pipeline for the survey's three-class model. Training uses scikit-learn;
nothing in this folder is imported at prediction time.

Input is `data/splits/train.csv` (606 responses, 202 respondents, three responses each)
and `data/splits/val.csv` (132 responses, 44 respondents). Splits are respondent-grouped:
no respondent appears in two splits.

## 1. Preprocessing

Two modules do the work. `data_preprocessing/encode_all.py` turns raw survey cells into
numbers, and `analysis/model_experiments.py` wraps it in a `SurveyPreprocessor` that fits
everything on the training rows of one fold and only transforms the held-out rows.
`features.py` here subclasses that preprocessor to allow a question block to be dropped.

### Structured answers

| Question | Encoding |
|---|---|
| Q1 ease of starting | numeric 0-10, divided by 10 |
| Q2 strategy dependence | strongly disagree 0, disagree 0.25, neutral 0.5, agree 0.75, strongly agree 1 |
| Q3 injury risk | same 0 to 1 map |
| Q4 individual brilliance | same 0 to 1 map |
| Q5 venue | 7 indicator columns |
| Q6 season | 4 indicator columns |
| Q7 music type | 10 indicator columns |
| Q12 willingness to pay | parsed to one number, missing filled with the training mean, then `log1p` |

An unknown or malformed structured answer is treated as missing before encoding.

### Text answers

Each of the four free-text answers gets its own TF-IDF block, fitted on the training rows
of the fold alone:

```
token pattern      (?u)\b[a-zA-Z]{2,}\b     two or more ASCII letters
lowercase          yes
n-grams            unigrams only
min_df             2
max_features       50 for food, 20 for colour, 50 for song, 50 for feeling
sublinear_tf       1 + log(tf)
smooth_idf         yes
norm               l2 over the row
stopwords          none
```

A reply that produces no term leaves its block all zeros; it is not treated as missing.
Non-Latin replies are emptied by the token pattern rather than by a language test.

### Missing values and flags

Every column is imputed from the training rows of the fold: the median for Q1-Q4, the
most frequent value for the multi-select and text columns. Seven flags record whether
Q1 to Q7 were blank (`q1_missing` .. `q7_missing`), and each text block carries a
`<field>_text_missing` flag.

### Resulting matrix

| Block | Columns |
|---|--:|
| Q1-Q4 ordinal | 4 |
| Q5-Q7 indicators | 21 |
| payment | 2 |
| missing flags Q1-Q7 | 7 |
| food TF-IDF | 50 |
| colour TF-IDF | 20 |
| song TF-IDF | 50 |
| feeling TF-IDF | 50 |
| text missing flags | 4 |
| **total** | **208** |

`id` and `label` never enter the matrix, and no column is standardised: a forest does not
need it, and the scaler exists only for the linear and neural baselines in
`analysis/model_experiments.py`.

## 2. Field ablation

Before the hyperparameter search, six feature sets are compared on a reduced grid
(`max_depth` 4 or None, `min_samples_leaf` 1 or 5, `max_features` sqrt) with one fold seed.

| Variant | Text blocks kept | Columns |
|---|---|--:|
| `all` | food, colour, song, feeling | 208 |
| `no_song` | food, colour, feeling | 157 |
| `no_feeling` | food, colour, song | 157 |
| `no_song_no_feeling` | food, colour | 106 |
| `food_only` | food | 85 |
| `none` | - | 34 |

Dropping a text field removes exactly its columns; nothing else changes. All six variants
landed inside fold noise, so all four text blocks are kept.

## 3. Hyperparameter search

```
grid          max_depth       4, 6, 8, 12, 16, None
              min_samples_leaf 1, 2, 3, 5, 8
              max_features     sqrt, log2, 0.3, 0.5, 1.0        150 combinations
folds         GroupKFold(5, shuffle=True, random_state=seed)
fold seeds    311, 312, 313
data          data/splits/train.csv only
fits          150 x 3 x 5 = 2250
fixed         n_estimators 300, criterion gini, bootstrap, class_weight None,
              oob_score False, n_jobs 1
```

Each fold contributes one record with accuracy, macro-F1, log loss, in-fold training
accuracy and feature count, so fold-to-fold spread stays visible.

Selection rule:

1. rank by mean accuracy over the 15 grouped observations;
2. form a one-standard-error band, `best - spread / sqrt(15)`;
3. inside the band, take the simplest configuration: shallower depth, then larger leaves,
   then fewer candidate split features.

Result:

| | value |
|---|---|
| best on the grid | 0.9082 at `max_depth` None, `max_features` sqrt, `min_samples_leaf` 1 |
| band width | 0.005196 |
| configurations inside the band | 20 of 150 |
| **selected** | **`max_depth` 6, `max_features` log2, `min_samples_leaf` 3** |
| selected CV accuracy | **0.9043** +- 0.0230 |
| selected in-fold training accuracy | 0.9260 |

## 4. Diagnostics

Run after selection, changing nothing:

| Diagnostic | Result |
|---|---|
| tree count 100 / 200 / 300 / 500 / 800 | accuracy 0.8977 / 0.8995 / 0.9061 / 0.9078 / 0.9078, log loss rising above 300, so 300 trees |
| learning curve 25% to 100% of the rows | 0.8879 to 0.9061, so four times the data buys 1.8 points |
| permutation importance by block | colour 0.0695, Q7 music 0.0403, song 0.0366, Q5 venue 0.0267, food 0.0233, payment 0.0100, feeling 0.0094, Q6 season 0.0091, Q1-Q4 negligible, missing flags 0.0000 |

## 5. Evaluation

The model is fitted on the 606 training responses and scored once on the 132 validation
responses:

| Metric | Value |
|---|--:|
| accuracy | 0.9091 |
| macro F1 | 0.9088 |
| bootstrap interval, resampling respondents | [0.8561, 0.9545] |
| recall Soccer | 0.8409 |
| recall Baseball | 0.9091 |
| recall Basketball | 0.9773 |

The predictions and the confusion matrix are kept next to the metrics, so the reading can
be checked row by row.

## 6. Final model

For the model that ships, the validation split is folded into training and the same
configuration is refitted on all of it:

| | value |
|---|---|
| rows / respondents | 738 / 246 |
| features | 208 |
| parameters | `max_depth` 6, `max_features` log2, `min_samples_leaf` 3, 300 trees |
| training accuracy | 0.9440 |
| forest size | 10,498 nodes, 5,399 leaves, every tree depth 6, 34 nodes per tree |
| split on | 200 of the 208 columns |

Because the validation rows are now inside the training set, this artifact cannot produce
a held-out reading of itself; `final_manifest.json` records
`"validation_now_inside_training": true` for that reason.

The test split has not been evaluated. `test_accessed` is `false` in every manifest, and
the `test` stage is behind two gates: an explicit `--i-accept-using-the-test-set` flag and
a `test_accessed.json` lock file that makes a second run abort.

## 7. Stages and outputs

Every stage needs a new `--output` directory; results are never overwritten.

| Stage | What it does | Main outputs |
|---|---|---|
| `ablation` | six field sets x 4 configurations x 5 folds | `ablation_folds.csv`, `ablation_summary.csv` |
| `search` | 150 configurations x 3 fold seeds x 5 folds, then 1-SE selection | `cv_folds.csv`, `cv_summary.csv`, `selection_rationale.json` |
| `diagnostics` | tree-count curve, learning curve, permutation importance by block | `trees_curve.csv`, `learning_curve.csv`, `permutation_importance_by_block.csv` |
| `fit` | fit on train, score validation with a respondent bootstrap interval | `validation_predictions.csv`, `validation_confusion.csv`, `validation_metrics.json` |
| `final` | refit preprocessing and model on train + validation | `final_model.joblib`, `final_manifest.json` |
| `test` | one-shot test evaluation, guarded by a lock file | `test_predictions.csv`, `test_metrics.json`, `test_accessed.json` |

```bash
python train/random_forest/run.py --dry-run --stage search
python train/random_forest/run.py --stage ablation  --output random_forest_deliverables/forest_ablation_v1
python train/random_forest/run.py --stage search    --output random_forest_deliverables/forest_search_v1
python train/random_forest/run.py --stage diagnostics --source random_forest_deliverables/forest_search_v1 \
    --output random_forest_deliverables/forest_diagnostics_v1
python train/random_forest/run.py --stage fit       --source random_forest_deliverables/forest_search_v1 \
    --output random_forest_deliverables/forest_validation_v1
python train/random_forest/run.py --stage final     --source random_forest_deliverables/forest_search_v1 \
    --output random_forest_deliverables/forest_final_v1
```

## 8. Guards

- Respondent grouping is asserted for every fold; a failure aborts the run.
- `id` and `label` never enter the feature matrix.
- Vocabularies, IDF weights, imputation values and missing flags are fitted on the
  fitting rows of one fold only.
- A row-level bootstrap would split a respondent's three responses across the training
  and evaluation halves, so OOB scoring is off and the confidence intervals resample
  respondents instead of responses.
- `predict_proba` columns follow `model.classes_`, which is alphabetical
  (`Baseball`, `Basketball`, `Soccer`). Log loss is called with `labels=list(model.classes_)`
  and the class mapping is never positional.
- The test split stays unread until the guarded `test` stage.

## 9. Smoke check

`smoke_check.py` fits only the preprocessor, trains no scored model and writes no file. It
confirms the column counts of every field variant and that the CV interface lines up.

```bash
python train/random_forest/smoke_check.py
```

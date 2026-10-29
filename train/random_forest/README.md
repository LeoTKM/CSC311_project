# train/random_forest

Random-forest training pipeline for the CSC311 survey project.

Training is allowed to use scikit-learn. **Only the eventual inference script must be
dependency-free**, so nothing in this folder is imported at prediction time.

## Why this exists next to `analysis/model_experiments.py`

`analysis/model_experiments.py` compared three families (softmax / forest / MLP) and
already searched 24 forest configurations. That run is the *reference implementation*
used for model selection and stays as report evidence. This folder is the production
path for the final forest:

- a wider, less gapped hyperparameter grid
- repeated grouped cross-validation (three fold seeds instead of one)
- a one-standard-error selection rule, so a 0.8-point difference inside fold noise no
  longer decides the model
- a text-field ablation (does dropping `song` / `feeling` help?)
- diagnostics the reference run never produced (tree-count convergence, learning
  curve, permutation importance by question block)
- a final refit on training + validation, leaving the test split for one evaluation

## Hyperparameters

Searched (`config.GRID`, 150 combinations):

| Parameter | Values |
|---|---|
| `max_depth` | 4, 6, 8, 12, 16, None |
| `min_samples_leaf` | 1, 2, 3, 5, 8 |
| `max_features` | `sqrt`, `log2`, 0.3, 0.5, 1.0 |

Fixed: 300 trees, Gini, bootstrap, no class weights (classes are exactly balanced),
one CPU thread, no OOB score (a row-level bootstrap does not keep a respondent's three
responses together).

Already tried in the reference run: `max_depth` 4/8/12/None x `min_samples_leaf` 1/3/5 x
`max_features` `sqrt`/0.5. The best was unlimited depth with leaf 1 and `sqrt`
(CV 0.9127 +- 0.0214, training accuracy 0.9934). Never varied there: `n_estimators`,
`criterion`, `bootstrap`, `class_weight`, `min_samples_split`, and the TF-IDF caps.

## Field ablation

`config.TEXT_VARIANTS` compares six feature sets on a reduced grid (4 combinations) and
one fold seed:

| Variant | Text blocks kept |
|---|---|
| `all` | food, colour, song, feeling |
| `no_song` | food, colour, feeling |
| `no_feeling` | food, colour, song |
| `no_song_no_feeling` | food, colour |
| `food_only` | food |
| `none` | - (structured + payment only) |

Motivation: `song` turns 130 of 574 non-blank answers (22.6%) and `feeling` 108 of 599
(18%) into all-zero TF-IDF rows, so about 100 of the 208 current features are sparse or
constant depending on the response. Dropping both removes 102 columns.

Use `--no-structured` for the reverse direction (text only).

## Stages

| Stage | What it does | Main outputs |
|---|---|---|
| `ablation` | six field sets x 4 configurations x 5 folds | `ablation_folds.csv`, `ablation_summary.csv` |
| `search` | 150 configurations x 3 fold seeds x 5 folds = 2250 fits, then 1-SE selection | `cv_folds.csv`, `cv_summary.csv`, `selection_rationale.json` |
| `diagnostics` | tree-count curve, learning curve, permutation importance | `trees_curve.csv`, `learning_curve.csv`, `permutation_importance_by_block.csv`, `diagnostics.png` |
| `fit` | refit on train, score validation with a respondent bootstrap CI | `validation_predictions.csv`, `validation_confusion.csv`, `validation_metrics.json` |
| `final` | refit preprocessing and model on train + validation | `final_model.joblib`, `final_manifest.json` |
| `test` | one-shot test evaluation, guarded by a lock file | `test_predictions.csv`, `test_metrics.json`, `test_accessed.json` |

Every stage needs a **new** `--output` directory; results are never overwritten.

## Running

```bash
# 1. see the plan without touching anything
python train/random_forest/run.py --dry-run --stage search
python train/random_forest/run.py --dry-run --stage ablation

# 2. does dropping song / feeling help?
python train/random_forest/run.py --stage ablation --output analysis/out/forest_ablation

# 3. full search on the chosen field set (about 20-30 minutes at 0.6 s/fit)
python train/random_forest/run.py --stage search --text-fields no_song_no_feeling \
    --output analysis/out/forest_search

# 4. diagnostics, then the validation check
python train/random_forest/run.py --stage diagnostics \
    --output analysis/out/forest_diagnostics --source analysis/out/forest_search
python train/random_forest/run.py --stage fit \
    --output analysis/out/forest_val --source analysis/out/forest_search

# 5. freeze, refit on train + validation, then evaluate the test split exactly once
python train/random_forest/run.py --stage final \
    --output analysis/out/forest_final --source analysis/out/forest_search
python train/random_forest/run.py --stage test \
    --output analysis/out/forest_final --source analysis/out/forest_search \
    --i-accept-using-the-test-set
```

## Selection rule

Primary metric: mean accuracy over 15 grouped observations (5 folds x 3 fold seeds).
The **one-standard-error band** is `best -+ spread`, where `spread` is either the
fold-and-seed standard deviation (`--se-basis sd`) or that deviation divided by
`sqrt(15)` (`--se-basis se`, the default and the usual ESL definition). Inside the band
the **simplest** configuration wins: shallower depth, then larger leaves, then fewer
candidate split features.

Use `--select-metric log_loss` to select on a continuous metric instead; accuracy on
121 held-out responses per fold has a resolution of only 0.83 points, which is the same
size as the differences between the top configurations.

## Smoke check

`smoke_check.py` verifies the ablation plumbing without training anything: it fits only
the preprocessor (no forest) and runs one 5-tree fit on one fold to confirm the CV
interface and the `predict_proba` column alignment. It writes no files.

```bash
python train/random_forest/smoke_check.py
```

Expected column counts on the full training split:

| Variant | Features |
|---|--:|
| `all` | 208 |
| `no_song` | 157 |
| `no_feeling` | 157 |
| `no_song_no_feeling` | 106 |
| `food_only` | 85 |
| `none` | 34 |
| text only (`--no-structured`) | 174 |

## Guards

- Respondent grouping is asserted for every fold; a failure aborts the run.
- `id` and `label` never enter the feature matrix (`AblationPreprocessor` inherits the
  checks of `SurveyPreprocessor`).
- Vocabularies, IDF weights, imputation values and missing flags are fitted on the
  fitting rows of one fold only.
- `predict_proba` columns follow `model.classes_`, which is **alphabetical**
  (`Baseball`, `Basketball`, `Soccer`). Log loss is therefore called with
  `labels=list(model.classes_)`, and the class mapping must never be positional.
- The test split stays unread until the guarded `test` stage.

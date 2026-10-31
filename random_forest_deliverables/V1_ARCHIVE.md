# v1 archive — random-forest model (frozen)

Archived on 2026-10-29. The test split was evaluated **once**, on 2026-10-30, against the
artifact listed below. **That evaluation is spent.** v1's test accuracy of 0.9015 is its single
independent estimate; any retraining produces a new artifact for which it no longer holds.

## Frozen directories

| Directory | Content |
|---|---|
| `random_forest_deliverables/forest_ablation_v1/` | text-field ablation (6 variants on a reduced grid) |
| `random_forest_deliverables/forest_search_v1/` | 150-configuration grouped repeated CV and the 1-SE decision |
| `random_forest_deliverables/forest_diagnostics_v1/` | tree-count curve, learning curve, permutation importance |
| `random_forest_deliverables/forest_validation_v1/` | validation predictions, metrics, confusion matrix |
| `random_forest_deliverables/forest_final_v1/` | **the model**: `final_model.joblib` + `final_manifest.json` |

These five directories are frozen. Do not write into them, do not re-run a stage with
these paths (the runner refuses existing directories anyway). Any v2 work goes into new
directories such as `random_forest_deliverables/forest_final_v2/`.

## Model card

**Task.** Three-class classification (Soccer / Baseball / Basketball) from a 12-question
survey. Each respondent answered three times, once per sport, so the 738 responses in the
final training set come from 246 people and must be grouped by `id` in every evaluation.

**Architecture.** Random forest of 300 CART trees, Gini impurity, bootstrap sampling,
probabilities averaged across trees, argmax for the label.

| Hyperparameter | Value | Origin |
|---|---|---|
| `n_estimators` | 300 | fixed; checked with a tree-count curve (500 gains 0.17 pp, log loss worsens) |
| `criterion` | gini | fixed |
| `max_depth` | 6 | searched, then chosen by the 1-SE rule |
| `min_samples_leaf` | 3 | searched, then chosen by the 1-SE rule |
| `max_features` | log2, that is 7 of 208 candidate columns per split | searched, then chosen by the 1-SE rule |
| `bootstrap` | True | fixed |
| `class_weight` | None | classes are exactly balanced (246 responses each) |
| `oob_score` | False | a row-level bootstrap does not keep a respondent together |
| `min_samples_split` | 2 (default) | not searched |
| `max_leaf_nodes` / `min_impurity_decrease` | None / 0.0 | not searched |
| `n_jobs` | 1 | single thread for reproducible timing |
| `random_state` | 311 | model seed; fold seeds were 311 / 312 / 313 |

**Measured size.** 300 trees, 10,498 nodes, 5,399 leaves, every tree exactly depth 6,
about 35 nodes per tree.

**Inputs.** 208 numeric features, in this order:

| Block | Columns |
|---|--:|
| TF-IDF text | 174 (food 50, colour 20, song 50, feeling 50, plus 4 `*_text_missing` flags) |
| Q5-Q7 multi-hot | 21 |
| missing flags `q1..q7` | 7 |
| Q2-Q4 Likert (0, .25, .5, .75, 1) | 3 |
| Q12 payment (`payment_amount`, `payment_missing`) | 2 |
| Q1 ease (0-10 divided by 10) | 1 |

**Preprocessing.** TF-IDF: lowercase, token pattern `(?u)\b[a-zA-Z]{2,}\b`, unigrams,
`min_df=2`, `sublinear_tf`, `smooth_idf`, `norm='l2'`, no stopwords, per-field caps
50 / 20 / 50 / 50. Structured imputation: median for Q1-Q4, mode for multi-hot (ties to 0),
zero for an entirely absent column. Payment is filled with the **train+val mean 256.92**
then `log1p` (the transform does not change any forest split). No standardisation.
All vocabularies, IDF weights, imputation values and flags were fitted on training rows
only.

**Class order inside the model.** `sklearn` stores `['Baseball', 'Basketball', 'Soccer']`
(alphabetical). Every consumer must map by name, never by column position.

## Selection record

| Item | Value |
|---|---|
| Field set | `all` — chosen from the ablation, which used training-only CV |
| Search grid | 150 configurations (`max_depth` 6 x `min_samples_leaf` 5 x `max_features` 5) |
| CV protocol | `GroupKFold(5)` by respondent x 3 fold seeds (311/312/313) = 15 observations, train only |
| Best CV configuration | `max_depth=None, max_features=sqrt, min_samples_leaf=1`, CV 0.9082 |
| 1-SE band | width 0.0052 (spread / sqrt(15)), threshold 0.9030, **20 of 150** configurations inside |
| **Chosen configuration** | `max_depth=6, max_features=log2, min_samples_leaf=3`, CV **0.9043** |
| Search runtime | 1600 s (2250 fits) |

The chosen model gives up 0.4 points of CV accuracy against the frontier in exchange for a
training accuracy of 0.9440 instead of 0.9934 and a model 6.8 times smaller in nodes.

## Headline numbers

| Item | Value |
|---|---|
| Validation accuracy (44 respondents / 132 responses) | 0.9091 |
| Validation macro-F1 | 0.9088 |
| Validation accuracy, respondent-bootstrap 95% CI | [0.8561, 0.9545] |
| Per-class recall | Soccer 0.8409, Baseball 0.9091, Basketball 0.9773 |
| Final training set | 738 responses / 246 respondents (canonical train + validation) |
| Test evaluations so far | **1** |

The earlier reference implementation (`analysis/model_experiments.py`, unlimited depth,
`sqrt`, leaf 1, trained on 606 responses) scored 0.9318 on the same validation set. The two
are 2.3 points apart, but the validation set holds only 44 people, so its interval is about
10 points wide; the difference is not distinguishable. v1 was chosen for stability and size,
not because it is measurably more accurate.

## Test evaluation

Opened on 2026-10-30 against `random_forest_deliverables/forest_final_v1/final_model.joblib`, sha256
`c08f5777706ad9a87740e0f6d732a80ac1a711d02a44d0983747a54fab765772`, and not re-opened since.
The runner wrote `test_accessed.json` in that directory, and refuses a second evaluation of the
same artifact.

| Item | Value |
|---|---|
| Test accuracy (44 respondents / 132 responses) | **0.9015** (119/132) |
| Test macro-F1 | 0.9022 |
| Test accuracy, respondent-bootstrap 95% CI | [0.8485, 0.9470] |
| Test macro-F1, respondent-bootstrap 95% CI | [0.8500, 0.9473] |
| Per-class precision / recall / F1 | Soccer .8864 / .8864 / .8864 |
| | Baseball .8511 / .9091 / .8791 |
| | Basketball .9756 / .9091 / .9412 |
| Respondents with all three correct | 32 of 44 |

    confusion, rows are actual
                 Soccer   Baseball  Basketball
    Soccer           39          5           0
    Baseball          3         40           1
    Basketball        2          2          40

Reading against the other two splits: cross-validation over fifteen folds gave 0.9043 +- 0.0230
and validation gave 0.9091. Test is 0.0028 below cross-validation and 0.0076 below validation,
both well inside a standard error of 0.026 on 132 rows, and the bootstrap intervals overlap
almost completely. The error total is stable at 12 and 13 rows, but the errors are not the same
ones: on validation the largest pair was Soccer called Basketball, on test that pair is empty
and Soccer loses five rows to Baseball while Basketball's recall falls from 43/44 to 40/44.
Soccer remains the weakest class on every split.

The pre-registration that fixes this reading to this artifact, and the decision rule under
which v1 was chosen, are in `experiment_log.md`.

## Pinned hashes (sha256)

Artifacts:

```
random_forest_deliverables/forest_ablation_v1/ablation_folds.csv                  889fb80664379a1430674e864b5dd0e959287e352ff0f0b2154785af0a0432af
random_forest_deliverables/forest_ablation_v1/ablation_summary.csv                11fb662cd20baf6a9112da7b9a85b3c0a63945848ea62357563a55bd0810a6e3
random_forest_deliverables/forest_ablation_v1/run_manifest.json                   db62a836f55ae72b1891e8641716bfdc7b303f3b3cddf13f6a5ac9a9a543f291
random_forest_deliverables/forest_search_v1/cv_folds.csv                          875ce0a5a6f5d91206a6bd8462d0b7e62b5bdac0fa355b624123b0106c945e63
random_forest_deliverables/forest_search_v1/cv_summary.csv                        3853576d745af329d9a50f0aa17f91ab991286bd0a05f62e5279b404a2641c7b
random_forest_deliverables/forest_search_v1/selection_rationale.json              596652f6acbb73f7388d9806ce8e108fee16778825fd63957853cc99d6d3e2ef
random_forest_deliverables/forest_diagnostics_v1/trees_curve.csv                  18ac28ea17b6464b983182c36ce2666d2841c4198d3aeb2f7bbd5848a3055c44
random_forest_deliverables/forest_diagnostics_v1/learning_curve.csv               57dc65bd9783d5ecd6b28fe21592f32d4cfde3d4c679687eb443dcc7a6cc0076
random_forest_deliverables/forest_diagnostics_v1/permutation_importance.csv       bc22b76a1f4222a49e2c29405d90e19bbc66187f1dd47477c9d8de9a25a5a45b
random_forest_deliverables/forest_diagnostics_v1/permutation_importance_by_block.csv b2ec7705b7dfd7540db77778709e6cd32f5e5261f3b70e09964e1b635ab165a5
random_forest_deliverables/forest_diagnostics_v1/diagnostics.png                  b557ca0ad5c79d6e00a02fd88c4d3e832f009bf35ee3bd693c6fa967a8c9e984
random_forest_deliverables/forest_validation_v1/validation_metrics.json           4fe456bc955f5006c63db516636eca7cd356b86122abc5113a4fbd29f9c49e01
random_forest_deliverables/forest_validation_v1/validation_predictions.csv        8e23de7b664d7407b31299b7765d6cee044c5dbeedc171b34691944023f48400
random_forest_deliverables/forest_validation_v1/validation_confusion.csv          aa89eb63a7437124b6b4f0d9f94e42106c2cafc73a4463200fb419c60b363c38
random_forest_deliverables/forest_final_v1/final_model.joblib                     c08f5777706ad9a87740e0f6d732a80ac1a711d02a44d0983747a54fab765772
random_forest_deliverables/forest_final_v1/final_manifest.json                    1c6f64dc1c641876532dafc999a6d889d3f4e460c29091942574a3fa2f0e3c89
```

Inputs and code that produced v1:

```
data/splits/train.csv                        336c0aa1c3202f6f24a1b8b1d9e12a78727e8852e5d126baabbbc6f142761a47
data/splits/val.csv                          d43c9a213a55e9a251bb22aeecec0c5fd363b1d054cbb8157a860285d85ff9f8
data/splits/test.csv                         7adedba7f665a60f435fa0a504debd715dad007c62cbc30cd62b7d6654fd8ac0
data_preprocessing/encode_all.py             c374e9502ec92808eeede80c8f17f6ad36024a4acd046da34eac0304c291c601
analysis/model_experiments.py                7c098ced486687a2b692eef18b693b1972031935b65fd563ec0e3133c033a0f3
train/random_forest/config.py                506789b900205952bf8797b2bd5e2d9ae1957db127a473db90d444e54bd03153
train/random_forest/features.py              f3e6052ca847b9fa370806c3e62a2c3b5534f873e0bc66d1c4043d2ff5022841
train/random_forest/selection.py             662b1398e5b04624a966f8853da2156069211d986eddfcd330702571d1de313d
train/random_forest/run.py                   00bb1b9e927094400bc508f33bef610ec919af971b2d419cbc2ecccdee89260e
```

If `encode_all.py`, `model_experiments.py` or the four files in `train/random_forest/`
are edited, v1 is **no longer reproducible from the working tree**; only the frozen
artifacts above still describe it. The saved `final_model.joblib` keeps working regardless,
because it stores the fitted vectorizers, token counts, IDF weights and imputation values
inside the artifact.

Two of the code hashes above were re-pinned when these artifacts moved from `analysis/out/`
into this directory. `config.py` changed only in its `DEFAULT_OUTPUT` constant and `run.py`
only in the example paths inside its module docstring; neither touches the preprocessing, the
fold construction or the model. Every artifact hash is unchanged, and the results were
re-derived from the moved tree and matched.

## Reproducing v1 (only while the hashes above still match)

```bash
python train/random_forest/run.py --stage ablation    --output <new>/ablation
python train/random_forest/run.py --stage search      --output <new>/search --text-fields all
python train/random_forest/run.py --stage diagnostics --output <new>/diag  --source <new>/search
python train/random_forest/run.py --stage fit         --output <new>/val   --source <new>/search
python train/random_forest/run.py --stage final       --output <new>/final --source <new>/search
```

The fitted trees depend on `random_state=311` and the fold seeds, not on library versions,
but the TF-IDF vocabulary order and the `max_features` sampling do depend on the
scikit-learn version recorded in `final_manifest.json` (`provenance.packages`).

## What is still allowed on v1

- Loading `final_model.joblib` and reading, plotting or re-deriving its predictions.
- Exporting the model to a dependency-free form (300 trees of depth 6, about 0.5 MB of arrays).
- Reading any of its CSV/JSON outputs.

**The test split is closed for this artifact.** It has been read once and the result is above.
Reading it again for the same artifact would make that number a maximum over two draws rather
than an estimate, and reading it for a *new* artifact would have to be recorded as
`test uses = 2` in `experiment_log.md`, with the report stating that the number is no longer
independent. Anything that changes the model — retraining, a different tree count, different
preprocessing — produces such a new artifact.

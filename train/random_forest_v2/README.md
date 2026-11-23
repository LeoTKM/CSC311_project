# Random forest on the CSC311 survey: text pipeline

The training pipeline for the survey's three-class model, built around the four
free-text answers. Training uses scikit-learn; the tokenizer, the standard form and the
colour lexicon are plain Python with no fitted state, so an inference script can reuse
them without any library.

Input is `data/splits/train.csv` (606 responses, 202 respondents, three responses each).
Splits are respondent-grouped. The other two split files are not read by anything here,
and no decision in this package is based on them.

`PLAN.md` records what is done to the four short-answer questions and the training-split
numbers behind each choice. This file describes the package: what it computes, how it is
fitted, how it is scored and how to run it.

## 1. Layout

```
train/random_forest_v2/
  PLAN.md            what is done to the four answers, and the numbers behind it
  README.md          this file
  requirements.txt   pandas, numpy, scikit-learn
  text.py            tokenizer and standard form
  encode.py          survey parsing, colour lexicon, column lists
  features.py        TextPreprocessor: the 183-column feature matrix
  selection.py       grouped repeated CV, one-standard-error selection
  run.py             training script, stages cv / search / fit
  smoke_check.py     fast checks on the pipeline before training
  analysis/          the diagnostic scripts behind every decision in PLAN.md
```

The package is self-contained. It imports nothing outside this directory except the split
files and its three dependencies, and it writes only to its own output directory,
`random_forest_deliverables/v2_forest`.

## 2. Tokenisation

One rule set, applied identically when a vocabulary is fitted and when a single new
answer is scored.

Step 0: the reply goes through `clean()`, so `n/a`, `na`, `null` and friends stay empty.

| # | Step |
|---|---|
| 1 | Unicode NFKC, so full-width letters and commas become half-width |
| 2 | delete apostrophes, ASCII and typographic, so `don't` becomes `dont` |
| 3 | lowercase |
| 4 | every character outside `a-z 0-9` becomes a space |
| 5 | split on spaces |
| 6 | drop tokens shorter than 2 characters |
| 7 | drop tokens containing no letter |

The hyphen is a separator, so `hot-dog` and `hot dog` produce the same tokens. Every dash
character is consumed by step 4, so no separate dash handling is needed.

Anonymisation placeholders such as `[this sport]` are kept: step 4 turns them into
ordinary tokens. They are a deterministic product of the survey instrument and appear
identically when fitting and when scoring.

## 3. Standard form

Applied to the `feeling` answer only, before its vocabulary is counted, so that the
derivations of one emotion become one column instead of several thin ones.

```
step 1  irregular table, 17 words mapped to 6 standard forms
        energetic <- energetic, energized, energize, energy, energetized, energies
        anxious   <- anxious, anxiety
        happy     <- happy, happiness
        sad       <- sad, sadness
        tense     <- tense, tension
        relaxe    <- relaxed, relaxing, relaxation, relax

step 2  suffix chain, up to three passes, longest suffix first, at least 3 characters left
        ations ation ments ment ness dom ions ion ances ance ences ence
        ities ity ings ing edly ed ly es

step 3  restore a silent e: if the remainder is at least 4 characters, ends in a
        consonant, and that consonant is preceded by a vowel preceded by a consonant,
        append e
```

Examples: `excited`, `excitement` and `exciting` all become `excite`; `bored` and
`boring` become `bor`; `confused` and `confusion` become `confuse`. Opposites are never
merged, because `uninterested` carries no suffix of its own and stays separate.

## 4. Feature matrix

```
27  structured   ease and the three agreement answers as 0-1, the 21 multi-select
                 options as 0-1, payment_amount mean filled then log1p, payment_missing
 7  flags        q1_missing .. q7_missing
33  food         30 terms selected by chi-square, then food_non_answer,
                 food_no_terms, food_item_count
20  colour       the 17 fixed lexicon colours, then the same three columns
33  song         as food
63  feeling      60 terms selected after the standard form, then the same three columns
--
183 total
```

### The four text blocks

| Field | Encoding | K | pool |
|---|---|--:|---|
| food | chi-square selection on the fitting rows | 30 | `min_df` 3 |
| colour | fixed 17-column lexicon multi-hot | 17 | none |
| song | chi-square selection on the fitting rows | 30 | `min_df` 3 |
| feeling | chi-square selection after the standard form | 60 | `min_df` 3 |

Open fields: the candidate pool is every token with document frequency at least 3 in the
fitting rows, ranked by chi-square against the 3-class label, top K kept, each column a
binary presence. Food leaves 119 tokens in the pool, song about 115, feeling about 150.
No idf weighting, no stopword list, no n-grams.

Colour uses no label at all. A colour column is 1 when any word of its token group appears
in the reply, and the 17 columns are identical in every fold and at inference.

### Auxiliary columns

Every field gets three columns, all computed from the raw token list before any vocabulary
or lexicon lookup:

| Column | Value |
|---|---|
| `<field>_non_answer` | 1 when the reply is empty or normalises to a non-answer |
| `<field>_no_terms` | 1 when the reply has content but no token reaches the vocabulary or lexicon |
| `<field>_item_count` | number of tokens the reply produces |

They exist because a token that appears in no fitting row is indistinguishable, to the
model, from a token that appears once and is dropped by `min_df`; a reply built from unseen
words still leaves a trace.

### Imputation

Every column is imputed from the fitting rows: the median for Q1-Q4, the most frequent
value otherwise. Payment is parsed to one number, mean filled, then `log1p`. No column is
standardised: a forest does not need it.

## 5. Cross-validation and selection

```
grid          max_depth        4, 6, 8, 12, 16, None
              min_samples_leaf 1, 2, 3, 5, 8
              max_features     sqrt, log2, 0.3, 0.5, 1.0       150 combinations
folds         GroupKFold(5, shuffle=True, random_state=seed)
fold seeds    311, 312, 313
data          data/splits/train.csv only
fixed         n_estimators 300, criterion gini, bootstrap, class_weight None,
              oob_score False, n_jobs 1
```

The preprocessor is fitted inside each fold on that fold's training rows only, so the text
vocabularies and the imputation values never see the held-out rows.

Selection rule: rank by mean accuracy over the 15 grouped observations, form a
one-standard-error band at `best - spread / sqrt(15)`, and inside the band take the
simplest configuration (shallower depth, then larger leaves, then fewer candidate split
features).

Result:

| | value |
|---|---|
| best on the grid | 0.9087 at `max_depth` None, `max_features` log2, `min_samples_leaf` 2 |
| band width | 0.005302 |
| configurations inside the band | 19 of 150 |
| **selected** | **`max_depth` 8, `max_features` log2, `min_samples_leaf` 2** |
| selected CV accuracy | **0.9043** +- 0.0222 |
| selected in-fold training accuracy | 0.9414 |

For reference, the same pipeline at `max_depth` 6, `max_features` log2,
`min_samples_leaf` 3 scores 0.8972, which is below the band.

## 6. Final model

The model that ships refits the preprocessor and the forest on all 738 responses from the
two training-side splits, 246 respondents:

| | value |
|---|---|
| features | 183 |
| parameters | `max_depth` 8, `max_features` log2, `min_samples_leaf` 2, 300 trees |
| training accuracy | 0.9350 |
| forest size | 21,766 nodes, 11,033 leaves, maximum depth 8 |
| split on | 173 of the 183 columns |
| payment fill value | 256.92003, the mean of the 738 parsed amounts |

Artifacts: `v2_model.joblib` and `v2_manifest.json`. The manifest records the column
count, both vocabularies, the colour columns, the imputation value, the source-file
hashes and `"test_accessed": false`. Loading the joblib needs this directory on
`sys.path`, because the preprocessor class lives in `features.py`.

The test split has never been read by this package.

## 7. Stages

| Stage | What it does | Writes |
|---|---|---|
| `cv` | grouped repeated CV at one hyperparameter setting, three fold seeds | `cv_records.csv`, `cv_summary.json` |
| `ablation` | six text-field sets on a reduced grid, one fold seed | `ablation_folds.csv`, `ablation_summary.csv`, `ablation_summary.json` |
| `search` | the same CV over the 150-configuration grid, then 1-SE selection | `search_records.csv`, `search_summary.csv`, `selection.json` |
| `diagnostics` | tree-count curve, learning curve, permutation importance by block | `trees_curve.csv`, `learning_curve.csv`, `permutation_importance_by_block.csv`, `permutation_importance_by_column.csv`, `diagnostics.json` |
| `validation` | fit on the training split, score the validation split once, keep the training-split model | `validation_predictions.csv`, `validation_confusion.csv`, `validation_metrics.json`, `v2_model_train_only.joblib` |
| `final` | refit preprocessing and model on the training side of both splits | `v2_model.joblib`, `v2_manifest.json` |

`all` runs everything up to and including `validation`. `final` is deliberately separate:
it absorbs the validation rows, so it should be the last step and it cannot score them
afterwards.

Output goes to `random_forest_deliverables/v2_forest` unless `--out` says otherwise. `diagnostics` and
`validation` read the chosen parameters from `selection.json` in `--source` (default
`--out`), and fall back to the command-line values.

## 8. Running

```bash
pip install -r train/random_forest_v2/requirements.txt
python train/random_forest_v2/smoke_check.py         # fast pipeline checks, ~10 s
python train/random_forest_v2/run.py --stage cv      # grouped CV, ~15 s
python train/random_forest_v2/run.py --stage ablation
python train/random_forest_v2/run.py --stage search  # full grid, about 30 min
python train/random_forest_v2/run.py --stage diagnostics
python train/random_forest_v2/run.py --stage validation
python train/random_forest_v2/run.py --stage final   # only once the reading is taken
```

`--dry-run` on `cv`, `ablation` or `search` uses one fold seed; on the search stages it
also narrows the grid to four configurations.

Paths are resolved from each file's own location, so the scripts run from any working
directory.

## 9. Smoke check

`smoke_check.py` fits the preprocessor and asserts the parts that are easy to break
silently:

- the tokenizer rules, on hand-written examples
- fifteen standard-form cases, that opposites stay apart, and that the rule is idempotent
- that the colour block has exactly 17 columns, that an unknown colour sets none of them
  and is flagged by `colour_no_terms`, and that a refusal is flagged by `colour_non_answer`
- that the matrix is 183 columns with the block sizes above, all finite, with unique names
- that every fold holds out every row exactly once, with no respondent on both sides

It writes no file and trains no scored model.

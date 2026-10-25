# analysis

## Training and model comparison

`model_comparison.ipynb` is the executed companion to the exploration notebook.
It covers the report template's Appendix B guiding questions, trains/tunes all
three model families, and compares selected configurations on validation.
The local test set is not read and final train-plus-validation refitting is deferred.

- `model_experiments.py`: fold-local preprocessing and the reproducible experiment.
- `test_model_preprocessing.py`: tests for metadata exclusion, held-out transform
  immutability, unknown-value handling and training-only imputation.
- `out/models/`: complete CV tables, fold assignments, boundary checks, seed
  stability, validation predictions/metrics, confusion matrices and saved models.

Run `python analysis/model_experiments.py` or Run All in the notebook. Completed
results are reused only when source/data hashes and runtime versions match. Text
hashes normalize CRLF to LF, so Git line-ending conversion alone does not invalidate
the experiment; actual content changes still do. To
deliberately rerun changed experiments, supply a new `--output` directory.
Dependencies are in `analysis/requirements.txt`. Development `.joblib` models
require scikit-learn and are not the final restricted-import prediction submission.
Only load model artifacts you trust.

The current artifacts were regenerated on 2026-10-25 with the same grids, folds
and seeds, after the previous manifest's experiment-source hash could not be
matched to the committed source. `out/models/previous_run_manifest.json` preserves
that manifest; `out/models/reproduction_check.json` links the complete original
results in Git history and records the comparison. All selected configurations,
selected-model comparison metrics and validation predictions match. Some MLP
fold metrics differ, and tuning retries changed from four to five; the original
exact source is unavailable, so the cause of these differences is not established.
The report's retry count reflects the regenerated run. Fixed seeds and dependency
versions do not promise bitwise-identical numerical optimization on every platform.

The completed initial search plus boundary extensions evaluated 44 settings
(220 CV fits), followed by 20 additional seed fits and three validation refits.
Random forest led validation with 93.18% accuracy / 0.9317 macro-F1; softmax and
MLP both reached 90.91% accuracy. These are model-selection validation results,
not final-test estimates. See the notebook for uncertainty and overfitting analysis.

## Consolidated proposal notebook

Open `data_exploration.ipynb` and Run All using the project `.venv` kernel.
Install its dependencies with `python -m pip install -r analysis/requirements.txt`.
The executed notebook combines the original categorical/text analysis with Q1,
parsed Q12, encoder-aligned missingness, text-vector coverage and redundancy checks.
Tables and figures are exported to `analysis/out/proposal/`.

The canonical partition is `data/splits/`. Rebuild generated encodings with
`python analysis/rebuild_encoded.py`; this preserves raw files and checks row IDs
and labels before replacing generated artifacts. The notebook verifies manifests
and source hashes. Its missingness tables distinguish the encoder's policy from
the older semantic heuristics below. Structured Q1–Q7 missing values still require
training-fitted imputation in the future modelling pipeline.

All scripts read the **train split** by default (`data/splits/train.csv`) and print
their results to the terminal; the tables are also written as CSV.

## Files

| File | Role |
|---|---|
| `survey_common.py` | Definitions: question catalogue, Likert/multi-select encodings, CSV loading, data-quality rules, effect-size helpers (`describe`, `cramers_v`, `eta_sq`) |
| `categorical_stats.py` | Single-choice and multi-select analysis: computes the statistics, prints them, writes the CSV tables |
| `text_word_counts.py` | Short-answer analysis: normalisation, word counting, printing, CSV tables |
| `csv_output.py` | Writes the CSV tables |
| `make_splits.py` | Grouped train/val/test split |

## Running

```bash
python analysis/make_splits.py           # rewrite data/splits/
python analysis/categorical_stats.py     # Likert + multi-select statistics
python analysis/text_word_counts.py      # short-answer word counts
```

Options: `--data` (input CSV, default the train split), `--out-dir` (CSV output
directory, default `analysis/out/<input file stem>`), `--split-summary`; the word
counts also take `--min-len` (drop tokens shorter than N) and `--top` (how many
words to print per question).

## Output

`categorical_stats.py` prints, in order:

- the data-quality counts per feature (blank / invalid / valid)
- per single-choice question: n, mean, variance, sd (overall and per class),
  Cramér's V, eta², and the 5-level counts (Soccer/Baseball/Basketball)
- per multi-select question: the distribution of the "number of options selected",
  then one line per option with the count of respondents who selected it (overall and
  per class) and Cramér's V. No mean and no variance for multi-select: for a 0/1
  indicator the variance is fully determined by the mean, so only raw counts are
  reported
- a ranking of all mapped features by Cramér's V


`text_word_counts.py` prints, per question: row / blank / invalid / token counts and
the top words with counts, share, document frequency and per-class counts.

## CSV tables (`analysis/out/train/`)

| File | Content |
|---|---|
| `single_choice_stats.csv` | Per-class statistics of the three Likert questions |
| `multi_select_stats.csv` | Per-option selection counts of the three multi-select questions |
| `selection_count_distribution.csv` | Distribution of the "number of options selected", per class |
| `text_token_counts.csv` | Long table: question, word, count, share, document counts, per-class counts |
| `text_token_summary.csv` | One row per question: blanks, invalids, tokens, vocabulary size |
| `text_normalized.csv` | Per-row normalised text |

## Conventions

- Class order is Soccer / Baseball / Basketball.
- CSVs are read with `keep_default_na=False`, otherwise answers such as `None` /
  `N/A` become missing values.
- `id` is grouped: the 870 rows come from 290 respondents × 3 sports, so "n" counts
  rows, not independent people.

# analysis

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
- per multi-select question: the "number of options selected" statistics, then one
  line per option with its selection rate, per-class rates, per-class variances
  and Cramér's V
- a ranking of all mapped features by Cramér's V


`text_word_counts.py` prints, per question: row / blank / invalid / token counts and
the top words with counts, share, document frequency and per-class counts.

## CSV tables (`analysis/out/train/`)

| File | Content |
|---|---|
| `single_choice_stats.csv` | Per-class statistics of the three Likert questions |
| `multi_select_stats.csv` | Per-option statistics of the three multi-select questions |
| `selection_count_stats.csv` | Statistics of the "number of options selected" feature |
| `text_token_counts.csv` | Long table: question, word, count, share, document counts, per-class counts |
| `text_token_summary.csv` | One row per question: blanks, invalids, tokens, vocabulary size |
| `text_normalized.csv` | Per-row normalised text |

## Conventions

- Class order is Soccer / Baseball / Basketball.
- CSVs are read with `keep_default_na=False`, otherwise answers such as `None` /
  `N/A` become missing values.
- `id` is grouped: the 870 rows come from 290 respondents × 3 sports, so "n" counts
  rows, not independent people.

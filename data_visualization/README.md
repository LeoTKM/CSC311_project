# data_visualization

All figures are produced by `plot_histograms.py` and written to `figures/`. Each figure shows
**how each of the 3 classes (Soccer / Baseball / Basketball) is distributed on one feature of the
train split**, as a grouped histogram where every bar is labelled with the class count (and, for
multi-select questions, the within-class percentage).

> **Data policy: train split only.** The script defaults to `data/splits/train.csv`
> (606 rows / 202 `id` groups), because the split is grouped by `id` and val / test must not take
> part in any descriptive analysis or figure.

> **No tables inside the figures.** Per-class means, variances, standard deviations and the value
> distributions are printed to the terminal by `analysis/categorical_stats.py` and written as tidy
> CSVs under `analysis/out/train/`. The figures only keep one compact `mean:` / `variance:` line
> (order: Soccer / Baseball / Basketball) for readability.

**All text inside the figures is English.** Each title follows the format `Q<number> | <full question wording>`,
with a second line stating the question type, for example:

```
Q3 | Players must be comfortable with a high risk of injury.
Single choice, Likert 1-5 (ordinal score 1-5)   |   Soccer / Baseball / Basketball
mean: 3.83 / 2.92 / 3.85      variance: 0.85 / 1.02 / 0.82
```

## Written report

`plotting_report.md` is a hand-written English report covering the plotting
methods (grouped-bar histograms, the overview grid, style conventions) and the
feature encodings used to build the figures
(Likert ordinal scores, multi-hot options, selection counts, text normalisation,
missing-value handling, standardisation).

## Running

```bash
conda activate csc311
python data_visualization/plot_histograms.py                       # train split (default)
# optional: different input or output location
python data_visualization/plot_histograms.py --data data/splits/val.csv --out-dir data_visualization/figures
```

Dependencies: `pandas`, `matplotlib`, `Pillow` (all present in the `csc311` environment).

## Figure files

| File | Content |
|---|---|
| `figures/00_overview.png` | Overview: top row = the 3 single-choice questions on their 5 levels; bottom row = the 3 multi-select questions' "number of options selected" |
| `figures/01_single_Q2.png` | Q2 strategy dependence: 5-level ordinal distribution (grouped by class) |
| `figures/02_single_Q3.png` | Q3 injury risk: 5-level ordinal distribution (grouped by class) |
| `figures/03_single_Q4.png` | Q4 individual brilliance: 5-level ordinal distribution (grouped by class) |
| `figures/04_multi_Q5.png` | Q5 venue: how many respondents of each class ticked each of the 7 options, plus within-class share |
| `figures/05_multi_Q6.png` | Q6 season: 4 options |
| `figures/06_multi_Q7.png` | Q7 music type: 10 options |
| `figures/07_count_Q5.png` | Q5 venue: distribution of the number of options selected (0-7) |
| `figures/08_count_Q6.png` | Q6 season: distribution of the number of options selected (0-4) |
| `figures/09_count_Q7.png` | Q7 music type: distribution of the number of options selected (0-10) |

## Encoding conventions

- Single choice (Likert) -> ordinal 1-5: `Strongly disagree=1, Disagree=2, Neutral=3, Agree=4, Strongly agree=5`
- Multi-select -> one 0/1 variable per option (ticked = 1); bar height = number of respondents in that class who ticked it
- For a 0/1 variable the variance is `n/(n-1)*p*(1-p)`, i.e. fully determined by the mean; the figure footnote says so
- Blank answers are excluded from the statistics; the actual per-class n is shown in the legend
- The CSVs are read with `keep_default_na=False`; otherwise real answers such as `None` / `N/A` would be treated as missing

## Companion statistics

`analysis/categorical_stats.py` prints, to the terminal, the dataset split, a data-quality table
(blank / invalid / valid for all 12 features), per-class `n / mean / variance / sd` with the
5-level distribution for every single-choice question, per-class selection rates and variances
with chi-square and Cramér's V for every multi-select option, and a feature ranking by effect size.
The same numbers are written as tidy CSVs (`single_choice_stats.csv`, `multi_select_stats.csv`,
`selection_count_stats.csv`) under `analysis/out/train/`.

Word-level statistics for the five short-answer questions (also train-only) are printed by
`analysis/text_word_counts.py`, with the tables written to `text_token_counts.csv`,
`text_token_summary.csv` and `text_normalized.csv`.

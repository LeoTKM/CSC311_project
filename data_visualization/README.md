# data_visualization

All figures are produced by `plot_histograms.py` and `plot_payment.py` and written to `figures/`.
Each figure shows **how each of the 3 classes (Soccer / Baseball / Basketball) is distributed on
one feature of the train split**, as a grouped histogram where every bar is labelled with the class
count (and, for multi-select questions, the within-class percentage).

> **Data policy: train split only.** `plot_histograms.py` defaults to `data/splits/train.csv`
> (606 rows / 202 `id` groups) and `plot_payment.py` to `data/encoded/train_encoded.csv`, because
> the split is grouped by `id` and val / test must not take part in any descriptive analysis or
> figure. Both scripts refuse to read a validation or test file.

> Per-class means, variances, standard deviations and the value
> distributions are printed to the terminal by `analysis/categorical_stats.py` and written as tidy
> CSVs under `analysis/out/train/`. The figures only keep one compact `mean:` / `variance:` line
> (order: Soccer / Baseball / Basketball) for readability.

Each title follows the format `Q<number> | <full question wording>`,
with a second line stating the question type, for example:

```
Q3 | Players must be comfortable with a high risk of injury.
Single choice, Likert 1-5 (ordinal score 1-5)   |   Soccer / Baseball / Basketball
mean: 3.83 / 2.92 / 3.85      variance: 0.85 / 1.02 / 0.82
```

## Written report

`plotting_report.md` is a report covering the plotting
methods (grouped-bar histograms, the Q12 amount buckets, the overview grid, style
conventions) and the feature encodings used to build the figures
(Likert ordinal scores, multi-hot options, selection counts, the payment parser,
text normalisation, missing-value handling, standardisation).

## Running

```bash
python data_visualization/plot_histograms.py                       # figures 00-09, train split (default)
python data_visualization/plot_payment.py                          # figures 10-11, encoded train split (default)
# optional: different input or output location
python data_visualization/plot_histograms.py --data data/splits/train.csv --out-dir data_visualization/figures
python data_visualization/plot_payment.py --data data/encoded/train_encoded.csv --out-dir data_visualization/figures
```

`plot_payment.py` needs the encoded train split, so run
`python data_preprocessing/encode_all.py --input data/class_dataset.csv` first if
`data/encoded/train_encoded.csv` is missing.

Dependencies: `pandas`, `matplotlib`, `Pillow` (all present in the `csc311` environment).

## Figure files

| File | Content |
|---|---|
| `figures/00_overview.png` | Overview: top row = the 3 single-choice questions on their 5 levels; bottom row = the 3 multi-select questions' option distributions (7 / 4 / 10 options) |
| `figures/01_single_Q2.png` | Q2 strategy dependence: 5-level ordinal distribution (grouped by class) |
| `figures/02_single_Q3.png` | Q3 injury risk: 5-level ordinal distribution (grouped by class) |
| `figures/03_single_Q4.png` | Q4 individual brilliance: 5-level ordinal distribution (grouped by class) |
| `figures/04_multi_Q5.png` | Q5 venue: how many respondents of each class ticked each of the 7 options, plus within-class share |
| `figures/05_multi_Q6.png` | Q6 season: 4 options |
| `figures/06_multi_Q7.png` | Q7 music type: 10 options |
| `figures/07_count_Q5.png` | Q5 venue: distribution of the number of options selected (0-7) |
| `figures/08_count_Q6.png` | Q6 season: distribution of the number of options selected (0-4) |
| `figures/09_count_Q7.png` | Q7 music type: distribution of the number of options selected (0-10) |
| `figures/10_payment_Q12.png` | Q12 willingness to pay: 8 CAD amount buckets plus a "not stated" bucket, grouped by class, with per-class missing count, fill value, IQR, min, max, mean and variance in the title |
| `figures/11_payment_Q12_trimmed.png` | The same Q12 data after dropping, inside each class, every amount above that class's `Q3 + 1.5 x IQR`. Three panels (one per class), each with its own continuous x axis, equal 50 CAD bins and a shared count axis. Everything from 0 up to the cut is kept, so the `$0` answers stay in; each panel reports its cut, how many were dropped and the mean of what is left |

## Encoding conventions

- Single choice (Likert) -> ordinal 1-5: `Strongly disagree=1, Disagree=2, Neutral=3, Agree=4, Strongly agree=5`
- Multi-select -> one 0/1 variable per option (ticked = 1); bar height = number of respondents in that class who ticked it
- For a 0/1 variable the variance is `n/(n-1)*p*(1-p)`, i.e. fully determined by the mean; the figure footnote says so
- Q12 (willingness to pay) is a short answer but is plotted from the two numeric columns the encoder produces, `payment_amount` (parsed CAD amount, missing replies filled with the train mean) and `payment_missing` (1 = no parseable amount). Amounts are read as CAD: currency notation is stripped without any conversion, so the dollar sign, `dollars` and `bucks` all mean CAD
- The trimmed Q12 figure drops only the upper tail: within each class, amounts above `Q3 + 1.5 x IQR` are removed, and every amount from 0 up to that cut is kept, so a `$0` answer is never dropped. Replies with no amount have no position on a numeric axis, so they are stated as text in each panel instead of drawn. The means in that figure are computed on what survives the drop
- Figure 10 keeps categorical amount buckets because its bins are not equally wide, so a numeric axis would draw wrong widths. Figure 11 can use a real continuous axis only because trimming reduces the within-class skew from about 7 to about 0.9 and shrinks the range to 0-625 / 0-200 / 0-400, which allows equal 50 CAD bins
- Blank answers are excluded from the statistics; the actual per-class n is shown in the legend
- The CSVs are read with `keep_default_na=False`; otherwise real answers such as `None` / `N/A` would be treated as missing

## Companion statistics

`analysis/categorical_stats.py` prints, to the terminal, the dataset split, a data-quality table
(blank / invalid / valid for all 12 features), per-class `n / mean / variance / sd` with the
5-level distribution for every single-choice question, per-class selection counts with Cramér's V
for every multi-select option, and a feature ranking by effect size. The same numbers are written
as tidy CSVs (`single_choice_stats.csv`, `multi_select_stats.csv`,
`selection_count_distribution.csv`) under `analysis/out/train/`.

Word-level statistics for the five short-answer questions (also train-only) are printed by
`analysis/text_word_counts.py`, with the tables written to `text_token_counts.csv`,
`text_token_summary.csv` and `text_normalized.csv`.

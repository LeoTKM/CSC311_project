# data/splits

Train / val / test split of the dataset, produced by `analysis/make_splits.py` (committed so
the whole team uses the exact same split).

## Files

| File | Content |
|---|---|
| `train.csv` | 606 rows (202 `id` groups), 202 rows per class |
| `val.csv` | 132 rows (44 `id` groups), 44 rows per class |
| `test.csv` | 132 rows (44 `id` groups), 44 rows per class |
| `id_to_split.csv` | Manifest: which split each `id` belongs to, its row count and labels |
| `split_summary.json` | Sizes and rows per class for each split |

## How it is built

- **Grouped by `id`**: each `id` owns 3 rows (Soccer / Baseball / Basketball) and all of them
  stay in the same split.
- **70 / 15 / 15** over groups: 202 train / 44 val / 44 test.
- Each split ends up 1:1:1 across the three classes, because every `id` covers all classes.
- The ids are sorted, then shuffled with `numpy.random.default_rng(seed)` using seed 311, so the
  same seed always reproduces the same split.

## Reproducing

```bash

python analysis/make_splits.py                      # seed 311, 0.15 / 0.15
python analysis/make_splits.py --seed 7 --test-size 0.1 --val-size 0.1
python analysis/make_splits.py --from-manifest data/splits/id_to_split.csv   # reuse the manifest
```

## Usage rules

- Feature engineering, tuning and statistics may only look at `train.csv`.
- `val.csv` is for model selection; `test.csv` is used once, for the final evaluation.
- `data/class_dataset_tfidf.csv` was fitted on all 870 rows, so it leaks vocabulary statistics;
  modelling with it requires re-fitting on train.

# Encode the entire survey

## Install

```bash
python -m pip install pandas numpy scikit-learn joblib
```

## Run once for all three existing splits

```bash
python data_preprocessing/encode_all.py \
  --train data/splits/train.csv \
  --validation data/splits/validation.csv \
  --test data/splits/test.csv
```

Adjust filenames to match your raw split files. The script rejects overlapping
respondent IDs and learns the vocabulary, IDF weights and payment mean from
training data only. Validation/test reuse the saved values.

**Alternatively, create a new split from the full CSV:**

```bash
python data_preprocessing/encode_all.py --input data/class_dataset.csv
```

This creates a NEW respondent-grouped split with seed 311, approximately 70/15/15:
606/132/132 rows for the supplied dataset. It does not use existing split files.

## Output

Files go in `data/encoded/`. Add `--output-dir data` to save directly in `data/`.

- `train_encoded.csv`, `validation_encoded.csv`, `test_encoded.csv`
- `preprocessor.joblib`: reusable fitted encoder
- `split_manifest.csv` and `encoding_summary.json`
- A payment audit CSV for each split

Existing outputs are never overwritten. Choose a new output directory to rerun.

## What it does

| Inputs | Encoding |
| --- | --- |
| Ease | Divide the original 0-10 rating by 10; higher means easier |
| Three agreement questions | Map ordered responses to 0, 0.25, 0.5, 0.75, 1 |
| Location / season / music type | 7 / 4 / 10 multi-hot features |
| Food / colour / song / feeling | Up to 50 / 20 / 50 / 50 TF-IDF features, plus four missing indicators |
| Payment | Extract amounts; use range midpoints and expand `1k`; impute missing/ambiguous amounts with the training mean, plus a missing indicator |

Text retains A-Z-only words appearing in at least two training responses. Chinese
and unseen words are ignored without dropping rows. Missing text gives zero
vectors. Payment assumes one currency; numeric zeros remain zero.

Missing single-choice and multi-select cells remain blank, as in the previous
scripts. Impute these using training data before models that require complete
inputs. Payment stays in raw monetary units; fit any scaling on training only.

The supplied split produces **201 feature columns**, plus `id` and `label`.
Exclude `id` and `label` from model inputs; `label` is the prediction target.
This script encodes data; it does not train or evaluate classifiers.

## Future responses

```bash
python data_preprocessing/encode_all.py \
  --transform data/new_responses.csv \
  --model data/encoded/preprocessor.joblib \
  --output data/encoded/new_responses_encoded.csv
```

Use raw question columns, not already encoded inputs. In transform mode, `id`
and `label` are optional. Only load preprocessor files you trust.

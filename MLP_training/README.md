# MLP training and section 3.4 submission

The prepared submission candidate is the existing PyTorch-trained MLP, exported
to NumPy arrays. Architecture: 208 inputs, 128-unit ReLU hidden layer, 3 outputs.
Training used dropout 0.5, Adam, learning rate 0.00011342488040637891,
weight decay 0.00004790869601049712, batch size 32 and 75 epochs.
The weights are unchanged by submission packaging.

## Submit to MarkUs

Upload **both files** from `MLP_training/submission/` to `project_prediction`:

- `pred.py`
- `mlp_params.npz`

Keep them in the same directory. `release/mlp_submission.zip` contains exactly
these two files at its root for convenient transfer; extract it and upload the
two files unless MarkUs explicitly accepts ZIP uploads. Do not upload training
scripts, datasets, reports, or the release verification files.

`pred.py` imports only standard Python libraries, NumPy and pandas. It loads
arrays with `allow_pickle=False`; it never imports sklearn, torch or tensorflow,
trains a model, downloads files, or prompts for input. Python 3.10+ is required.

```python
from pred import predict_all
labels = predict_all('unlabelled.csv')
```

The function accepts original survey headers or the supported short headers,
does not require `id` or `label`, and preserves row order. It returns strings
`Soccer`, `Baseball`, or `Basketball`. The CLI prints one label per row:

```sh
python MLP_training/submission/pred.py path/to/unlabelled.csv
```

## Verification and packaging

From the repository root, with NumPy and pandas installed:

```sh
python MLP_training/package_submission.py
```

This validates imports, frozen weight identity, prediction parity with the
archived test predictions, label-free/reordered CSVs, missing values, empty
inputs and 60-row runtime in a fresh isolated Python process with prohibited
ML imports blocked. It produces a two-file ZIP, a hash manifest and a verification
record under `MLP_training/release/`. The archive is checked after extraction.
Replaying archived predictions is a regression check, not a new independent
evaluation or an opportunity to tune against test results.

## Evaluation and selection

| Model fit | Evaluation | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Training only | Held-out validation, 132 rows | 92.42% | 0.9238 |
| Train + validation; submission weights | Local test, 132 rows | 93.94% | 0.9396 |

The saved tuning summary in `config.yaml` reports 90.92% +/- 1.86% grouped
5-fold CV accuracy and 0.9089 macro-F1, with trial 19 selected. The original
trial records are missing: see `TUNING_EVIDENCE.md`. These CV figures are
recorded claims, not independently reproduced search results.

**Do not evaluate the submission model on val.csv and call that held-out
validation accuracy:** these 132 rows were included in its fit. The train-only
model and its validation predictions are in `out/final_train/`.

The MLP is the prepared deployment candidate because its export is complete and
its validation performance is competitive. The local test results for MLP,
softmax and forest have already been compared; any selection informed by that
ranking makes this a selection-exposed holdout, not an untouched estimate for
the chosen winner. The two-response MLP advantage over softmax is not decisive.
The course's hidden TA/instructor test is separate and may differ in distribution.
The historical 88.6-98.5% bootstrap interval does not account for model selection
or that distribution shift.

## Training (separate from submission)

```sh
pip install -r MLP_training/requirements.txt
python MLP_training/optuna_search.py --out MLP_training/out/new_search --no-update-config
python MLP_training/train_mlp.py --out MLP_training/out/new_train
python MLP_training/train_mlp.py --train-on train+val --out MLP_training/out/new_trainval
```

These commands create new experiments. They do not recover the original Optuna
records and must not silently replace the reviewed submission weights.
Training dependencies may include PyTorch and scikit-learn; inference does not.

## Other model work

Random forests remain development artifacts: their sklearn/joblib pipelines
and traversal benchmarks are not standalone section 3.4 submissions. A forest
export is unnecessary for this MLP bundle and would be separate work if the
team changes its final model.

Softmax PR #7 has its own alternative export. Its adjacent parameters are
train-only; the refit parameters are `train/softmax/results/local_test/softmax_params.json`.
Do not mix those files with this MLP bundle. PR #7 is merged into main and included in this branch.

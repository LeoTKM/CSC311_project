# MLP_training

PyTorch MLP for the 3-class sport classifier. Python 3.10+.

## Model

208 features -> Linear(128) -> ReLU -> Dropout(0.5) -> Linear(3) -> softmax.
Adam, lr 1.13e-4, weight decay 4.8e-5, batch 32, 75 epochs.
Chosen by Optuna (60 trials) with grouped 5-fold CV on `train.csv` only.

## Results

| Model | Evaluated on | Accuracy | Macro-F1 |
|---|---|---|---|
| CV during tuning | 5 folds of train | 90.92% +/- 1.86 | 0.9089 |
| Trained on train | **validation (132 rows, unseen)** | **92.42%** | 0.9238 |
| Trained on train+val (submitted) | **test (132 rows, run once)** | **93.94%** (95% CI 88.6–98.5) | 0.9396 |

## Folders

| Path | What |
|---|---|
| `submission/` | **What we hand in:** `pred.py` + `mlp_params.npz` (413 KB, train+val model) |
| `out/final_train/` | Model trained on train only + its validation predictions, confusion matrix, metrics |
| `out/final_trainval/` | Model trained on train+val (same as the one in `submission/`) |
| `out/test_eval/` | One-time test evaluation of the submission model |
| `out/optuna/` | Optuna study, all trials, best parameters |
| `config.yaml` | Search space; `final_model:` holds the tuned hyperparameters |

## Commands (from the repository root)

```bash
pip install -r MLP_training/requirements.txt            # torch, optuna, pyyaml (training only)

python MLP_training/optuna_search.py                     # re-tune (~25 min), updates config.yaml
python MLP_training/train_mlp.py --out MLP_training/out/final_train                       # train -> val
python MLP_training/train_mlp.py --train-on train+val --out MLP_training/out/final_trainval
cp MLP_training/out/final_trainval/mlp_final.npz MLP_training/submission/mlp_params.npz

python MLP_training/submission/pred.py data/splits/val.csv   # CSV inference (+ accuracy if labelled)
python MLP_training/submission/pred.py                       # answer 12 questions, live prediction
```

## pred.py

Imports only the standard library, numpy and pandas. `predict_all(filename)` returns a list of
labels. It rebuilds the team encoder in NumPy (verified identical to the sklearn pipeline on
every train/val row). The live-question mode is the last line of the file; comment it out to
disable it.

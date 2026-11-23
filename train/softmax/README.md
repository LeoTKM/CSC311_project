# Softmax training

Multiclass logistic regression with L2 regularization and L-BFGS. The search
starts with six C values for the proposal's feature representation, then compares
larger word vocabularies, normalization, character n-grams and block weighting.
It tests a fixed colour lexicon, text coverage indicators and field removal on the
leading representation. Accuracy selects the winner, followed by macro-F1, log
loss and mean feature count. The runs reuse the canonical respondent-grouped split.

```sh
python -m pip install -r train/softmax/requirements.txt
python train/softmax/train.py --out /path/to/new/softmax-results
```

The final search uses five GroupKFold folds for each of seeds 311, 312 and 313.
The search and each preprocessing fit use only the fitting respondents. Separately,
five outer folds with seed 811 evaluate the whole selection procedure with four inner
folds per outer fold. This estimates how the search procedure transfers when the
validation respondents are unavailable. The selected configuration is frozen before
the canonical validation file is read. The result files distinguish tuning CV,
nested CV and validation. `nested_cv.json` also reports synthetic missing-song,
unknown-song and formatting checks on outer held-out rows.

The runner writes a standalone `pred.py` and `softmax_params.json` into the output
folder. Keep both files together and use `predict_all(filename)` on a raw survey CSV.
The inference script needs Python 3.10+ and NumPy. Its latency benchmark uses 60 rows
from training without labels. The Python process and module imports are ready when
timing begins. The first-call trials clear the in-process parameter cache and time
CSV reading, parameter loading, preprocessing and prediction. OS file caching is
left in its normal state. The selected model is fit on train. Validation is model
comparison data and the local test split is reserved for a separately agreed final
evaluation protocol.

## Results

| Model | Accuracy on the same 132 validation rows | Macro-F1 |
|---|---:|---:|
| Softmax, proposal baseline | 90.91% (120 correct) | 0.9089 |
| Softmax, selected | 91.67% (121 correct) | 0.9161 |
| PyTorch MLP | 92.42% (122 correct) | 0.9238 |
| Random forest v2 | 92.42% (122 correct) | 0.9238 |
| Random forest, proposal configuration | 93.18% (123 correct) | 0.9317 |

The external rows are taken from the repository's train-only validation artifacts
for each model. One response is 0.76 percentage points. The MLP's published
93.94% comes from a train+validation refit scored on local test, a separate split.

The selected softmax uses C=0.3 and 790 features: structured inputs, normalized
word TF-IDF, 3–5-character TF-IDF within words and a 17-colour lexicon. It
converged in 34 L-BFGS iterations on the full training split. Repeated CV accuracy
was 90.76% versus 89.55% for the fixed C=1 baseline under the same 15 folds.
Nested CV gave 90.43% for the selection procedure and 89.27% for the fixed
baseline. The descriptive paired respondent bootstrap interval for the nested
accuracy difference spans −0.50 to +2.97 percentage points. The validation
log loss was 0.2502 versus 0.2473 for the baseline. The synthetic unseen-song
check declined slightly relative to the baseline, and dropping the entire song
block substantially reduced tuning CV accuracy.

The export reproduces all 738 training and validation labels from scikit-learn.
On the development machine, the 25 first-call trials on 60 raw responses took
0.065 s at the median and 0.089 s at the maximum. The two inference files total
80,486 bytes. See `results/metrics.json` for exact metrics, `results/nested_cv.json`
for outer-fold and stress results, and `results/search_summary.csv` for the
candidate comparison.

A second benchmark with default NumPy thread settings took 0.113 s at most on
60 raw rows and 0.310 s at most on 60 rows with artificially lengthened text.
See `results/inference_default_environment.json`.

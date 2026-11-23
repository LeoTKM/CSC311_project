#!/usr/bin/env python3
"""NumPy-only inference for the exported MLP (mlp_final.npz). Imports nothing but numpy.

Two entry points:
  * predict_proba(params, X) for an already-built feature matrix in the model's
    input order (`feature_columns`, 208 columns), raw = before imputation/log/scaling.
  * prepare(params, X_raw) applies the stored imputation, log1p and standardization;
    predict_proba_from_raw(params, X_raw) does prepare + forward.

    params = load('MLP_training/out/final/mlp_final.npz')
    probs  = predict_proba(params, X)      # (n, 3), columns in params['classes'] order
    labels = predict(params, X)            # array of class names

Building the 208 raw features from survey answers (TF-IDF with the stored vocabulary
and idf, multi-hot, payment parsing) still uses the team encoder today; a NumPy-only
version of that step is the remaining piece for a torch/sklearn-free submission.
"""
import json
import sys

import numpy as np


def load(path):
    with np.load(path, allow_pickle=False) as f:
        params = {k: f[k] for k in f.files}
    params['meta'] = json.loads(str(params.pop('meta_json')))
    return params


def prepare(params, X_raw):
    """Fill NaNs, apply log1p to payment, standardize: raw 208 features -> model input."""
    X = np.array(X_raw, dtype=np.float64, copy=True)
    nan = np.isnan(X)
    X[nan] = np.broadcast_to(params['fill_values'], X.shape)[nan]
    m = params['log1p_mask']
    X[:, m] = np.log1p(X[:, m])
    return (X - params['scale_mean']) / params['scale_std']


def predict_proba(params, X):
    """Forward pass on model-ready inputs (output of prepare)."""
    h = np.asarray(X, dtype=np.float64)
    n = int(params['n_linear'])
    for i in range(n - 1):
        h = np.maximum(0.0, h @ params[f'W{i}'] + params[f'b{i}'])
    logits = h @ params[f'W{n - 1}'] + params[f'b{n - 1}']
    logits -= logits.max(axis=1, keepdims=True)          # numerically stable softmax
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def predict(params, X):
    return params['classes'][predict_proba(params, X).argmax(axis=1)]


def predict_proba_from_raw(params, X_raw):
    return predict_proba(params, prepare(params, X_raw))


if __name__ == '__main__':
    p = load(sys.argv[1] if len(sys.argv) > 1 else 'MLP_training/out/final/mlp_final.npz')
    shapes = {k: v.shape for k, v in p.items() if k.startswith(('W', 'b'))}
    print('classes:', list(p['classes']), '| features:', len(p['feature_columns']))
    print('layers:', shapes)
    print('meta:', p['meta'])

"""Shared pieces for the PyTorch MLP: config, data/preprocessing, model, training loop, export.

Preprocessing is the team's SurveyPreprocessor from analysis/model_experiments.py,
so the MLP sees exactly the 208 features described in the proposal, and every
fitted value (vocabulary, IDF, imputation, scaler) comes from training rows only.
"""
from __future__ import annotations

import json
import random
import sys

if sys.version_info < (3, 10):
    raise RuntimeError('MLP_training needs Python 3.10 or later.')
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'data_preprocessing'))
from model_experiments import SurveyPreprocessor, sanitize  # noqa: E402  (team preprocessing)
from encode_all import (TOKEN_PATTERN, encode_all, clean, EASE_COLUMN,  # noqa: E402
                        AGREEMENT_COLUMNS, SCHEMA)


# ---------------------------------------------------------------- config
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_layers(value) -> list[int]:
    """"64,32" -> [64, 32]; also accepts a list."""
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(v) for v in str(value).split(',') if v.strip()]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(name='auto'):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


# ---------------------------------------------------------------- data
def read_split(path) -> pd.DataFrame:
    # Team convention (encode_all.read_csv / model_experiments): read everything as str and
    # keep_default_na=False so answers like "None" / "N/A" are not turned into NaN.
    return pd.read_csv(ROOT / path, dtype=str, keep_default_na=False, encoding='utf-8-sig')


def encode_labels(labels, classes) -> np.ndarray:
    index = {c: i for i, c in enumerate(classes)}
    unknown = set(labels) - set(index)
    if unknown:
        raise ValueError(f'Unknown labels: {unknown}')
    return np.array([index[v] for v in labels], dtype=np.int64)


def fit_transform(train_raw, *others):
    """Fit SurveyPreprocessor on train_raw only, transform train and any held-out frames."""
    pre = SurveyPreprocessor().fit(train_raw)
    out = [pre.transform(train_raw).astype(np.float32)]
    out += [pre.transform(df).astype(np.float32) for df in others]
    return pre, out


def raw_features(pre: SurveyPreprocessor, df) -> np.ndarray:
    """The 208 features BEFORE imputation / log1p / scaling (NaN where missing).

    Same steps as SurveyPreprocessor._frame up to the fillna, so
    predict_numpy.prepare(raw_features(pre, df)) == pre.transform(df).
    """
    raw = sanitize(df)
    encoded, _, _ = encode_all(raw, pre.encoder_)
    feats = encoded.drop(columns=['id', 'label'], errors='ignore').copy()
    for q, col in enumerate([EASE_COLUMN, *AGREEMENT_COLUMNS, *SCHEMA], 1):
        feats[f'q{q}_missing'] = raw[col].map(clean).eq('').astype(float)
    return feats.reindex(columns=pre.columns_).to_numpy(dtype=np.float64)


# ---------------------------------------------------------------- model
class MLP(nn.Module):
    """input -> [Linear -> ReLU -> Dropout] x L -> Linear (logits).

    forward() returns logits for CrossEntropyLoss; predict_proba() applies softmax.
    """

    def __init__(self, input_dim: int, hidden_layers: list[int], num_classes: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        layers, prev = [], input_dim
        for width in hidden_layers:
            layers += [nn.Linear(prev, width), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=1)


def build_optimizer(name, params, lr, weight_decay):
    if name == 'adam':     # L2 penalty added to the gradient (closest to sklearn's alpha)
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == 'adamw':    # decoupled weight decay
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f'Unsupported optimizer: {name}')


# ---------------------------------------------------------------- training
@torch.no_grad()
def evaluate(model, X, y, criterion):
    model.eval()
    logits = model(X)
    loss = criterion(logits, y).item()
    pred = logits.argmax(1).cpu().numpy()
    true = y.cpu().numpy()
    return {'loss': loss, 'accuracy': accuracy_score(true, pred),
            'macro_f1': f1_score(true, pred, average='macro', zero_division=0)}, pred


def train_model(X_tr, y_tr, hp, epochs, seed, device, X_ev=None, y_ev=None):
    """Train for a fixed number of epochs. If eval data is given, return per-epoch curves."""
    set_seed(seed)
    X_tr = torch.as_tensor(X_tr, device=device)
    y_tr = torch.as_tensor(y_tr, device=device)
    if X_ev is not None:
        X_ev = torch.as_tensor(X_ev, device=device)
        y_ev = torch.as_tensor(y_ev, device=device)

    model = MLP(X_tr.shape[1], parse_layers(hp['hidden_layers']), 3, hp['dropout']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(hp['optimizer'], model.parameters(), hp['learning_rate'],
                                hp['weight_decay'])
    gen = torch.Generator(device='cpu').manual_seed(seed)
    n, bs = len(X_tr), int(hp['batch_size'])
    curves = {'train_loss': [], 'eval_loss': [], 'eval_accuracy': [], 'eval_macro_f1': []}

    for _ in range(epochs):
        model.train()
        order = torch.randperm(n, generator=gen).to(device)
        total = 0.0
        for start in range(0, n, bs):
            idx = order[start:start + bs]
            optimizer.zero_grad()
            loss = criterion(model(X_tr[idx]), y_tr[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        curves['train_loss'].append(total / n)
        if X_ev is not None:
            m, _ = evaluate(model, X_ev, y_ev, criterion)
            curves['eval_loss'].append(m['loss'])
            curves['eval_accuracy'].append(m['accuracy'])
            curves['eval_macro_f1'].append(m['macro_f1'])
    return model, curves


def confusion(true, pred, classes):
    cm = confusion_matrix(true, pred, labels=list(range(len(classes))))
    return pd.DataFrame(cm, index=[f'true_{c}' for c in classes],
                        columns=[f'pred_{c}' for c in classes])


# ---------------------------------------------------------------- export
def preprocessing_payload(pre: SurveyPreprocessor) -> dict:
    """Everything needed to rebuild the 208 inputs from raw answers, as plain Python
    types/lists (no sklearn objects), so torch.load(..., weights_only=True) works.

    Pipeline order at inference:
      1. structured encoding (encode_all.encode_structured: Q1/10, Likert -> 0..1, multi-hot)
      2. text TF-IDF per field with `text_encoders` (sublinear tf, smoothed idf, L2 norm)
         + `<field>_text_missing`; payment parsing, fill with `payment_fill_value`
         + `payment_missing`
      3. `q1..q7_missing` flags (structured answer blank/invalid)
      4. fill remaining NaNs with `fill_values`
      5. payment_amount = log1p(payment_amount)
      6. (x - scale_mean) / scale_std on `scale_columns`
      7. order columns as `feature_columns`
    """
    text = pre.encoder_['text_encoder']
    text_encoders = {}
    for field, vec in text['vectorizers'].items():
        if vec is None:
            text_encoders[field] = None
            continue
        vocab = vec.get_feature_names_out().tolist()
        text_encoders[field] = {
            'vocabulary': vocab,
            'idf': vec.idf_.astype(float).tolist(),
            'token_pattern': TOKEN_PATTERN, 'lowercase': True,
            'sublinear_tf': True, 'norm': 'l2',
        }
    return {
        'feature_columns': list(pre.columns_),
        'fill_values': {k: float(v) for k, v in pre.fill_.items()},
        'log1p_columns': ['payment_amount'],
        'scale_columns': list(pre.scale_cols_),
        'scale_mean': pre.scaler_.mean_.astype(float).tolist(),
        'scale_std': pre.scaler_.scale_.astype(float).tolist(),
        'payment_fill_value': float(text['payment_fill_value']),
        'text_encoders': text_encoders,
    }


def export_checkpoint(path, model: MLP, hp, classes, pre, extra=None):
    """Save one .pt with weights, architecture and the fitted preprocessing numbers."""
    layers = parse_layers(hp['hidden_layers'])
    payload = {
        'format': 'csc311_mlp_v1',
        'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'architecture': {
            'input_dim': int(model.net[0].in_features),
            'hidden_layers': layers,
            'num_classes': len(classes),
            'activation': 'relu',
            'dropout': float(hp['dropout']),
            'output_activation': 'softmax',   # apply to logits at inference
        },
        'classes': list(classes),
        'hyperparameters': {k: (v if isinstance(v, (int, float, str)) else str(v))
                            for k, v in hp.items()},
        'preprocessing': preprocessing_payload(pre),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return payload


def export_numpy(path, model: MLP, hp, classes, pre, extra_meta=None):
    """Save a NumPy-only .npz (load with np.load(path) -- no pickle, no torch needed).

    Forward pass:  h = x;  for i in range(n_hidden): h = max(0, h @ W{i} + b{i})
                   logits = h @ W{n_hidden} + b{n_hidden};  probs = softmax(logits)
    W{i} has shape (in, out) so it multiplies row vectors directly (torch weights transposed).
    Dropout is inactive at inference, so it does not appear.

    Preprocessing arrays are aligned with `feature_columns` (208 names, model input order).
    """
    pp = preprocessing_payload(pre)
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    arrays = {}
    for i, lin in enumerate(linears):
        arrays[f'W{i}'] = lin.weight.detach().cpu().numpy().T.astype(np.float64)
        arrays[f'b{i}'] = lin.bias.detach().cpu().numpy().astype(np.float64)
    cols = pp['feature_columns']
    scale_set = dict(zip(pp['scale_columns'], zip(pp['scale_mean'], pp['scale_std'])))
    arrays.update({
        'n_linear': np.array(len(linears)),
        'hidden_layers': np.array(parse_layers(hp['hidden_layers']), dtype=np.int64),
        'classes': np.array(classes),
        'feature_columns': np.array(cols),
        # step 4: NaN fill value per feature
        'fill_values': np.array([pp['fill_values'][c] for c in cols], dtype=np.float64),
        # step 5: 1 where log1p is applied
        'log1p_mask': np.array([c in pp['log1p_columns'] for c in cols]),
        # step 6: per-feature mean/std; mean 0 / std 1 for unscaled columns
        'scale_mean': np.array([scale_set.get(c, (0.0, 1.0))[0] for c in cols], dtype=np.float64),
        'scale_std': np.array([scale_set.get(c, (0.0, 1.0))[1] for c in cols], dtype=np.float64),
        'payment_fill_value': np.array(pp['payment_fill_value']),
        'text_token_pattern': np.array(TOKEN_PATTERN),
    })
    for field, enc in pp['text_encoders'].items():
        if enc is not None:
            arrays[f'text_{field}_vocabulary'] = np.array(enc['vocabulary'])
            arrays[f'text_{field}_idf'] = np.array(enc['idf'], dtype=np.float64)
    meta = {'format': 'csc311_mlp_numpy_v1', 'activation': 'relu', 'output': 'softmax',
            'hyperparameters': {k: str(v) for k, v in hp.items()}, **(extra_meta or {})}
    arrays['meta_json'] = np.array(json.dumps(meta))
    np.savez(path, **arrays)


def load_checkpoint(path, device='cpu'):
    """Rebuild the model from a .pt produced by export_checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    a = ckpt['architecture']
    model = MLP(a['input_dim'], a['hidden_layers'], a['num_classes'], a['dropout'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt

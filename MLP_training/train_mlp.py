#!/usr/bin/env python3
"""Train the final PyTorch MLP with the Optuna-selected hyperparameters and export a .pt.

Hyperparameters come from the `final_model:` section of config.yaml (written by
optuna_search.py); --params best_params.json overrides it.

Default: fit preprocessing + MLP on train.csv, report train and validation metrics,
and export MLP_training/out/final/mlp_final.pt (PyTorch) and mlp_final.npz (NumPy only,
see predict_numpy.py). The test split is NOT read unless
you pass --evaluate-test (do that once, after every choice is frozen).

The .pt contains weights, architecture and every fitted preprocessing number
(column order, imputation values, log1p, scaler mean/std, TF-IDF vocab + idf,
payment fill value), stored as plain types so it loads with weights_only=True.
preprocessor.joblib (sklearn) is also saved for convenient raw-CSV -> features.

Usage (from the repository root):
  python MLP_training/train_mlp.py                       # train on train, check on val
  python MLP_training/train_mlp.py --train-on train+val  # final refit (val no longer independent)
  python MLP_training/train_mlp.py --train-on train+val --evaluate-test   # once, at the end
  python MLP_training/train_mlp.py --params MLP_training/out/optuna/best_params.json
  python MLP_training/predict_numpy.py MLP_training/out/final/mlp_final.npz   # inspect the export
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import mlp_common as C
import predict_numpy

HP_KEYS = ['hidden_layers', 'dropout', 'learning_rate', 'batch_size',
           'optimizer', 'weight_decay', 'epochs']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    ap.add_argument('--params', default=None, help='best_params.json from optuna_search.py')
    ap.add_argument('--train-on', choices=['train', 'train+val'], default='train')
    ap.add_argument('--evaluate-test', action='store_true',
                    help='Also score test.csv. Use once, after all choices are frozen.')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--out', default=None, help='Output dir (default <output.dir>/final)')
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    dcfg = cfg['data']
    classes = dcfg['classes']
    seed = args.seed if args.seed is not None else cfg['training']['seed']
    device = C.get_device(cfg['training']['device'])
    out = Path(args.out) if args.out else C.ROOT / cfg['output']['dir'] / 'final'
    out.mkdir(parents=True, exist_ok=True)

    # Hyperparameters: config.yaml `final_model:` (written by optuna_search.py),
    # or a best_params.json passed with --params.
    if args.params:
        hp = json.loads(Path(args.params).read_text())['hyperparameters']
    else:
        hp = cfg['final_model']
        print('From config final_model:', hp.get('source', ''))
    hp = {k: hp[k] for k in HP_KEYS}
    epochs = int(hp['epochs'])
    print('Hyperparameters:', json.dumps(hp))

    # ---- data (preprocessing fitted on the training rows only)
    train = C.read_split(dcfg['train'])
    val = C.read_split(dcfg['val'])
    fit_df = pd.concat([train, val], ignore_index=True) if args.train_on == 'train+val' else train
    assert set(train[dcfg['id_column']]).isdisjoint(val[dcfg['id_column']])
    held = {'validation': val} if args.train_on == 'train' else {}
    if args.evaluate_test:
        test = C.read_split(dcfg['test'])
        assert set(fit_df[dcfg['id_column']]).isdisjoint(test[dcfg['id_column']])
        held['test'] = test

    pre, mats = C.fit_transform(fit_df, *held.values())
    X_fit, X_held = mats[0], dict(zip(held, mats[1:]))
    y_fit = C.encode_labels(fit_df[dcfg['label_column']], classes)
    y_held = {k: C.encode_labels(df[dcfg['label_column']], classes) for k, df in held.items()}
    print(f'Fit rows: {len(X_fit)} ({args.train_on}), features: {X_fit.shape[1]}')

    # ---- train for the CV-selected number of epochs (no early stopping on val)
    model, curves = C.train_model(X_fit, y_fit, hp, epochs, seed, device)
    criterion = nn.CrossEntropyLoss()

    results = {'trained_on': args.train_on, 'seed': seed, 'epochs': epochs,
               'hyperparameters': hp, 'final_train_loss': curves['train_loss'][-1]}
    m, _ = C.evaluate(model, torch.as_tensor(X_fit, device=device),
                      torch.as_tensor(y_fit, device=device), criterion)
    results['fit_metrics'] = m
    print(f'[{args.train_on}] acc={m["accuracy"]:.4f} macro-F1={m["macro_f1"]:.4f} loss={m["loss"]:.4f}')

    for name in held:
        Xt = torch.as_tensor(X_held[name], device=device)
        yt = torch.as_tensor(y_held[name], device=device)
        m, pred = C.evaluate(model, Xt, yt, criterion)
        results[f'{name}_metrics'] = m
        print(f'[{name}] acc={m["accuracy"]:.4f} macro-F1={m["macro_f1"]:.4f} loss={m["loss"]:.4f}')
        cm = C.confusion(y_held[name], pred, classes)
        print(cm)
        cm.to_csv(out / f'{name}_confusion.csv')
        proba = model.predict_proba(Xt).cpu().numpy()
        df = held[name]
        pd.DataFrame({'id': df[dcfg['id_column']], 'actual': df[dcfg['label_column']],
                      'predicted': [classes[i] for i in pred],
                      **{f'p_{c}': proba[:, i] for i, c in enumerate(classes)}}
                     ).to_csv(out / f'{name}_predictions.csv', index=False)

    # ---- export
    pt_path = out / 'mlp_final.pt'
    C.export_checkpoint(pt_path, model, hp, classes, pre,
                        extra={'metrics': {k: v for k, v in results.items() if k.endswith('metrics')},
                               'trained_on': args.train_on, 'seed': seed})
    joblib.dump(pre, out / 'preprocessor.joblib')
    (out / 'metrics.json').write_text(json.dumps(results, indent=2))

    npz_path = out / 'mlp_final.npz'
    C.export_numpy(npz_path, model, hp, classes, pre,
                   extra_meta={'trained_on': args.train_on, 'seed': seed})

    # ---- checks: reloaded .pt and NumPy-only inference must match the trained model
    model = model.cpu().eval()
    reloaded, _ = C.load_checkpoint(pt_path)
    params = predict_numpy.load(npz_path)
    check = {'fit': (fit_df, X_fit), **{k: (held[k], X_held[k]) for k in held}}
    for name, (df, X) in check.items():
        with torch.no_grad():
            p_torch = model.predict_proba(torch.as_tensor(X)).numpy()
            p_pt = reloaded.predict_proba(torch.as_tensor(X)).numpy()
        p_np = predict_numpy.predict_proba(params, X)                       # model-ready input
        p_raw = predict_numpy.predict_proba_from_raw(params, C.raw_features(pre, df))  # raw input
        assert np.array_equal(p_torch.argmax(1), p_pt.argmax(1)), '.pt reload disagrees'
        for label, p in [('numpy', p_np), ('numpy from raw', p_raw)]:
            diff = np.abs(p - p_torch).max()
            assert diff < 1e-5 and np.array_equal(p.argmax(1), p_torch.argmax(1)), f'{label} disagrees on {name}: {diff}'
        print(f'[check {name}] .pt reload OK | numpy max |dp| = {np.abs(p_np - p_torch).max():.1e} '
              f'| numpy-from-raw max |dp| = {np.abs(p_raw - p_torch).max():.1e}')
    print(f'Saved {pt_path}, {npz_path}, preprocessor.joblib, metrics.json')


if __name__ == '__main__':
    main()

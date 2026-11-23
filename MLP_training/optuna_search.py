#!/usr/bin/env python3
"""Optuna search for the PyTorch MLP using grouped 5-fold CV on the training split.

Protocol (matches the proposal):
  * Only data/splits/train.csv is used. val.csv and test.csv are never read here.
  * GroupKFold by respondent id (shuffled, seed 311): one person's 3 rows stay together.
  * Each fold refits the preprocessing (TF-IDF, imputation, scaler) on its own
    training respondents only; it is cached once per fold and reused by all trials.
  * Each trial trains every fold for `max_epochs` and records the held-out curve.
    The number of epochs is chosen like a hyperparameter: the epoch with the highest
    mean CV accuracy (ties -> lower mean CV loss). That mean accuracy is the objective.

Outputs (in --out, default MLP_training/out/optuna):
  optuna.db, trials.csv, best_params.json
  and it rewrites the `final_model:` section of config.yaml, which train_mlp.py reads
  (pass --no-update-config to leave config.yaml alone).

Usage (from the repository root):
  python MLP_training/optuna_search.py --config MLP_training/config.yaml
  python MLP_training/optuna_search.py --n-trials 10          # quick check
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
import torch
from sklearn.model_selection import GroupKFold

import mlp_common as C


def suggest(trial, space):
    """Turn the YAML search_space into trial.suggest_* calls; returns flat hp dict."""
    hp = {}
    for full_name, spec in space.items():
        key = full_name.split('.')[-1]
        if spec['type'] == 'categorical':
            hp[key] = trial.suggest_categorical(key, spec['choices'])
        elif spec['type'] == 'float':
            hp[key] = trial.suggest_float(key, spec['low'], spec['high'],
                                          log=spec.get('log', False), step=spec.get('step'))
        elif spec['type'] == 'int':
            hp[key] = trial.suggest_int(key, spec['low'], spec['high'],
                                        log=spec.get('log', False), step=spec.get('step', 1))
        else:
            raise ValueError(f'Unknown search type for {full_name}: {spec["type"]}')
    return hp


def build_folds(cfg):
    train = C.read_split(cfg['data']['train'])
    classes = cfg['data']['classes']
    y = C.encode_labels(train[cfg['data']['label_column']], classes)
    groups = train[cfg['data']['id_column']].to_numpy()
    gkf = GroupKFold(n_splits=cfg['cv']['n_splits'], shuffle=True,
                     random_state=cfg['cv']['seed'])
    folds = []
    for k, (tr, ho) in enumerate(gkf.split(train, y, groups)):
        assert set(groups[tr]).isdisjoint(groups[ho]), 'respondent leaked across folds'
        _, (X_tr, X_ho) = C.fit_transform(train.iloc[tr].reset_index(drop=True),
                                          train.iloc[ho].reset_index(drop=True))
        folds.append((X_tr, y[tr], X_ho, y[ho]))
        print(f'[fold {k}] train={len(tr)} heldout={len(ho)} features={X_tr.shape[1]}')
    return folds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    ap.add_argument('--n-trials', type=int, default=None, help='Override optuna.n_trials')
    ap.add_argument('--out', default=None, help='Output dir (default <output.dir>/optuna)')
    ap.add_argument('--no-update-config', action='store_true',
                    help='Do not write the best trial into the final_model section of --config')
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    ocfg = cfg['optuna']
    out = Path(args.out) if args.out else C.ROOT / cfg['output']['dir'] / 'optuna'
    out.mkdir(parents=True, exist_ok=True)
    device = C.get_device(cfg['training']['device'])
    torch.set_num_threads(max(1, torch.get_num_threads()))
    max_epochs = cfg['training']['max_epochs']
    seed = cfg['training']['seed']

    folds = build_folds(cfg)

    def objective(trial):
        hp = suggest(trial, ocfg['search_space'])
        accs, losses, f1s = [], [], []
        for k, (X_tr, y_tr, X_ho, y_ho) in enumerate(folds):
            _, curves = C.train_model(X_tr, y_tr, hp, max_epochs, seed, device, X_ho, y_ho)
            accs.append(curves['eval_accuracy'])
            losses.append(curves['eval_loss'])
            f1s.append(curves['eval_macro_f1'])
            # Pruning signal: running mean of each finished fold's best accuracy.
            trial.report(float(np.mean([max(a) for a in accs])), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()

        acc = np.mean(accs, axis=0)       # mean over folds, per epoch
        loss = np.mean(losses, axis=0)
        f1 = np.mean(f1s, axis=0)
        # best epoch: max mean accuracy, tie-break on lower mean loss
        best = int(np.lexsort((loss, -acc))[0])
        fold_acc_at_best = [a[best] for a in accs]
        trial.set_user_attr('best_epoch', best + 1)
        trial.set_user_attr('cv_accuracy_sd', float(np.std(fold_acc_at_best, ddof=1)))
        trial.set_user_attr('cv_macro_f1', float(f1[best]))
        trial.set_user_attr('cv_loss', float(loss[best]))
        trial.set_user_attr('min_cv_loss', float(loss.min()))
        trial.set_user_attr('min_cv_loss_epoch', int(loss.argmin()) + 1)
        return float(acc[best])

    study = optuna.create_study(
        direction='maximize',
        study_name=ocfg['study_name'],
        storage=f'sqlite:///{(out / "optuna.db").resolve()}',
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=ocfg['sampler_seed']),
        pruner=optuna.pruners.MedianPruner(**ocfg['pruner']),
    )
    start = time.time()
    study.optimize(objective, n_trials=args.n_trials or ocfg['n_trials'], n_jobs=1)
    print(f'Search took {time.time() - start:.0f}s')

    df = study.trials_dataframe()
    df.to_csv(out / 'trials.csv', index=False)

    # Select among completed trials: CV accuracy, then CV macro-F1, then lower CV loss.
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best = sorted(done, key=lambda t: (-t.value, -t.user_attrs['cv_macro_f1'],
                                       t.user_attrs['cv_loss']))[0]
    result = {
        'study_name': study.study_name,
        'trial_number': best.number,
        'cv_accuracy': best.value,
        **{k: best.user_attrs[k] for k in
           ['cv_accuracy_sd', 'cv_macro_f1', 'cv_loss', 'best_epoch']},
        'hyperparameters': {**best.params, 'epochs': best.user_attrs['best_epoch']},
        'n_trials_complete': len(done),
        'n_trials_pruned': sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        'protocol': 'GroupKFold(5, shuffle, seed 311) on train.csv only; fold-local preprocessing',
        'reference_sklearn_mlp_cv_accuracy': 0.9009756097560976,
    }
    (out / 'best_params.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not args.no_update_config:
        write_final_model(Path(args.config), result)
        print(f'Updated final_model in {args.config}')


def write_final_model(config_path: Path, result: dict):
    """Replace the trailing `final_model:` block of config.yaml, keeping all comments above it."""
    text = config_path.read_text()
    marker = '\nfinal_model:'
    head = text[:text.index(marker)] if marker in text else text.rstrip() + '\n'
    hp = result['hyperparameters']
    lines = ['', 'final_model:',
             f'  source: "optuna study {result["study_name"]}, trial {result["trial_number"]} '
             f'(CV acc {result["cv_accuracy"]:.4f} +/- {result["cv_accuracy_sd"]:.4f}, '
             f'macro-F1 {result["cv_macro_f1"]:.4f})"',
             f'  hidden_layers: "{hp["hidden_layers"]}"']
    lines += [f'  {k}: {hp[k]}' for k in
              ['dropout', 'learning_rate', 'batch_size', 'optimizer', 'weight_decay', 'epochs']]
    config_path.write_text(head + '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()

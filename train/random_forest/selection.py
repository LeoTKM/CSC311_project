"""Grouped repeated cross-validation, one-standard-error selection, bootstrap intervals."""
import json
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold

CLASSES = ['Soccer', 'Baseball', 'Basketball']
# Accuracy and macro-F1 are larger-is-better; log loss and error are smaller-is-better.
HIGHER_IS_BETTER = {'accuracy': True, 'macro_f1': True, 'log_loss': False}


def make_forest(params, seed, n_estimators=300):
    """Build the forest. The three searched keys are passed straight through."""
    return RandomForestClassifier(
        n_estimators=n_estimators, criterion='gini', bootstrap=True,
        class_weight=None, oob_score=False, n_jobs=1, random_state=seed, **params,
    )


def key_of(params):
    return json.dumps(params, sort_keys=True)


def folds_of(frame, fold_seed, n_splits):
    """Respondent-grouped folds; one respondent never straddles the split."""
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=fold_seed)
    return list(splitter.split(frame, frame['label'], frame['id']))


def prepare_folds(train, text_fields, keep_structured, fold_seed, n_splits):
    """Fit the preprocessor on each fold's fitting rows only, then transform."""
    from features import AblationPreprocessor

    prepared = []
    for fold, (fit_idx, held_idx) in enumerate(folds_of(train, fold_seed, n_splits)):
        fitting = train.iloc[fit_idx]
        held = train.iloc[held_idx]
        if set(fitting['id']) & set(held['id']):
            raise AssertionError(f'fold {fold}: a respondent appears on both sides')
        pre = AblationPreprocessor(text_fields, keep_structured).fit(fitting)
        prepared.append({
            'fold': fold,
            'fold_seed': fold_seed,
            'X_fit': pre.transform(fitting, scaled=False),
            'X_held': pre.transform(held, scaled=False),
            'y_fit': fitting['label'].to_numpy(),
            'y_held': held['label'].to_numpy(),
            'n_features': len(pre.columns_),
            # Feature names of this fold; the vocabulary differs per fold, so the
            # permutation importances must be labelled with the fold's own names.
            'columns': list(pre.columns_),
        })
    return prepared


def evaluate_config(prepared, params, model_seed=311, n_estimators=300):
    """One record per fold, so fold-to-fold spread stays visible."""
    records = []
    for data in prepared:
        model = make_forest(params, model_seed, n_estimators).fit(data['X_fit'], data['y_fit'])
        held_pred = model.predict(data['X_held'])
        fit_pred = model.predict(data['X_fit'])
        # predict_proba columns follow model.classes_ (alphabetical), not CLASSES.
        proba = model.predict_proba(data['X_held'])
        records.append({
            'fold': data['fold'],
            'fold_seed': data['fold_seed'],
            'params': key_of(params),
            'accuracy': float(accuracy_score(data['y_held'], held_pred)),
            'macro_f1': float(f1_score(data['y_held'], held_pred, labels=CLASSES,
                                       average='macro', zero_division=0)),
            'log_loss': float(log_loss(data['y_held'], proba, labels=list(model.classes_))),
            'train_accuracy': float(accuracy_score(data['y_fit'], fit_pred)),
            'n_features': int(data['n_features']),
        })
    return records


def summarize(records):
    """Collapse folds and fold seeds into one row per configuration."""
    frame = pd.DataFrame(records)
    grouped = frame.groupby('params', sort=False)
    summary = grouped.agg(
        cv_accuracy=('accuracy', 'mean'), cv_accuracy_sd=('accuracy', 'std'),
        cv_macro_f1=('macro_f1', 'mean'), cv_macro_f1_sd=('macro_f1', 'std'),
        cv_log_loss=('log_loss', 'mean'), cv_log_loss_sd=('log_loss', 'std'),
        train_accuracy=('train_accuracy', 'mean'),
        n_observations=('accuracy', 'size'), n_features=('n_features', 'max'),
    ).reset_index()
    summary['params_dict'] = summary['params'].map(json.loads)
    if 'fold_seed' in frame.columns:
        summary['fold_seeds'] = [
            len(sorted(set(frame.loc[frame.params == p, 'fold_seed'])))
            for p in summary['params']
        ]
    return summary


def simplicity(params):
    """Smaller is simpler: shallower depth, then larger leaves, then fewer features."""
    depth = params['max_depth']
    max_features = params['max_features']
    rank = {'sqrt': 0, 'log2': 1}.get(max_features, 2)
    numeric = max_features if isinstance(max_features, float) else 0.0
    return (100000 if depth is None else int(depth),
            -int(params['min_samples_leaf']), rank, numeric)


def one_se_select(summary, metric='accuracy', basis='se', complexity=simplicity):
    """Pick the simplest configuration inside one standard error of the best.

    basis='se' divides the spread by the square root of the number of fold-and-seed
    observations (the ESL definition); basis='sd' uses the raw spread, which makes
    the band wider.
    """
    if metric not in HIGHER_IS_BETTER:
        raise ValueError(f'unknown metric {metric!r}')
    higher = HIGHER_IS_BETTER[metric]
    column = f'cv_{metric}'
    values = summary[column].to_numpy(dtype=float)
    spread = summary[f'{column}_sd'].to_numpy(dtype=float)
    counts = summary['n_observations'].to_numpy(dtype=float)

    best_index = int(np.nanargmax(values) if higher else np.nanargmin(values))
    width = float(spread[best_index])
    if basis == 'se':
        width /= math.sqrt(max(counts[best_index], 1.0))
    elif basis != 'sd':
        raise ValueError(f'unknown basis {basis!r}')

    best = float(values[best_index])
    threshold = best - width if higher else best + width
    inside = values >= threshold if higher else values <= threshold
    band = summary.loc[inside].copy()
    band['simplicity'] = band['params_dict'].map(complexity)
    band = band.sort_values(['simplicity'], kind='stable')
    chosen = band.iloc[0]
    return {
        'metric': metric, 'basis': basis, 'higher_is_better': higher,
        'best_value': best, 'best_params': summary.loc[best_index, 'params_dict'],
        'spread': float(spread[best_index]),
        'band_width': width, 'threshold': threshold,
        'band_size': int(inside.sum()), 'n_configurations': int(len(summary)),
        'chosen_params': chosen['params_dict'],
        'chosen_value': float(chosen[column]),
        'chosen_n_features': int(chosen['n_features']),
    }


def grouped_bootstrap(y_true, y_pred, groups, n_boot=4000, seed=311):
    """Percentile intervals that resample respondents, not responses."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    ids = np.unique(groups)
    index_by_id = {name: np.flatnonzero(groups == name) for name in ids}
    accuracy, macro_f1 = [], []
    for _ in range(n_boot):
        drawn = rng.choice(ids, size=len(ids), replace=True)
        rows = np.concatenate([index_by_id[name] for name in drawn])
        accuracy.append(accuracy_score(y_true[rows], y_pred[rows]))
        macro_f1.append(f1_score(y_true[rows], y_pred[rows], labels=CLASSES,
                                 average='macro', zero_division=0))
    out = {}
    for name, values in (('accuracy', accuracy), ('macro_f1', macro_f1)):
        values = np.asarray(values)
        out[name] = {'mean': float(values.mean()),
                     'low': float(np.percentile(values, 2.5)),
                     'high': float(np.percentile(values, 97.5))}
    return out

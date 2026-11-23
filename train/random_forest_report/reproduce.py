"""Reproduce the report's random forest as a named version: `report`.

`analysis/report/proposal.tex` reports one forest row:

    Tree count 300, gini, bootstrap, seed 311, one CPU thread
    max_depth = None, max_features = 'sqrt', min_samples_leaf = 1
    208 predictors from ~train/random_forest/features.py's parent preprocessor
    CV   91.27 +- 2.14   over five grouped folds, fold seed 311
    val  93.18           123 of 132, macro F1 0.9317
    train 99.34          602 of 606 on the full-training refit

That row was produced by `analysis/model_experiments.py`, and this script re-runs exactly
that code rather than reimplementing it: it imports `SurveyPreprocessor`, `fit_model` and
`metrics` from that module and `GroupKFold(n_splits=5, shuffle=True, random_state=311)`
from scikit-learn, which is the fold construction written in its `run()`.

Three readings are reproduced and checked:

    cv5           the report's own protocol: 5 folds, fold seed 311
    cv15          the protocol v1, v2 and v3 use: 3 fold seeds x 5 splits
    fit606        one model on all 606 training rows, scored on those same rows
    validation    that model scored on the 132 validation rows

`cv15` exists because this version has to be comparable with v1, v2 and v3, which all
report fifteen folds. It is not part of the report; it is the same configuration read at a
finer resolution. The expected `cv15` number is the value v1's own search recorded for
`max_depth=None, max_features='sqrt', min_samples_leaf=1`, which is 0.9082.

Nothing here reads the test split. Validation is read, because that is what the report's
number is.

Run: python reproduce.py
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT / 'analysis', ROOT / 'data_preprocessing'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import model_experiments as experiments    # noqa: E402  the code that produced the report

TRAIN_CSV = ROOT / 'data' / 'splits' / 'train.csv'
VAL_CSV = ROOT / 'data' / 'splits' / 'val.csv'
RECORDED = ROOT / 'analysis' / 'out' / 'models'          # the report's own artifacts
V1_SEARCH = ROOT / 'random_forest_deliverables' / 'forest_search_v1'
DEFAULT_OUT = ROOT / 'random_forest_deliverables' / 'report_forest'

VERSION = 'report'
PARAMS = {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1}
MODEL_SEED = 311
REPORT_FOLD_SEED = 311
REPORT_SPLITS = 5
REPORT_FOLDS = 1                 # the report used a single fold seed
COMPARISON_SEEDS = (311, 312, 313)
N_ESTIMATORS = 300

CLASSES = list(experiments.CLASSES)

# Hashes the report recorded for the two source files it ran.
REPORT_SOURCES = {
    'analysis/model_experiments.py':
        '4a7e8f2633c3d0c44b1c0d0be288461c55b9674435f0388002c0e4c11a2df4f8',
    'data_preprocessing/encode_all.py':
        'c374e9502ec92808eeede80c8f17f6ad36024a4acd046da34eac0304c291c601',
}
TOLERANCE = 1e-9


def load_split(path):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in ('id', 'label'):
        if column not in frame.columns:
            raise ValueError(f'{path} is missing the {column!r} column')
    return frame


def recorded_numbers():
    """What the report says, read from its artifacts instead of its prose."""
    row = pd.read_csv(RECORDED / 'comparison.csv').query('family == "forest"').iloc[0]
    predictions = pd.read_csv(RECORDED / 'validation_predictions.csv',
                              dtype=str, keep_default_na=False)
    predictions = predictions[predictions['family'] == 'forest'].copy()
    v1_summary = pd.read_csv(V1_SEARCH / 'cv_summary.csv')
    v1_row = v1_summary[[json.loads(text) == PARAMS for text in v1_summary['params']]]
    return {
        'cv_accuracy': float(row.cv_accuracy),
        'cv_accuracy_sd': float(row.cv_accuracy_sd),
        'cv_macro_f1': float(row.cv_macro_f1),
        'train_accuracy': float(row.train_accuracy),
        'validation_accuracy': float(row.validation_accuracy),
        'validation_macro_f1': float(row.validation_macro_f1),
        'predictions': predictions,
        'v1_cv15_accuracy': float(v1_row.cv_accuracy.iloc[0]) if len(v1_row) else None,
        'v1_cv15_sd': float(v1_row.cv_accuracy_sd.iloc[0]) if len(v1_row) else None,
    }


def source_check():
    """Do the two files the report ran still hash to what its manifest recorded?"""
    rows = []
    for relative, expected in REPORT_SOURCES.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        rows.append({'file': relative, 'report_sha256': expected,
                     'current_sha256': actual, 'unchanged': actual == expected})
    return rows


def provenance(splits):
    names = ['analysis/model_experiments.py', 'data_preprocessing/encode_all.py',
             'train/random_forest_report/reproduce.py']
    hashed = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in names if (ROOT / name).exists()}
    return {
        'version': VERSION,
        'test_accessed': False,
        'splits_read': list(splits),
        'params': PARAMS, 'model_seed': MODEL_SEED, 'n_estimators': N_ESTIMATORS,
        'sources': hashed,
        'source_check': source_check(),
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name)
                     for name in ('pandas', 'numpy', 'scikit-learn', 'joblib')},
    }


def fit_one(fitting, held, seed=MODEL_SEED):
    """Fold-local preprocessing, then the report's forest. Returns (fit, held) arrays."""
    preprocessor = experiments.SurveyPreprocessor().fit(fitting)
    X_fit = preprocessor.transform(fitting, False)
    X_held = preprocessor.transform(held, False)
    model, warnings, retry, seconds = experiments.fit_model(
        'forest', PARAMS, seed, X_fit, fitting['label'].to_numpy())
    return preprocessor, model, X_fit, X_held, warnings, retry, seconds


def stage_cv(train, seeds, label):
    """One record per fold, using the report's own fold constructor."""
    records, started = [], pd.Timestamp.now()
    for fold_seed in seeds:
        folds = GroupKFold(n_splits=REPORT_SPLITS, shuffle=True,
                           random_state=fold_seed).split(train, train['label'], train['id'])
        for fold, (fit_index, held_index) in enumerate(folds):
            fitting, held = train.iloc[fit_index], train.iloc[held_index]
            if set(fitting['id']) & set(held['id']):
                raise AssertionError(f'fold {fold}: a respondent appears twice')
            preprocessor, model, X_fit, X_held, warnings, retry, seconds = \
                fit_one(fitting, held)
            held_metrics = experiments.metrics(held['label'], model.predict(X_held))
            fit_metrics = experiments.metrics(fitting['label'], model.predict(X_fit))
            records.append({
                'fold_seed': fold_seed, 'fold': fold,
                'fit_rows': len(fitting), 'held_rows': len(held),
                'n_features': len(preprocessor.columns_),
                'accuracy': float(held_metrics['accuracy']),
                'macro_f1': float(held_metrics['macro_f1']),
                'train_accuracy': float(fit_metrics['accuracy']),
                'retry': int(retry),
            })
    frame = pd.DataFrame(records)
    summary = {
        'label': label, 'fold_seeds': list(seeds), 'n_splits': REPORT_SPLITS,
        'folds': int(len(frame)),
        'cv_accuracy': float(frame.accuracy.mean()),
        'cv_accuracy_sd': float(frame.accuracy.std(ddof=1)) if len(frame) > 1 else 0.0,
        'cv_macro_f1': float(frame.macro_f1.mean()),
        'cv_macro_f1_sd': float(frame.macro_f1.std(ddof=1)) if len(frame) > 1 else 0.0,
        'train_accuracy': float(frame.train_accuracy.mean()),
        'n_features': int(frame.n_features.max()),
        'seconds': float((pd.Timestamp.now() - started).total_seconds()),
    }
    return frame, summary


def same(actual, expected):
    if expected is None:
        return None
    return bool(abs(float(actual) - float(expected)) < TOLERANCE)


def main():
    parser = argparse.ArgumentParser(description='reproduce the report forest as a version')
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    args = parser.parse_args()

    train = load_split(TRAIN_CSV)
    val = load_split(VAL_CSV)
    if set(train['id']) & set(val['id']):
        raise ValueError('a respondent appears in both splits')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recorded = recorded_numbers()

    print(f'version {VERSION}: 300 trees, gini, bootstrap, seed {MODEL_SEED}')
    print(f'  parameters {json.dumps(PARAMS, sort_keys=True)}')
    print(f'  train {len(train)} rows / {train.id.nunique()} respondents, '
          f'validation {len(val)} rows / {val.id.nunique()} respondents')
    print('\n  sources the report ran')
    for row in source_check():
        print(f'    {row["file"]:42s} '
              f'{"unchanged" if row["unchanged"] else "CHANGED since the report run"}')
    print()

    # 1. the report's own protocol: five folds, fold seed 311
    folds5, cv5 = stage_cv(train, (REPORT_FOLD_SEED,), 'report protocol')
    folds5.to_csv(out / 'cv5_records.csv', index=False)
    print(f'1. cv5  (5 folds, fold seed {REPORT_FOLD_SEED})')
    print(f'     accuracy {cv5["cv_accuracy"]:.4f} +- {cv5["cv_accuracy_sd"]:.4f}   '
          f'report {recorded["cv_accuracy"]:.4f} +- {recorded["cv_accuracy_sd"]:.4f}')
    print(f'     macro F1 {cv5["cv_macro_f1"]:.4f}                 '
          f'report {recorded["cv_macro_f1"]:.4f}')
    print(f'     in-fold train {cv5["train_accuracy"]:.4f} (mean over folds)')

    # 2. the protocol v1, v2 and v3 use, so this version is comparable with them
    folds15, cv15 = stage_cv(train, COMPARISON_SEEDS, 'fifteen folds')
    folds15.to_csv(out / 'cv15_records.csv', index=False)
    print(f'\n2. cv15 (3 fold seeds x 5 splits, the v1/v2/v3 protocol)')
    print(f'     accuracy {cv15["cv_accuracy"]:.4f} +- {cv15["cv_accuracy_sd"]:.4f}   '
          f'v1 search recorded {recorded["v1_cv15_accuracy"]:.4f} '
          f'+- {recorded["v1_cv15_sd"]:.4f}')
    print(f'     macro F1 {cv15["cv_macro_f1"]:.4f}   '
          f'in-fold train {cv15["train_accuracy"]:.4f}   {cv15["n_features"]} features')

    # 3. one model on all 606 rows, scored on those same rows
    preprocessor = experiments.SurveyPreprocessor().fit(train)
    X_fit = preprocessor.transform(train, False)
    X_val = preprocessor.transform(val, False)
    model, warnings, retry, seconds = experiments.fit_model(
        'forest', PARAMS, MODEL_SEED, X_fit, train['label'].to_numpy())
    fit_predictions = model.predict(X_fit)
    val_predictions = model.predict(X_val)
    train_metrics = experiments.metrics(train['label'], fit_predictions)
    val_marks = experiments.metrics(val['label'], val_predictions)
    print(f'\n3. fit606: {len(preprocessor.columns_)} predictors, train accuracy '
          f'{train_metrics["accuracy"]:.4f} ({int((fit_predictions == train["label"]).sum())}'
          f'/{len(train)})   report {recorded["train_accuracy"]:.4f}')

    # 4. the validation reading, and a row-by-row check against the report's predictions
    print(f'\n4. validation: accuracy {val_marks["accuracy"]:.4f}   '
          f'report {recorded["validation_accuracy"]:.4f}')
    print(f'   macro F1   {val_marks["macro_f1"]:.4f}   '
          f'report {recorded["validation_macro_f1"]:.4f}')
    mine = pd.DataFrame({'id': val['id'].to_numpy(), 'actual': val['label'].to_numpy(),
                         'predicted': val_predictions})
    theirs = recorded['predictions']
    aligned = (len(mine) == len(theirs)
               and (mine['id'].to_numpy() == theirs['id'].to_numpy()).all())
    if aligned:
        differing = int((mine['predicted'].to_numpy()
                         != theirs['predicted'].to_numpy()).sum())
    else:
        differing = None
    print(f'   predictions differing from the report\'s saved file: '
          f'{"cannot compare" if differing is None else f"{differing} of {len(mine)}"}')

    confusion = pd.DataFrame(
        confusion_matrix(val['label'], val_predictions, labels=CLASSES),
        index=CLASSES, columns=CLASSES)

    checks = {
        'cv5_accuracy': same(cv5['cv_accuracy'], recorded['cv_accuracy']),
        'cv5_accuracy_sd': same(cv5['cv_accuracy_sd'], recorded['cv_accuracy_sd']),
        'cv5_macro_f1': same(cv5['cv_macro_f1'], recorded['cv_macro_f1']),
        'cv15_accuracy': same(cv15['cv_accuracy'], recorded['v1_cv15_accuracy']),
        'train_accuracy': same(train_metrics['accuracy'], recorded['train_accuracy']),
        'validation_accuracy': same(val_marks['accuracy'],
                                    recorded['validation_accuracy']),
        'validation_macro_f1': same(val_marks['macro_f1'],
                                    recorded['validation_macro_f1']),
        'validation_predictions_identical': differing == 0 if differing is not None else None,
    }
    reproduced = all(value for value in checks.values() if value is not None)

    print('\n--- reproduction check ---')
    for name, value in checks.items():
        mark = 'match' if value else ('cannot compare' if value is None else 'DIFFERS')
        print(f'  {name:34s} {mark}')
    print(f'  {"all recorded numbers":34s} '
          f'{"reproduced" if reproduced else "NOT reproduced"}')

    mine.to_csv(out / 'validation_predictions.csv', index=False)
    confusion.to_csv(out / 'validation_confusion.csv')
    (out / 'reproduction.json').write_text(json.dumps({
        'version': VERSION,
        'reproduced': reproduced,
        'checks': checks,
        'reproduced_numbers': {'cv5': cv5, 'cv15': cv15,
                               'fit606_accuracy': float(train_metrics['accuracy']),
                               'fit606_macro_f1': float(train_metrics['macro_f1']),
                               'validation_accuracy': float(val_marks['accuracy']),
                               'validation_macro_f1': float(val_marks['macro_f1']),
                               'validation_errors': int(len(val) - (val_predictions
                                                                   == val['label']).sum()),
                               'n_features': int(len(preprocessor.columns_)),
                               'retry': int(retry)},
        'recorded_numbers': {name: value for name, value in recorded.items()
                             if name != 'predictions'},
        'predictions_differing': differing,
        **provenance(['data/splits/train.csv', 'data/splits/val.csv']),
    }, indent=2), encoding='utf-8')
    (out / 'run_manifest.json').write_text(json.dumps({
        'version': VERSION, 'stage': 'reproduce', 'reproduced': reproduced,
        **provenance(['data/splits/train.csv', 'data/splits/val.csv']),
    }, indent=2), encoding='utf-8')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()

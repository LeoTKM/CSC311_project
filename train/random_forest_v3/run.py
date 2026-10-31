"""v3 training pipeline. Five stages.

    cv           one fixed setting, 15 grouped folds, training split only
    ablation     which short-answer field earns its columns, training split only
    search       150-configuration grid, 15 grouped folds, training split only
    diagnostics  tree count, learning curve, permutation importance, training split only
    validation   fit the 606 training rows, score the 132 validation rows, once

The first four stages are what `--stage all` runs, and none of them touches the
validation split. `validation` is a separate stage that has to be asked for by name,
because every reading it takes is one that stops being available for later design
decisions.

`validation` is not a train-plus-validation refit. The preprocessor, the four term
vocabularies and the trees are fitted on the 606 training rows only; the 132 validation
rows are transformed and scored, never fitted. It exists so that v3 has a number standing
next to the ones the two earlier models reported on the same 132 rows and 44 respondents.

No stage reads the test split, and no stage refits on train plus validation.

Run:

    python run.py --stage all
    python run.py --stage cv --dry-run
    python run.py --stage diagnostics --source random_forest_deliverables/v3_forest
    python run.py --stage validation
"""
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (HERE,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import features    # noqa: E402
import selection   # noqa: E402

TRAIN_CSV = ROOT / 'data' / 'splits' / 'train.csv'
VAL_CSV = ROOT / 'data' / 'splits' / 'val.csv'
DEFAULT_OUT = ROOT / 'random_forest_deliverables' / 'v3_forest'

TRAIN_RELATIVE = 'data/splits/train.csv'
VAL_RELATIVE = 'data/splits/val.csv'

# The default for `cv` before the search has decided anything. It is v2's chosen setting
# on purpose: the two are then read at the same tree shape, and the search replaces it
# with whatever v3's own grid prefers.
STARTING_POINT = {'max_depth': 8, 'min_samples_leaf': 2, 'max_features': 'log2'}

TEXT_VARIANTS = {
    'all': features.TEXT_FIELDS,
    'no_food': ('colour', 'song', 'feeling'),
    'no_colour': ('food', 'song', 'feeling'),
    'no_song': ('food', 'colour', 'feeling'),
    'no_feeling': ('food', 'colour', 'song'),
    'no_song_no_feeling': ('food', 'colour'),
    'colour_only': ('colour',),
    'none': (),
}
ABLATION_GRID = {'max_depth': [4, None], 'min_samples_leaf': [1, 5],
                 'max_features': ['sqrt']}

TREE_COUNTS = (100, 200, 300, 500, 800)
LEARNING_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
PERMUTATION_REPEATS = 5
FLAG_NAMES = {name for name, _ in features.FLAG_QUESTIONS}


def load_split(path):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in ('id', 'label'):
        if column not in frame.columns:
            raise ValueError(f'{path} is missing the {column!r} column')
    return frame


def provenance(*splits):
    """Which files produced an artifact, and which splits it was allowed to read."""
    names = ['text.py', 'encode.py', 'features.py', 'selection.py', 'run.py']
    files = list(splits) or [TRAIN_RELATIVE]
    hashes = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
              for path in files}
    record = {
        'test_accessed': False,
        'validation_accessed': VAL_RELATIVE in files,
        'splits_read': files,
        'sources': {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                    for name in names},
        'train_sha256': hashes[TRAIN_RELATIVE],
    }
    if VAL_RELATIVE in hashes:
        record['val_sha256'] = hashes[VAL_RELATIVE]
    record.update({
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name)
                     for name in ('pandas', 'numpy', 'scikit-learn', 'joblib')},
    })
    return record


def grid_of(dry_run):
    if not dry_run:
        return [dict(zip(selection.GRID, values))
                for values in itertools.product(*selection.GRID.values())]
    return [dict(zip(selection.GRID, values))
            for values in itertools.product([4, None], [1, 5], ['sqrt'])]


def params_from_args(args):
    return {'max_depth': args.max_depth, 'min_samples_leaf': args.min_samples_leaf,
            'max_features': args.max_features}


def params_from_selection(source):
    """The parameters chosen by the search stage, if that stage has been run."""
    path = Path(source) / 'selection.json'
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))['chosen_params']


def resolve_params(args):
    return params_from_selection(args.source or args.out) or params_from_args(args)


def report_frame(prepared, params):
    frame = prepared[0]
    print(f'  features per row : {frame["n_features"]}')
    print(f'  rows per fold    : fit {len(frame["y_fit"])}, held {len(frame["y_held"])}')
    print(f'  parameters       : {json.dumps(params, sort_keys=True)}')


def block_of(column):
    for field in features.TEXT_FIELDS:
        if column.startswith(f'{field}_'):
            return field
    if column in FLAG_NAMES:
        return 'missing flags'
    return 'structured'


# --------------------------------------------------------------------------- cv
def stage_cv(args):
    train = load_split(TRAIN_CSV)
    seeds = selection.FOLD_SEEDS[:1] if args.dry_run else selection.FOLD_SEEDS
    params = params_from_args(args)
    print(f'train rows {len(train)}, respondents {train.id.nunique()}, '
          f'fold seeds {seeds}')
    records, started = [], time.perf_counter()
    for seed in seeds:
        prepared = selection.prepare_folds(train, seed)
        seeded = selection.evaluate_config(prepared, params,
                                           n_estimators=args.n_estimators)
        records.extend(seeded)
        print(f'  seed {seed}: accuracy {np.mean([r["accuracy"] for r in seeded]):.4f}'
              f'   macro F1 {np.mean([r["macro_f1"] for r in seeded]):.4f}'
              f'   log loss {np.mean([r["log_loss"] for r in seeded]):.4f}')
        report_frame(prepared, params)
    frame = pd.DataFrame(records)
    print()
    print(f'grouped repeated CV over {len(records)} folds in '
          f'{time.perf_counter() - started:.0f}s')
    print(f'  cv accuracy    {frame.accuracy.mean():.4f} +- {frame.accuracy.std():.4f}')
    print(f'  cv macro F1    {frame.macro_f1.mean():.4f} +- {frame.macro_f1.std():.4f}')
    print(f'  cv log loss    {frame.log_loss.mean():.4f}')
    print(f'  train accuracy {frame.train_accuracy.mean():.4f}')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / 'cv_records.csv', index=False)
    summary = {
        'stage': 'cv', 'rows': int(len(train)), 'fold_seeds': list(seeds),
        'params': params, 'n_estimators': args.n_estimators,
        'cv_accuracy': float(frame.accuracy.mean()),
        'cv_accuracy_sd': float(frame.accuracy.std()),
        'cv_macro_f1': float(frame.macro_f1.mean()),
        'cv_log_loss': float(frame.log_loss.mean()),
        'train_accuracy': float(frame.train_accuracy.mean()),
        'n_features': int(frame.n_features.max()),
        **provenance(),
    }
    (out / 'cv_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\nwrote {out / "cv_records.csv"} and {out / "cv_summary.json"}')
    return summary


# --------------------------------------------------------------------- ablation
def stage_ablation(args):
    train = load_split(TRAIN_CSV)
    seed = selection.FOLD_SEEDS[0]
    grid = [dict(zip(selection.GRID, values))
            for values in itertools.product(*ABLATION_GRID.values())]
    print(f'train rows {len(train)}, fold seed {seed}, '
          f'{len(grid)} configurations per field set, {len(TEXT_VARIANTS)} field sets')
    records, rows = [], []
    for name, fields in TEXT_VARIANTS.items():
        prepared = selection.prepare_folds(train, seed, text_fields=fields)
        block = []
        for params in grid:
            block.extend(selection.evaluate_config(prepared, params,
                                                   n_estimators=args.n_estimators))
        frame = pd.DataFrame(block).assign(variant=name,
                                           n_features=prepared[0]['n_features'])
        records.extend(frame.to_dict('records'))
        best = frame.loc[frame.accuracy.idxmax()]
        rows.append({'variant': name, 'text_fields': ','.join(fields) or '-',
                     'n_features': prepared[0]['n_features'],
                     'best_accuracy': float(best.accuracy),
                     'best_macro_f1': float(best.macro_f1),
                     'best_params': best.params,
                     'mean_accuracy': float(frame.accuracy.mean())})
    table = pd.DataFrame(rows)
    print()
    print(f'{"variant":20s} {"fields":26s} {"cols":>5s} {"best acc":>9s} '
          f'{"mean acc":>9s}  best configuration')
    for row in table.itertuples():
        print(f'{row.variant:20s} {row.text_fields:26s} {row.n_features:5d} '
              f'{row.best_accuracy:9.4f} {row.mean_accuracy:9.4f}  '
              f'{json.dumps(json.loads(row.best_params), sort_keys=True)}')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out / 'ablation_folds.csv', index=False)
    table.to_csv(out / 'ablation_summary.csv', index=False)
    (out / 'ablation_summary.json').write_text(
        json.dumps({'stage': 'ablation', 'fold_seed': seed,
                    'grid': ABLATION_GRID, 'variants': rows, **provenance()},
                   indent=2), encoding='utf-8')
    print(f'\nwrote {out / "ablation_folds.csv"}, {out / "ablation_summary.csv"} '
          f'and {out / "ablation_summary.json"}')
    return table


# ------------------------------------------------------------------------- search
def stage_search(args):
    train = load_split(TRAIN_CSV)
    seeds = selection.FOLD_SEEDS[:1] if args.dry_run else selection.FOLD_SEEDS
    grid = grid_of(args.dry_run)
    print(f'train rows {len(train)}, fold seeds {seeds}, {len(grid)} configurations')
    started = time.perf_counter()
    prepared_by_seed = {seed: selection.prepare_folds(train, seed) for seed in seeds}
    print(f'  preprocessed {len(seeds)} fold sets in '
          f'{time.perf_counter() - started:.0f}s')
    records = []
    for index, params in enumerate(grid, start=1):
        for seed in seeds:
            records.extend(selection.evaluate_config(
                prepared_by_seed[seed], params, n_estimators=args.n_estimators))
        if index % 25 == 0 or index == len(grid):
            print(f'  {index}/{len(grid)} configurations, '
                  f'{time.perf_counter() - started:.0f}s', flush=True)
    summary = selection.summarize(records)
    chosen = selection.one_se_select(summary, metric='accuracy')
    print()
    print(f'best cv accuracy {chosen["best_value"]:.4f} at '
          f'{json.dumps(chosen["best_params"], sort_keys=True)}')
    print(f'one-SE band width {chosen["band_width"]:.4f}, '
          f'{chosen["band_size"]} of {chosen["n_configurations"]} configurations inside')
    print(f'chosen (simplest inside the band): '
          f'{json.dumps(chosen["chosen_params"], sort_keys=True)} '
          f'at {chosen["chosen_value"]:.4f}')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out / 'search_records.csv', index=False)
    summary.drop(columns=['params_dict']).to_csv(out / 'search_summary.csv', index=False)
    (out / 'selection.json').write_text(
        json.dumps({**chosen, **provenance()}, indent=2, default=str), encoding='utf-8')
    print(f'\nwrote {out / "search_records.csv"}, {out / "search_summary.csv"} '
          f'and {out / "selection.json"}')
    return chosen


# -------------------------------------------------------------------- diagnostics
def stage_diagnostics(args):
    train = load_split(TRAIN_CSV)
    params = resolve_params(args)
    seeds = selection.FOLD_SEEDS[:1] if args.dry_run else selection.FOLD_SEEDS
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f'train rows {len(train)}, fold seeds {seeds}')
    print(f'parameters {json.dumps(params, sort_keys=True)}')
    prepared_by_seed = {seed: selection.prepare_folds(train, seed) for seed in seeds}
    print(f'  {prepared_by_seed[seeds[0]][0]["n_features"]} features per row')

    # 1. tree count
    print('\n--- tree count')
    print(f'{"trees":>6s} {"accuracy":>10s} {"macro F1":>10s} {"log loss":>10s}')
    tree_rows = []
    for count in TREE_COUNTS:
        records = []
        for seed in seeds:
            records.extend(selection.evaluate_config(prepared_by_seed[seed], params,
                                                     n_estimators=count))
        frame = pd.DataFrame(records)
        tree_rows.append({'n_estimators': count,
                          'accuracy': float(frame.accuracy.mean()),
                          'macro_f1': float(frame.macro_f1.mean()),
                          'log_loss': float(frame.log_loss.mean())})
        print(f'{count:6d} {frame.accuracy.mean():10.4f} {frame.macro_f1.mean():10.4f} '
              f'{frame.log_loss.mean():10.4f}')

    # 2. learning curve
    print('\n--- learning curve')
    print(f'{"fraction":>9s} {"rows":>6s} {"respondents":>12s} {"accuracy":>10s} '
          f'{"macro F1":>10s}')
    curve_rows = []
    for fraction in LEARNING_FRACTIONS:
        records = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            for fit_index, held_index in selection.folds_of(train, seed):
                fitting, held = train.iloc[fit_index], train.iloc[held_index]
                ids = np.unique(fitting['id'])
                keep = ids if fraction >= 1 else rng.choice(
                    ids, size=max(2, int(round(fraction * len(ids)))), replace=False)
                subset = fitting[fitting['id'].isin(keep)]
                pre = features.TextPreprocessor().fit(subset)
                model = selection.make_forest(params, seed, args.n_estimators)
                model.fit(pre.transform(subset), subset['label'].to_numpy())
                predicted = model.predict(pre.transform(held))
                records.append((accuracy_score(held['label'], predicted),
                                f1_score(held['label'], predicted, labels=selection.CLASSES,
                                         average='macro', zero_division=0),
                                len(subset), subset['id'].nunique()))
        frame = pd.DataFrame(records, columns=['accuracy', 'macro_f1', 'rows', 'people'])
        curve_rows.append({'fraction': fraction, 'rows': int(frame.rows.mean()),
                           'respondents': int(frame.people.mean()),
                           'accuracy': float(frame.accuracy.mean()),
                           'macro_f1': float(frame.macro_f1.mean())})
        print(f'{fraction:9.2f} {frame.rows.mean():6.0f} {frame.people.mean():12.0f} '
              f'{frame.accuracy.mean():10.4f} {frame.macro_f1.mean():10.4f}')

    # 3. permutation importance by block
    #
    # The term vocabularies are fitted per fold, so the blocks are summed inside each fold
    # using that fold's own columns and then averaged across folds. Summing over the union
    # of column names instead would count a column once per fold it appeared in and then
    # divide by the number of folds, which understates every rare column.
    print('\n--- permutation importance by block')
    block_totals, column_values, folds_seen = {}, {}, 0
    for seed in seeds:
        for fold, data in enumerate(prepared_by_seed[seed]):
            model = selection.make_forest(params, seed, args.n_estimators)
            model.fit(data['X_fit'], data['y_fit'])
            result = permutation_importance(
                model, data['X_held'], data['y_held'], n_repeats=PERMUTATION_REPEATS,
                random_state=seed + fold, n_jobs=1, scoring='accuracy')
            series = pd.Series(result.importances_mean, index=data['columns'])
            for block in {block_of(column) for column in series.index}:
                part = float(series[[column for column in series.index
                                     if block_of(column) == block]].sum())
                block_totals.setdefault(block, []).append(part)
            for column, value in series.items():
                column_values.setdefault(column, []).append(value)
            folds_seen += 1
    print(f'{folds_seen} folds')
    print(f'{"block":16s} {"cols/fold":>10s} {"sum":>8s} {"share":>7s}')
    shared = sum(np.mean(values) for values in block_totals.values())
    block_rows = []
    sizes = {}
    for data in prepared_by_seed[seeds[0]]:
        for block in {block_of(column) for column in data['columns']}:
            sizes[block] = sizes.get(block, 0) + sum(
                1 for column in data['columns'] if block_of(column) == block)
    for block, values in sorted(block_totals.items(), key=lambda kv: -np.mean(kv[1])):
        part = float(np.mean(values))
        block_rows.append({'block': block,
                           'columns_per_fold': sizes.get(block, 0) / len(
                               prepared_by_seed[seeds[0]]),
                           'importance': part,
                           'share_of_positive': part / shared if shared else float('nan')})
        print(f'{block:16s} {block_rows[-1]["columns_per_fold"]:10.1f} {part:8.4f} '
              f'{block_rows[-1]["share_of_positive"]:7.1%}')
    print('\ntop 15 columns, importance averaged over the folds each appeared in')
    column_mean = {column: float(np.mean(values)) for column, values in column_values.items()}
    for column, value in sorted(column_mean.items(), key=lambda kv: -kv[1])[:15]:
        appeared = len(column_values[column])
        print(f'  {column:44s} {value:+.5f}   in {appeared:2d} folds')

    pd.DataFrame(tree_rows).to_csv(out / 'trees_curve.csv', index=False)
    pd.DataFrame(curve_rows).to_csv(out / 'learning_curve.csv', index=False)
    pd.DataFrame(block_rows).to_csv(out / 'permutation_importance_by_block.csv', index=False)
    pd.Series(column_mean, name='importance').sort_values(ascending=False).to_csv(
        out / 'permutation_importance_by_column.csv')
    (out / 'diagnostics.json').write_text(json.dumps({
        'stage': 'diagnostics', 'params': params, 'fold_seeds': list(seeds),
        'n_estimators_curve': tree_rows, 'learning_curve': curve_rows,
        'importance_by_block': block_rows, **provenance()}, indent=2), encoding='utf-8')
    print(f'\nwrote four tables and diagnostics.json to {out}')
    return block_rows


# --------------------------------------------------------------------- validation
def stage_validation(args):
    """Fit the 606 training rows, then score the 132 validation rows once.

    Nothing from the validation split is fitted: the four vocabularies, the structured
    imputers, the payment fill value and every tree come from the training rows alone.
    """
    train, val = load_split(TRAIN_CSV), load_split(VAL_CSV)
    overlap = set(train['id']) & set(val['id'])
    if overlap:
        raise ValueError(f'{len(overlap)} respondents appear in both splits')
    if not val.groupby('id').size().eq(3).all():
        raise ValueError('a validation respondent does not have exactly three responses')
    params = resolve_params(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f'fitting train:     {len(train)} rows, {train.id.nunique()} respondents')
    print(f'scoring validation: {len(val)} rows, {val.id.nunique()} respondents')
    print(f'parameters {json.dumps(params, sort_keys=True)}')

    preprocessor = features.TextPreprocessor().fit(train)
    X_fit, X_held = preprocessor.transform(train), preprocessor.transform(val)
    model = selection.make_forest(params, 311, args.n_estimators)
    model.fit(X_fit, train['label'].to_numpy())
    predicted = model.predict(X_held)
    proba = model.predict_proba(X_held)

    metrics = {
        'stage': 'validation',
        'rows': int(len(val)), 'respondents': int(val['id'].nunique()),
        'params': params, 'n_estimators': args.n_estimators,
        'n_features': len(preprocessor.columns_),
        'accuracy': float(accuracy_score(val['label'], predicted)),
        'macro_f1': float(f1_score(val['label'], predicted, labels=selection.CLASSES,
                                   average='macro', zero_division=0)),
        'log_loss': float(log_loss(val['label'], proba, labels=list(model.classes_))),
        'bootstrap': selection.grouped_bootstrap(val['label'], predicted, val['id']),
        'per_class': classification_report(val['label'], predicted,
                                           labels=selection.CLASSES,
                                           output_dict=True, zero_division=0),
        'confusion': confusion_matrix(val['label'], predicted,
                                      labels=selection.CLASSES).tolist(),
        'classes': list(model.classes_),
        'train_accuracy': float(accuracy_score(train['label'], model.predict(X_fit))),
        'refit_on': 'train split only',
        **provenance(TRAIN_RELATIVE, VAL_RELATIVE),
    }
    pd.DataFrame({'id': val['id'], 'actual': val['label'], 'predicted': predicted}) \
        .to_csv(out / 'validation_predictions.csv', index=False)
    pd.DataFrame(metrics['confusion'], index=selection.CLASSES, columns=selection.CLASSES) \
        .to_csv(out / 'validation_confusion.csv')
    (out / 'validation_metrics.json').write_text(json.dumps(metrics, indent=2),
                                                 encoding='utf-8')

    print()
    print(f'  accuracy   {metrics["accuracy"]:.4f}')
    print(f'  macro F1   {metrics["macro_f1"]:.4f}')
    print(f'  log loss   {metrics["log_loss"]:.4f}')
    print(f'  train      {metrics["train_accuracy"]:.4f}   (gap '
          f'{metrics["train_accuracy"] - metrics["accuracy"]:+.4f})')
    print(f'  bootstrap  accuracy [{metrics["bootstrap"]["accuracy"]["low"]:.4f}, '
          f'{metrics["bootstrap"]["accuracy"]["high"]:.4f}]  '
          f'macro F1 [{metrics["bootstrap"]["macro_f1"]["low"]:.4f}, '
          f'{metrics["bootstrap"]["macro_f1"]["high"]:.4f}]')
    for name in selection.CLASSES:
        row = metrics['per_class'][name]
        print(f'  {name:11s} precision {row["precision"]:.4f}  recall {row["recall"]:.4f}  '
              f'F1 {row["f1-score"]:.4f}')
    print('\n  confusion, rows are actual')
    print(f'  {"":12s}' + ''.join(f'{name:>12s}' for name in selection.CLASSES))
    for name, line in zip(selection.CLASSES, metrics['confusion']):
        print(f'  {name:12s}' + ''.join(f'{count:12d}' for count in line))
    print(f'\nwrote validation_predictions.csv, validation_confusion.csv and '
          f'validation_metrics.json to {out}')
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='v3 training pipeline (validation is opt-in by stage name)')
    parser.add_argument('--stage', default='cv',
                        choices=['cv', 'ablation', 'search', 'diagnostics',
                                 'validation', 'all'])
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    parser.add_argument('--source', default=None,
                        help='directory holding selection.json, defaults to --out')
    parser.add_argument('--n-estimators', type=int, default=selection.N_ESTIMATORS)
    parser.add_argument('--max-depth', type=int, default=STARTING_POINT['max_depth'])
    parser.add_argument('--min-samples-leaf', type=int,
                        default=STARTING_POINT['min_samples_leaf'])
    parser.add_argument('--max-features', default=STARTING_POINT['max_features'])
    parser.add_argument('--dry-run', action='store_true',
                        help='one fold seed; a four-configuration search grid')
    args = parser.parse_args()

    print('# ' + '=' * 76)
    print(f'# stage {args.stage}')
    print('# ' + '=' * 76)
    if args.stage in ('cv', 'all'):
        stage_cv(args)
    if args.stage in ('ablation', 'all'):
        stage_ablation(args)
    if args.stage in ('search', 'all'):
        stage_search(args)
    if args.stage in ('diagnostics', 'all'):
        stage_diagnostics(args)
    # Deliberately not part of `all`: scoring the validation split is a one-way read.
    if args.stage == 'validation':
        stage_validation(args)


if __name__ == '__main__':
    main()

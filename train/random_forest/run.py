#!/usr/bin/env python3
"""Random-forest training pipeline for the CSC311 survey project.

Training is allowed to use scikit-learn; only the eventual inference script must be
dependency-free, so nothing here is reused at prediction time.

Stages (all write into one output directory that must not already exist):

    ablation     compare text-field sets (with/without song and feeling) on a
                 reduced grid and one fold seed
    search       full grid on grouped repeated cross-validation, then a
                 one-standard-error selection
    diagnostics  tree-count convergence, learning curve, permutation importance
    fit          refit on the canonical training split and score the validation set
    final        refit preprocessing and model on training + validation
    test         one-shot evaluation of the final model on the test split
                 (requires an explicit acknowledgement flag)

Typical order:

    python train/random_forest/run.py --dry-run --stage search
    python train/random_forest/run.py --stage ablation --output random_forest_deliverables/forest_abl
    python train/random_forest/run.py --stage search --text-fields no_song_no_feeling \
        --output random_forest_deliverables/forest_search
    python train/random_forest/run.py --stage diagnostics --source random_forest_deliverables/forest_search
    python train/random_forest/run.py --stage fit --source random_forest_deliverables/forest_search
    python train/random_forest/run.py --stage final --source random_forest_deliverables/forest_search
    python train/random_forest/run.py --stage test --source random_forest_deliverables/forest_search \
        --i-accept-using-the-test-set
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config       # noqa: E402
import features     # noqa: E402
import selection    # noqa: E402

ROOT = config.ROOT
STAGES = ('ablation', 'search', 'diagnostics', 'fit', 'final', 'test')


# --------------------------------------------------------------------- helpers
def load_split(path):
    """Read a raw split exactly as text so answers such as "None" survive."""
    if path is None or not Path(path).exists():
        raise SystemExit(f'missing split file: {path}')
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in ('id', 'label'):
        if column not in frame.columns:
            raise SystemExit(f'{path}: missing {column!r} column')
    return frame


def grid_of(grid):
    """Expand a dict of lists into a list of parameter dicts."""
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[key] for key in keys))]


def ensure_output(path, stage):
    path = Path(path)
    if path.exists():
        raise SystemExit(
            f'{path} already exists. Choose a fresh --output directory; results are '
            'never overwritten.')
    path.mkdir(parents=True)
    (path / 'stage.txt').write_text(f'{stage}\n')
    return path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(train_path, val_path, test_path=None):
    import importlib.metadata
    files = [train_path, val_path,
             ROOT / 'data_preprocessing' / 'encode_all.py',
             ROOT / 'analysis' / 'model_experiments.py',
             HERE / 'config.py', HERE / 'features.py',
             HERE / 'selection.py', HERE / 'run.py']
    if test_path is not None:
        files.append(test_path)
    packages = {}
    for name in ('numpy', 'pandas', 'scikit-learn', 'scipy'):
        try:
            packages[name] = importlib.metadata.version(name)
        except Exception:                                    # pragma: no cover
            packages[name] = 'unknown'
    return {
        'sha256': {str(Path(p).relative_to(ROOT)): sha256(p) for p in files},
        'packages': packages,
        'python': platform.python_version(),
        'fold_seeds': list(config.FOLD_SEEDS),
        'ablation_fold_seeds': list(config.ABLATION_FOLD_SEEDS),
        'n_splits': config.N_SPLITS,
        'grid': config.GRID,
        'ablation_grid': config.ABLATION_GRID,
        'fixed': config.FIXED,
        'selection_rule': config.SELECTION_NOTE,
    }


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, default=str))


def show(frame, columns=None, n=None):
    view = frame if columns is None else frame[columns]
    if n is not None:
        view = view.head(n)
    print(view.to_string(index=False), flush=True)


def load_selection(source):
    """Read the chosen configuration either from a search dir or from a JSON path."""
    source = Path(source)
    if source.is_dir():
        source = source / 'selection_rationale.json'
    if not source.exists():
        raise SystemExit(f'no selection file at {source}; run the search stage first')
    payload = json.loads(source.read_text())
    text_fields = tuple(payload['text_fields'])
    keep_structured = bool(payload.get('keep_structured', True))
    return payload, text_fields, keep_structured


# ---------------------------------------------------------------------- stages
def stage_ablation(args):
    out = ensure_output(args.output, 'ablation')
    train = load_split(config.TRAIN_CSV)
    records = []
    for name, text_fields in config.TEXT_VARIANTS.items():
        keep_structured = args.structured
        for fold_seed in config.ABLATION_FOLD_SEEDS:
            prepared = selection.prepare_folds(
                train, text_fields, keep_structured, fold_seed, config.N_SPLITS)
            for params in grid_of(config.ABLATION_GRID):
                for record in selection.evaluate_config(prepared, params):
                    record.update(variant=name, text_fields='|'.join(text_fields),
                                  keep_structured=keep_structured, fold_seed=fold_seed)
                    records.append(record)
        print(f'ablation: {name} done', flush=True)

    frame = pd.DataFrame(records)
    frame.to_csv(out / 'ablation_folds.csv', index=False)
    summary = (frame.groupby(['variant', 'text_fields', 'keep_structured'], sort=False)
               .agg(cv_accuracy=('accuracy', 'mean'), cv_accuracy_sd=('accuracy', 'std'),
                    cv_macro_f1=('macro_f1', 'mean'), cv_log_loss=('log_loss', 'mean'),
                    train_accuracy=('train_accuracy', 'mean'),
                    n_features=('n_features', 'max'), n_fits=('accuracy', 'size'))
               .reset_index())
    # Best configuration per field set, so the comparison is not diluted by the grid.
    best = (frame.sort_values(['accuracy'], ascending=False)
            .groupby('variant', sort=False).head(1)
            [['variant', 'params', 'accuracy', 'macro_f1', 'log_loss']]
            .rename(columns={'accuracy': 'best_accuracy', 'macro_f1': 'best_macro_f1',
                             'log_loss': 'best_log_loss', 'params': 'best_params'}))
    summary = summary.merge(best, on='variant', how='left')
    summary = summary.sort_values('best_accuracy', ascending=False)
    summary.to_csv(out / 'ablation_summary.csv', index=False)
    print('\nText-field ablation (higher accuracy is better; n_features from the fitted fold):')
    show(summary, ['variant', 'text_fields', 'n_features', 'cv_accuracy', 'cv_accuracy_sd',
                   'best_accuracy', 'best_macro_f1', 'best_log_loss', 'best_params'])
    write_json(out / 'run_manifest.json',
               {'stage': 'ablation', 'provenance': provenance(config.TRAIN_CSV, config.VAL_CSV)})
    print(f'saved {out}')


def stage_search(args):
    out = ensure_output(args.output, 'search')
    train = load_split(config.TRAIN_CSV)
    text_fields = config.TEXT_VARIANTS[args.text_fields]
    keep_structured = args.structured
    params_list = grid_of(config.GRID)
    n_folds = config.N_SPLITS * len(config.FOLD_SEEDS)
    print(f'search: {len(params_list)} configurations x {n_folds} folds = '
          f'{len(params_list) * n_folds} fits', flush=True)

    started = time.perf_counter()
    records = []
    for fold_seed in config.FOLD_SEEDS:
        prepared = selection.prepare_folds(
            train, text_fields, keep_structured, fold_seed, config.N_SPLITS)
        for index, params in enumerate(params_list, 1):
            records.extend(selection.evaluate_config(prepared, params))
        print(f'  fold seed {fold_seed} done '
              f'({time.perf_counter() - started:.0f}s)', flush=True)

    frame = pd.DataFrame(records)
    frame.to_csv(out / 'cv_folds.csv', index=False)
    summary = selection.summarize(records)
    summary.drop(columns=['params_dict']).sort_values('cv_accuracy', ascending=False) \
           .to_csv(out / 'cv_summary.csv', index=False)

    decision = selection.one_se_select(summary, metric=args.select_metric,
                                       basis=args.se_basis)
    payload = {
        'text_fields': list(text_fields), 'keep_structured': keep_structured,
        'fold_seeds': list(config.FOLD_SEEDS), 'n_splits': config.N_SPLITS,
        'grid': config.GRID, 'fixed': config.FIXED,
        'selection_rule': config.SELECTION_NOTE,
        'passed_frontier_value': decision['best_value'],
        'passed_frontier_params': decision['best_params'],
        'chosen_params': decision['chosen_params'],
        'chosen_value': decision['chosen_value'],
        'band': {k: decision[k] for k in
                 ('metric', 'basis', 'spread', 'band_width', 'threshold',
                  'band_size', 'n_configurations', 'higher_is_better')},
        'elapsed_seconds': round(time.perf_counter() - started, 1),
        'provenance': provenance(config.TRAIN_CSV, config.VAL_CSV),
    }
    write_json(out / 'selection_rationale.json', payload)

    print('\nTop configurations by mean grouped repeated CV:')
    show(summary.sort_values('cv_accuracy', ascending=False),
         ['params', 'cv_accuracy', 'cv_accuracy_sd', 'cv_macro_f1', 'cv_log_loss',
          'train_accuracy', 'n_features'], n=10)
    print(f"\nmetric={decision['metric']} basis={decision['basis']} "
          f"best={decision['best_value']:.4f} band_width={decision['band_width']:.4f} "
          f"threshold={decision['threshold']:.4f} "
          f"({decision['band_size']}/{decision['n_configurations']} inside)")
    print(f"chosen (simplest inside the band): {decision['chosen_params']} "
          f"@ {decision['chosen_value']:.4f}")
    print(f'saved {out}')


def block_of(column):
    """Map a feature name to the question block it came from."""
    if column == features.EASE_COLUMN:
        return 'Q1 ease'
    if column in features.AGREEMENT_COLUMNS:
        return f'Q{features.AGREEMENT_COLUMNS.index(column) + 2} Likert'
    for number, (question, choices) in zip((5, 6, 7), features.SCHEMA.items()):
        if column in [name for _, name in choices]:
            return f'Q{number} multi-select'
    if column.startswith('payment'):
        return 'Q12 payment'
    if column.startswith('q') and column.endswith('_missing'):
        return 'missing flags'
    for field in features.TEXT:
        if column.startswith(f'{field}_tfidf_') or column == f'{field}_text_missing':
            return f'text: {field}'
    return 'other'


def stage_diagnostics(args):
    out = ensure_output(args.output, 'diagnostics')
    payload, text_fields, keep_structured = load_selection(args.source)
    params = payload['chosen_params']
    train = load_split(config.TRAIN_CSV)
    fold_seed = config.FOLD_SEEDS[0]
    prepared = selection.prepare_folds(train, text_fields, keep_structured,
                                       fold_seed, config.N_SPLITS)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.inspection import permutation_importance

    # 1) tree-count convergence: does 300 trees already plateaus?
    tree_rows = []
    for trees in args.trees:
        records = selection.evaluate_config(prepared, params, n_estimators=trees)
        frame = pd.DataFrame(records)
        tree_rows.append({'n_estimators': trees,
                          'cv_accuracy': frame.accuracy.mean(),
                          'cv_accuracy_sd': frame.accuracy.std(),
                          'cv_log_loss': frame.log_loss.mean()})
    trees_frame = pd.DataFrame(tree_rows)
    trees_frame.to_csv(out / 'trees_curve.csv', index=False)

    # 2) learning curve: subsample respondents inside each fitting fold
    fraction_rows = []
    rng = np.random.default_rng(311)
    for fraction in args.fractions:
        records = [_learning_point(train, text_fields, keep_structured, params,
                                   fold_seed, data['fold'], fraction, rng)
                   for data in prepared]
        frame = pd.DataFrame(records)
        fraction_rows.append({'fraction': fraction,
                              'cv_accuracy': frame.accuracy.mean(),
                              'cv_accuracy_sd': frame.accuracy.std()})
    learning = pd.DataFrame(fraction_rows)
    learning.to_csv(out / 'learning_curve.csv', index=False)

    # 3) permutation importance on held-out folds, aggregated by question block
    importances = {}
    for data in prepared:
        model = selection.make_forest(params, 311).fit(data['X_fit'], data['y_fit'])
        result = permutation_importance(model, data['X_held'], data['y_held'],
                                        n_repeats=5, random_state=311, n_jobs=1)
        for name, value in zip(data['columns'], result.importances_mean):
            importances.setdefault(name, []).append(float(value))
    importance = pd.DataFrame({'feature': list(importances),
                               'importance': [float(np.mean(v)) for v in importances.values()]})
    importance['block'] = importance['feature'].map(block_of)
    by_block = (importance.groupby('block').importance.sum().sort_values(ascending=False)
                .rename('permutation_importance').reset_index())
    importance.sort_values('importance', ascending=False).to_csv(
        out / 'permutation_importance.csv', index=False)
    by_block.to_csv(out / 'permutation_importance_by_block.csv', index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(trees_frame.n_estimators, trees_frame.cv_accuracy, 'o-')
    axes[0].set(xlabel='Number of trees', ylabel='Mean CV accuracy',
                title='Tree-count convergence')
    axes[1].plot(learning.fraction * 100, learning.cv_accuracy, 'o-')
    axes[1].set(xlabel='Share of fitting respondents used (%)', ylabel='Mean CV accuracy',
                title='Learning curve')
    axes[2].barh(by_block.block[::-1], by_block.permutation_importance[::-1])
    axes[2].set(xlabel='Mean permutation importance', title='Importance by question block')
    plt.tight_layout()
    plt.savefig(out / 'diagnostics.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print('\nTree-count convergence:')
    show(trees_frame.round(4))
    print('\nLearning curve:')
    show(learning.round(4))
    print('\nPermutation importance by question block:')
    show(by_block.round(4))
    write_json(out / 'run_manifest.json', {'stage': 'diagnostics', 'params': params,
                                           'text_fields': list(text_fields)})
    print(f'saved {out}')


def _fold_rows(train, fold_seed, fold):
    fit_idx, held_idx = selection.folds_of(train, fold_seed, config.N_SPLITS)[fold]
    return fit_idx, held_idx


def _learning_point(train, text_fields, keep_structured, params, fold_seed, fold,
                    fraction, rng):
    fit_idx, held_idx = _fold_rows(train, fold_seed, fold)
    fitting = train.iloc[fit_idx]
    held = train.iloc[held_idx]
    ids = np.unique(fitting['id'])
    keep_n = max(1, int(round(fraction * len(ids))))
    if keep_n < len(ids):
        chosen = set(rng.choice(ids, size=keep_n, replace=False))
        fitting = fitting[fitting['id'].isin(chosen)]
    pre = features.AblationPreprocessor(text_fields, keep_structured).fit(fitting)
    model = selection.make_forest(params, 311).fit(
        pre.transform(fitting, scaled=False), fitting['label'].to_numpy())
    pred = model.predict(pre.transform(held, scaled=False))
    return {'fold': fold,
            'accuracy': float((pred == held['label'].to_numpy()).mean())}


def stage_fit(args):
    out = ensure_output(args.output, 'fit')
    payload, text_fields, keep_structured = load_selection(args.source)
    params = payload['chosen_params']
    train = load_split(config.TRAIN_CSV)
    val = load_split(config.VAL_CSV)
    if set(train['id']) & set(val['id']):
        raise SystemExit('train and validation share respondent ids')

    pre = features.AblationPreprocessor(text_fields, keep_structured).fit(train)
    model = selection.make_forest(params, 311).fit(
        pre.transform(train, scaled=False), train['label'].to_numpy())
    predicted = model.predict(pre.transform(val, scaled=False))

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    metrics = {
        'accuracy': float(accuracy_score(val['label'], predicted)),
        'macro_f1': float(f1_score(val['label'], predicted, labels=config.CLASSES,
                                   average='macro', zero_division=0)),
        'bootstrap': selection.grouped_bootstrap(val['label'], predicted, val['id']),
        'per_class': classification_report(val['label'], predicted, labels=config.CLASSES,
                                           output_dict=True, zero_division=0),
    }
    predictions = pd.DataFrame({'id': val['id'], 'actual': val['label'],
                                'predicted': predicted})
    predictions.to_csv(out / 'validation_predictions.csv', index=False)
    pd.DataFrame(confusion_matrix(val['label'], predicted, labels=config.CLASSES),
                 index=config.CLASSES, columns=config.CLASSES) \
        .to_csv(out / 'validation_confusion.csv')
    write_json(out / 'validation_metrics.json', metrics)
    write_json(out / 'run_manifest.json',
               {'stage': 'fit', 'params': params, 'text_fields': list(text_fields),
                'train_rows': len(train), 'validation_rows': len(val),
                'test_accessed': False,
                'provenance': provenance(config.TRAIN_CSV, config.VAL_CSV)})

    print(f"validation accuracy {metrics['accuracy']:.4f} "
          f"({int(round(metrics['accuracy'] * len(val)))}/{len(val)})  "
          f"macro-F1 {metrics['macro_f1']:.4f}")
    for name in ('accuracy', 'macro_f1'):
        band = metrics['bootstrap'][name]
        print(f"  {name}: 95% respondent-bootstrap CI "
              f"[{band['low']:.4f}, {band['high']:.4f}]")
    print('per class:', {c: round(metrics['per_class'][c]['recall'], 4)
                         for c in config.CLASSES}, '(recall)')
    print(f'saved {out}')


def stage_final(args):
    out = ensure_output(args.output, 'final')
    payload, text_fields, keep_structured = load_selection(args.source)
    params = payload['chosen_params']
    train = load_split(config.TRAIN_CSV)
    val = load_split(config.VAL_CSV)
    if set(train['id']) & set(val['id']):
        raise SystemExit('training and validation share respondent ids')
    combined = pd.concat([train, val], ignore_index=True)
    # Every respondent contributes exactly three responses, so a repeated id is
    # expected; the split-overlap check above is what actually guards the merge.
    if not combined.groupby('id').size().eq(3).all():
        raise SystemExit('every respondent must contribute exactly three responses')

    pre = features.AblationPreprocessor(text_fields, keep_structured).fit(combined)
    model = selection.make_forest(params, 311).fit(
        pre.transform(combined, scaled=False), combined['label'].to_numpy())

    import joblib
    joblib.dump({'preprocessor': pre, 'model': model, 'scaled': False,
                 'trained_on': 'train + validation', 'text_fields': list(text_fields),
                 'keep_structured': keep_structured, 'params': params},
                out / 'final_model.joblib')
    write_json(out / 'final_manifest.json', {
        'stage': 'final', 'params': params, 'text_fields': list(text_fields),
        'keep_structured': keep_structured,
        'rows': len(combined), 'respondents': int(combined['id'].nunique()),
        'n_features': len(pre.columns_), 'classes': list(model.classes_),
        'validation_now_inside_training': True,
        'test_accessed': False,
        'provenance': provenance(config.TRAIN_CSV, config.VAL_CSV),
    })
    print(f'final model trained on {len(combined)} rows '
          f'({combined["id"].nunique()} respondents), {len(pre.columns_)} features')
    print(f"classes order inside the model: {list(model.classes_)} "
          '(alphabetical - map by name, never by position)')
    print(f'saved {out}')


def stage_test(args):
    if not args.i_accept_using_the_test_set:
        raise SystemExit('the test split may be opened once. Re-run with '
                         '--i-accept-using-the-test-set after the model is frozen.')
    out = Path(args.output)
    if not (out / 'final_model.joblib').exists():
        raise SystemExit(f'{out} has no final_model.joblib; run the final stage first')
    lock = out / 'test_accessed.json'
    if lock.exists():
        raise SystemExit(f'{lock} exists: the test split was already evaluated for this model')
    payload = json.loads((out / 'final_manifest.json').read_text())
    test = load_split(config.TEST_CSV)

    import joblib
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    artifact = joblib.load(out / 'final_model.joblib')
    pre, model = artifact['preprocessor'], artifact['model']
    predicted = model.predict(pre.transform(test, scaled=False))

    metrics = {
        'accuracy': float(accuracy_score(test['label'], predicted)),
        'macro_f1': float(f1_score(test['label'], predicted, labels=config.CLASSES,
                                   average='macro', zero_division=0)),
        'bootstrap': selection.grouped_bootstrap(test['label'], predicted, test['id']),
        'per_class': classification_report(test['label'], predicted, labels=config.CLASSES,
                                           output_dict=True, zero_division=0),
    }
    pd.DataFrame({'id': test['id'], 'actual': test['label'], 'predicted': predicted}) \
        .to_csv(out / 'test_predictions.csv', index=False)
    pd.DataFrame(confusion_matrix(test['label'], predicted, labels=config.CLASSES),
                 index=config.CLASSES, columns=config.CLASSES) \
        .to_csv(out / 'test_confusion.csv')
    write_json(out / 'test_metrics.json', metrics)
    write_json(lock, {'evaluated_utc': pd.Timestamp.now(tz='UTC').isoformat(),
                      'rows': len(test), 'params': payload['params'],
                      'test_accessed': True})
    print(f"test accuracy {metrics['accuracy']:.4f}  macro-F1 {metrics['macro_f1']:.4f}")
    for name in ('accuracy', 'macro_f1'):
        band = metrics['bootstrap'][name]
        print(f"  {name}: 95% respondent-bootstrap CI "
              f"[{band['low']:.4f}, {band['high']:.4f}]")
    print(f'saved {out}  (test lock written: this result is final)')


# ------------------------------------------------------------------- dry run
def count_rows(path):
    """Count CSV records, not physical lines: answers contain embedded newlines."""
    import csv
    with open(path, newline='', encoding='utf-8-sig') as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def describe(args):
    train_n = count_rows(config.TRAIN_CSV)
    val_n = count_rows(config.VAL_CSV)
    lines = [f'stage            : {args.stage}', f'output           : {args.output}',
             f'train / val rows : {train_n} / {val_n}',
             f'selection metric : {args.select_metric} (basis {args.se_basis})']
    if args.stage in ('ablation', 'search'):
        lines += [f'text fields      : {args.text_fields}'
                  f' -> {config.TEXT_VARIANTS[args.text_fields]}',
                  f'keep structured  : {args.structured}']
    if args.stage == 'search':
        grid = grid_of(config.GRID)
        folds = config.N_SPLITS * len(config.FOLD_SEEDS)
        lines += [f'grid             : {len(grid)} configurations',
                  f'fold seeds       : {list(config.FOLD_SEEDS)} x {config.N_SPLITS} folds',
                  f'total fits       : {len(grid) * folds}',
                  'estimated time   : '
                  f'~{len(grid) * folds * args.seconds_per_fit / 60:.0f} min '
                  f'at {args.seconds_per_fit}s/fit']
    elif args.stage == 'ablation':
        grid = grid_of(config.ABLATION_GRID)
        folds = config.N_SPLITS * len(config.ABLATION_FOLD_SEEDS)
        variants = len(config.TEXT_VARIANTS)
        lines += [f'variants         : {list(config.TEXT_VARIANTS)}',
                  f'grid             : {len(grid)} configurations per variant',
                  f'total fits       : {len(grid) * folds * variants}',
                  'estimated time   : '
                  f'~{len(grid) * folds * variants * args.seconds_per_fit / 60:.0f} min']
    elif args.stage == 'diagnostics':
        lines += [f'source           : {args.source}',
                  f'tree counts      : {args.trees}',
                  f'learning fractions: {args.fractions}']
    else:
        lines += [f'source           : {args.source}']
    print('\n'.join(lines))
    print('dry run: nothing was written and no model was fitted.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=STAGES, default='search')
    parser.add_argument('--output', type=Path, default=config.DEFAULT_OUTPUT,
                        help='new directory for this stage; must not exist')
    parser.add_argument('--text-fields', choices=list(config.TEXT_VARIANTS),
                        default='all', help='which text blocks to keep')
    parser.add_argument('--no-structured', dest='structured', action='store_false',
                        help='drop Q1-Q7 and payment (text-only features)')
    parser.add_argument('--select-metric', default='accuracy',
                        choices=['accuracy', 'macro_f1', 'log_loss'])
    parser.add_argument('--se-basis', default='se', choices=['se', 'sd'],
                        help="'se' = spread/sqrt(n) (ESL); 'sd' = raw spread")
    parser.add_argument('--source', default=None,
                        help='the search stage output directory (or its '
                             'selection_rationale.json) holding the chosen configuration')
    parser.add_argument('--trees', type=int, nargs='+', default=[100, 200, 300, 500, 800])
    parser.add_argument('--fractions', type=float, nargs='+',
                        default=[0.25, 0.4, 0.55, 0.7, 0.85, 1.0])
    parser.add_argument('--seconds-per-fit', type=float, default=0.6,
                        help='only used to estimate the dry-run runtime')
    parser.add_argument('--dry-run', action='store_true',
                        help='print the plan and exit without touching anything')
    parser.add_argument('--i-accept-using-the-test-set', action='store_true')
    args = parser.parse_args(argv)

    if args.stage in ('diagnostics', 'fit', 'final', 'test') and args.source is None:
        parser.error(f'--stage {args.stage} needs --source <search output directory>')
    if args.dry_run:
        describe(args)
        return 0
    if not config.TRAIN_CSV.exists():
        raise SystemExit(f'missing {config.TRAIN_CSV}')

    handlers = {'ablation': stage_ablation, 'search': stage_search,
                'diagnostics': stage_diagnostics, 'fit': stage_fit,
                'final': stage_final, 'test': stage_test}
    handlers[args.stage](args)
    return 0


if __name__ == '__main__':
    sys.exit(main())

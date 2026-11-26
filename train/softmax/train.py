"""Train and verify a softmax candidate using canonical grouped splits."""
import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass, replace
from functools import partial
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import pred

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'analysis'), str(ROOT / 'data_preprocessing')]
from model_experiments import fit_model
from encode_all import AGREEMENT_MAP, FULL_COLUMNS, SCHEMA, SHORT_COLUMNS

C_VALUES = (.03, .1, .3, 1., 3., 10.)
FOLD_SEEDS = (311, 312, 313)
CLASSES = ['Soccer', 'Baseball', 'Basketball']
FIELDS = ('food', 'colour', 'song', 'feeling')
SCHEMA_PAYLOAD = {
    'headers': dict(zip(SHORT_COLUMNS, FULL_COLUMNS)),
    'agreement': {key: float(value) for key, value in AGREEMENT_MAP.items()},
    'multi': {key: [option.lower() for option, _ in options]
              for key, options in zip(('location', 'season', 'music_type'), SCHEMA.values())},
}
# Fixed aliases from train/random_forest_v3/encode.py.
COLOUR_LEXICON = {
    'blue': ['blue', 'blues'], 'red': ['red'], 'orange': ['orange'], 'green': ['green'],
    'white': ['white'], 'black': ['black', 'balck'], 'brown': ['brown'],
    'yellow': ['yellow', 'yellowish'], 'gold': ['gold', 'golden'], 'purple': ['purple'],
    'navy': ['navy'], 'gray': ['gray', 'grey'], 'beige': ['beige'], 'maroon': ['maroon'],
    'burgundy': ['burgundy'], 'cream': ['cream'], 'tan': ['tan'],
}


@dataclass(frozen=True)
class Representation:
    name: str
    word_caps: tuple = (50, 20, 50, 50)
    word_analyzer: str = 'legacy'
    char_caps: tuple = (0, 0, 0, 0)
    text_weight: float = 1.
    fields: tuple = FIELDS
    colour_lexicon: bool = False
    coverage_flags: bool = False


BASELINE = Representation('baseline')
WIDE = Representation('wide_words', (125, 45, 125, 100))
NORMALIZED = replace(WIDE, name='normalized_words', word_analyzer='normalized')
CHAR = replace(NORMALIZED, name='word_plus_char', char_caps=(100, 60, 100, 80))
TEXT_CHOICES = (WIDE, NORMALIZED, CHAR, replace(BASELINE, name='weighted_baseline', text_weight=2.),
                replace(CHAR, name='weighted_word_plus_char', text_weight=2.))


def read(path):
    rows = pred.read_rows(path, SCHEMA_PAYLOAD)
    meta = pd.read_csv(path, dtype=str, usecols=['id', 'label'], keep_default_na=False)
    assert len(rows) == len(meta) and set(meta.label) == set(CLASSES)
    assert meta.groupby('id').label.nunique().eq(3).all() and meta.groupby('id').size().eq(3).all()
    return rows, meta.label.to_numpy(), meta.id.to_numpy()


def fit_parameters(rows, representation):
    matrix = pred.structure(rows, SCHEMA_PAYLOAD)
    fill = np.zeros(matrix.shape[1])
    for col in range(26):
        values = matrix[:, col]
        observed = values[~np.isnan(values)]
        if observed.size:
            if col < 4:
                fill[col] = np.median(observed)
            elif col == 25:
                fill[col] = observed.mean()
            else:
                values, counts = np.unique(observed, return_counts=True)
                fill[col] = values[counts.argmax()]
    identity = {'fill': fill, 'mean': np.zeros(len(fill)), 'scale': np.ones(len(fill))}
    fitted = pred.prepared_structure(matrix, identity)
    scale_indices = [0, 1, 2, 3, 25]
    scaler = StandardScaler().fit(fitted[:, scale_indices])
    identity['mean'][scale_indices] = scaler.mean_
    identity['scale'][scale_indices] = scaler.scale_
    blocks = []
    for i, field in enumerate(FIELDS):
        if field not in representation.fields:
            continue
        text = [pred.clean(row[field]) for row in rows]
        for analyzer, cap in ((representation.word_analyzer, representation.word_caps[i]),
                              ('char', representation.char_caps[i])):
            if cap == 0:
                continue
            vectorizer = TfidfVectorizer(analyzer=partial(pred.terms, analyzer=analyzer),
                                        max_features=cap, min_df=2, sublinear_tf=True)
            try:
                vectorizer.fit(text)
            except ValueError as error:
                if not any(reason in str(error).lower() for reason in
                           ('empty vocabulary', 'no terms remain', 'max_df corresponds')):
                    raise
                continue
            blocks.append({'field': field, 'analyzer': analyzer, 'weight': representation.text_weight,
                           'vocabulary': vectorizer.get_feature_names_out().tolist(),
                           'idf': vectorizer.idf_.tolist()})
    return {'schema': SCHEMA_PAYLOAD, 'structured': {key: value.tolist() for key, value in identity.items()},
            'text_blocks': blocks, 'fields': representation.fields,
            'colour_lexicon': COLOUR_LEXICON if representation.colour_lexicon else {},
            'coverage_flags': representation.coverage_flags}


def score(model, matrix, labels):
    predictions = model.predict(matrix)
    return {'accuracy': float(accuracy_score(labels, predictions)),
            'macro_f1': float(f1_score(labels, predictions, labels=CLASSES, average='macro')),
            'log_loss': float(log_loss(labels, model.predict_proba(matrix), labels=list(model.classes_)))}


def partition(labels, groups, seeds, n_splits):
    for seed in seeds:
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (fitting, held) in enumerate(splitter.split(labels, labels, groups)):
            assert set(groups[fitting]).isdisjoint(groups[held])
            yield seed, fold, fitting, held


def take(rows, indices):
    return [rows[i] for i in indices]


def fit(matrix, labels, strength):
    return fit_model('softmax', {'C': strength}, 311, matrix, labels)


def rank(records):
    return sorted(records, key=lambda row: (-row['cv_accuracy'], -row['cv_macro_f1'],
                                            row['cv_log_loss'], row['mean_features'], row['C']))


def search(rows, labels, groups, seeds, n_splits, verbose=False):
    folds = [(seed, fold, take(rows, tr), take(rows, ho), labels[tr], labels[ho])
             for seed, fold, tr, ho in partition(labels, groups, seeds, n_splits)]
    all_fits, summary, representations = [], [], {}

    def evaluate(representation, stage):
        representations[representation.name] = representation
        results = {C: [] for C in C_VALUES}
        for seed, fold, fitting, held, y_fit, y_held in folds:
            params = fit_parameters(fitting, representation)
            X, V = pred.features(fitting, params), pred.features(held, params)
            for C in C_VALUES:
                model, _, retries, _ = fit(X, y_fit, C)
                metrics = score(model, V, y_held)
                record = {'stage': stage, 'representation': representation.name, 'C': C,
                          'fold_seed': seed, 'fold': fold, 'features': X.shape[1],
                          **metrics, 'train_accuracy': float(model.score(X, y_fit)),
                          'optimizer_retries': retries}
                results[C].append(record)
                all_fits.append(record)
        candidates = []
        for C, records in results.items():
            aggregate = {f'cv_{name}': float(np.mean([row[name] for row in records]))
                         for name in ('accuracy', 'macro_f1', 'log_loss')}
            candidate = {'stage': stage, 'representation': representation.name, 'C': C, **aggregate,
                         'cv_accuracy_sd': float(np.std([row['accuracy'] for row in records], ddof=1)),
                         'train_accuracy': float(np.mean([row['train_accuracy'] for row in records])),
                         'mean_features': float(np.mean([row['features'] for row in records]))}
            candidates.append(candidate)
            summary.append(candidate)
        if verbose:
            best = rank(candidates)[0]
            print(f"{stage} | {representation.name} | C={best['C']:g} | CV {best['cv_accuracy']:.2%}", flush=True)

    evaluate(BASELINE, 'regularization')
    for representation in TEXT_CHOICES:
        evaluate(representation, 'text')
    leader = representations[rank(summary)[0]['representation']]
    for name, change in (
        ('colour_lexicon', {'colour_lexicon': True}),
        ('coverage_flags', {'coverage_flags': True}),
        ('without_song', {'fields': tuple(f for f in FIELDS if f != 'song')}),
        ('without_feeling', {'fields': tuple(f for f in FIELDS if f != 'feeling')}),
    ):
        evaluate(replace(leader, name=f'{leader.name}_{name}', **change), 'ablation')
    best = rank(summary)[0]
    return representations[best['representation']], best, rank(summary), all_fits


def fit_and_score(fitting, held, labels, held_labels, representation, C):
    params = fit_parameters(fitting, representation)
    X, V = pred.features(fitting, params), pred.features(held, params)
    model, warnings, retries, _ = fit(X, labels, C)
    params.update(weights=model.coef_.T.tolist(), bias=model.intercept_.tolist(),
                  classes=model.classes_.tolist())
    return params, model, {'train': score(model, X, labels), 'held': score(model, V, held_labels),
                           'features': X.shape[1], 'optimizer_iterations': int(model.n_iter_[0]),
                           'optimizer_retries': retries, 'warnings': warnings}


def stress(rows, scenario):
    stressed = [row.copy() for row in rows]
    for row in stressed:
        for field in FIELDS:
            if scenario == 'missing_song':
                if field == 'song':
                    row[field] = ''
            elif scenario == 'unseen_song':
                if field == 'song':
                    row[field] = 'qzxv blmft'
            elif scenario == 'punctuation_case':
                row[field] = '  ' + (row[field] or '').upper().replace(' ', '  ') + '  '
    return stressed


def nested_evaluation(rows, labels, groups):
    records = []
    y_observed, group_ids = [], []
    outputs = {kind: [] for kind in ('selected', 'baseline')}
    for _, fold, tr, ho in partition(labels, groups, (811,), 5):
        inner, held = take(rows, tr), take(rows, ho)
        representation, selection, _, _ = search(inner, labels[tr], groups[tr], (311,), 4)
        y_observed.extend(labels[ho])
        group_ids.extend(groups[ho])
        for kind, rep, C in [('selected', representation, selection['C']), ('baseline', BASELINE, 1.)]:
            params, model, metrics = fit_and_score(inner, held, labels[tr], labels[ho], rep, C)
            labels_pred = model.predict(pred.features(held, params))
            outputs[kind].extend(labels_pred)
            metrics['stress_accuracy'] = {name: score(model, pred.features(stress(held, name), params), labels[ho])['accuracy']
                                          for name in ('missing_song', 'unseen_song', 'punctuation_case')}
            records.append({'outer_fold': fold, 'method': kind, 'representation': rep.name, 'C': C, **metrics})
        print(f"nested CV | fold {fold + 1}/5 | {representation.name} C={selection['C']:g}", flush=True)
    observed = np.asarray(y_observed)
    delta = (np.asarray(outputs['selected']) == observed).astype(int) - (np.asarray(outputs['baseline']) == observed)
    ids = np.asarray(group_ids)
    group_deltas = np.array([delta[ids == group].mean() for group in np.unique(ids)])
    boot = np.random.default_rng(311).choice(group_deltas, size=(5000, len(group_deltas)), replace=True).mean(axis=1)
    pooled = {key: {'accuracy': float(accuracy_score(observed, values)),
                    'macro_f1': float(f1_score(observed, values, labels=CLASSES, average='macro'))}
              for key, values in outputs.items()}
    return {'outer_seed': 811, 'outer_folds': 5, 'inner_folds': 4, 'records': records, 'pooled': pooled,
            'paired_accuracy_delta': float(delta.mean()), 'respondent_bootstrap_95_interval': np.quantile(boot, [.025, .975]).tolist(),
            'interval_note': 'Descriptive paired bootstrap of fixed outer predictions, omitting refit uncertainty.'}


def benchmark_and_verify(out, train_path, val_path, model, parameters, train, val):
    parameter_file = out / 'softmax_params.json'
    parameter_file.write_text(json.dumps(parameters, separators=(',', ':')) + '\n', encoding='utf-8')
    # Run the exported entry point next to its final bundle.
    script = out / 'pred.py'
    script.write_bytes((HERE / 'pred.py').read_bytes())
    import importlib.util
    spec = importlib.util.spec_from_file_location('exported_softmax', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for path, rows in [(train_path, train), (val_path, val)]:
        X = pred.features(rows, parameters)
        # Native norm/dot reductions may differ by a few floating-point ulps.
        np.testing.assert_allclose(module.features(rows, module.load_parameters()), X, rtol=1e-12, atol=1e-14)
        np.testing.assert_array_equal(module.predict_all(path), model.predict(X))

    with tempfile.TemporaryDirectory() as folder:
        batch = Path(folder) / 'timing.csv'
        for row_set, scenario in [(train[:60], 'raw'), (stress(train[:60], 'punctuation_case'), 'formatting')]:
            with batch.open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(SHORT_COLUMNS))
                writer.writeheader()
                writer.writerows(row_set)
            expected = model.predict(pred.features(row_set, parameters))
            np.testing.assert_array_equal(module.predict_all(batch), expected)
            if scenario == 'raw':
                times, warm = [], []
                for _ in range(25):
                    module.load_parameters.cache_clear()
                    start = time.perf_counter()
                    labels = module.predict_all(batch)
                    times.append(time.perf_counter() - start)
                    start = time.perf_counter()
                    module.predict_all(batch)
                    warm.append(time.perf_counter() - start)
                    assert len(labels) == 60
    latency = {'rows': 60, 'repeats': 25, 'includes': 'CSV reading, parameter loading, preprocessing and prediction',
               'first_call_median_seconds': float(np.median(times)), 'first_call_p95_seconds': float(np.percentile(times, 95)),
               'first_call_max_seconds': max(times), 'cached_median_seconds': float(np.median(warm)),
               'under_one_second_all_trials': max(times) < 1,
               'bundle_bytes': script.stat().st_size + parameter_file.stat().st_size,
               'parity_rows_train_and_validation': len(train) + len(val)}
    return latency


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New output directory')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    train_path = ROOT / 'data/splits/train.csv'
    val_path = ROOT / 'data/splits/val.csv'
    train, y, groups = read(train_path)
    assert len(train) == 606 and len(set(groups)) == 202

    # One thread matches the proposal runtime and avoids overhead for small matrices.
    with threadpool_limits(limits=1):
        chosen, selection, summary, folds = search(train, y, groups, FOLD_SEEDS, 5, verbose=True)
        pd.DataFrame(summary).to_csv(out / 'search_summary.csv', index=False)
        pd.DataFrame(folds).to_csv(out / 'cv_folds.csv', index=False)
        nested = nested_evaluation(train, y, groups)
        write_json(out / 'nested_cv.json', nested)
        configuration = {'model': 'multinomial_softmax_regression', 'optimizer': 'L-BFGS',
                         'regularization': 'L2', 'C': selection['C'], 'tolerance': 1e-4,
                         'max_iterations': 2000, 'retry_max_iterations': 10000,
                         'preprocessing': asdict(chosen), 'tfidf_min_df': 2,
                         'fold_seeds': FOLD_SEEDS, 'folds_per_seed': 5}
        write_json(out / 'selected_config.json', configuration)

        # The decision above is fixed before reading this split.
        val, y_val, val_groups = read(val_path)
        assert len(val) == 132 and len(set(val_groups)) == 44 and set(groups).isdisjoint(val_groups)
        parameters, model, metrics = fit_and_score(train, val, y, y_val, chosen, selection['C'])
        base_params, base_model, base_metrics = fit_and_score(train, val, y, y_val, BASELINE, 1.)
        predictions = model.predict(pred.features(val, parameters))
        base_predictions = base_model.predict(pred.features(val, base_params))
        archive = pd.read_csv(ROOT / 'analysis/out/models/validation_predictions.csv')
        recorded = archive.query('family == "softmax"').sort_values('source_row').predicted.to_numpy()
        archive_match = int(np.sum(base_predictions != recorded))
        assert archive_match == 0, 'Baseline prediction parity with the archived run failed'
        cm = confusion_matrix(y_val, predictions, labels=CLASSES)
        pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out / 'validation_confusion.csv')
        latency = benchmark_and_verify(out, train_path, val_path, model, parameters, train, val)
    base_cv = next(row for row in summary if row['representation'] == 'baseline' and row['C'] == 1.)
    report = {'configuration': configuration, 'search_candidates': len(summary), 'search_fits': len(folds),
              'selected_cv': selection, 'baseline_cv': base_cv, 'nested_cv': nested['pooled'],
              'nested_paired_accuracy_delta': nested['paired_accuracy_delta'],
              'validation': {**metrics['held'], 'correct': int(np.sum(predictions == y_val)), 'rows': len(val),
                             'per_class': classification_report(y_val, predictions, labels=CLASSES, output_dict=True, zero_division=0)},
              'baseline_validation': base_metrics['held'], 'train': metrics['train'],
              'features': metrics['features'], 'optimizer_iterations': metrics['optimizer_iterations'],
              'optimizer_retries': metrics['optimizer_retries'],
              'cv_optimizer_retries': sum(row['optimizer_retries'] for row in folds),
              'baseline_archive_prediction_differences': archive_match,
              'latency': latency, 'local_test_accessed': False, 'fit_split': 'train',
              'python': platform.python_version(), 'machine': platform.machine(),
              'packages': {name: importlib.metadata.version(name) for name in
                           ('numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib', 'threadpoolctl')},
              'source_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
              'sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (HERE / 'pred.py', HERE / 'train.py', train_path, val_path,
                                      ROOT / 'analysis/model_experiments.py', ROOT / 'data_preprocessing/encode_all.py')}}
    write_json(out / 'metrics.json', report)
    print(f"Selected {chosen.name} C={selection['C']:g} | validation {metrics['held']['accuracy']:.2%}", flush=True)
    print(f"60-row max inference: {latency['first_call_max_seconds']:.4f}s", flush=True)


if __name__ == '__main__':
    main()

"""Inference cost of the four forest versions, measured rather than assumed.

The project's constraint is that inference may not use scikit-learn, so the time that
matters is not `model.predict` but a traversal written by hand. This script measures both,
on the same 132-row batch, for every version:

    preprocess     building the feature matrix for 132 rows
    sklearn        RandomForestClassifier.predict, for reference only
    traversal      NumPy-only descent over the exported trees, which is what the
                   deliverable has to do

The traversal is checked against sklearn's predictions first, so a timing that comes from
a wrong implementation cannot be reported.

Nothing is read but `data/splits/train.csv`. The 132 rows used as the timing batch are the
first 132 rows of the training split, which is the same size as the held-out splits
without touching either of them.

Run: python measure_inference.py
"""
import importlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
TRAIN_CSV = ROOT / 'data' / 'splits' / 'train.csv'

BATCH_ROWS = 132
N_ESTIMATORS = 300
MODEL_SEED = 311
REPEATS = 7
BUDGET_SECONDS = 1.0

# Modules whose names collide across versions; cleared before each version is imported.
COLLIDING = ('features', 'encode', 'text', 'selection', 'model_experiments', 'encode_all')

VERSIONS = {
    'report': {
        'path': [ROOT / 'analysis', ROOT / 'data_preprocessing'],
        'module': 'model_experiments', 'class': 'SurveyPreprocessor',
        'transform': dict(scaled=False),
        'params': {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1},
    },
    'v1': {
        'path': [ROOT / 'train' / 'random_forest', ROOT / 'analysis',
                 ROOT / 'data_preprocessing'],
        'module': 'features', 'class': 'AblationPreprocessor',
        'transform': dict(scaled=False),
        'kwargs': dict(text_fields=('food', 'colour', 'song', 'feeling'),
                       keep_structured=True),
        'params': {'max_depth': 6, 'max_features': 'log2', 'min_samples_leaf': 3},
    },
    'v2': {
        'path': [ROOT / 'train' / 'random_forest_v2'],
        'module': 'features', 'class': 'TextPreprocessor',
        'transform': dict(),
        'params': {'max_depth': 8, 'max_features': 'log2', 'min_samples_leaf': 2},
    },
    'v3': {
        'path': [ROOT / 'train' / 'random_forest_v3'],
        'module': 'features', 'class': 'TextPreprocessor',
        'transform': dict(),
        'params': {'max_depth': 8, 'max_features': 'sqrt', 'min_samples_leaf': 2},
    },
}


def build_preprocessor(spec):
    """Import one version's pipeline with the colliding module names cleared first."""
    for name in COLLIDING:
        sys.modules.pop(name, None)
    sys.path[:] = [p for p in sys.path if Path(p).name not in
                   ('random_forest', 'random_forest_v2', 'random_forest_v3',
                    'analysis', 'data_preprocessing')]
    for path in reversed(spec['path']):
        sys.path.insert(0, str(path))
    module = importlib.import_module(spec['module'])
    return getattr(module, spec['class'])(**spec.get('kwargs', {}))


def flatten(model):
    """The forest as plain arrays: what a standard-library predictor would hold.

    Each node's class counts are normalised, because the forest's probability averages one
    vote per tree rather than one vote per training sample in a leaf.
    """
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        counts = tree.value[:, 0, :].astype(np.float64)
        totals = counts.sum(axis=1, keepdims=True)
        weights = counts / np.where(totals > 0.0, totals, 1.0)
        trees.append((
            tree.children_left.astype(np.int32),
            tree.children_right.astype(np.int32),
            tree.feature.astype(np.int32),
            tree.threshold.astype(np.float64),
            weights,
        ))
    return trees


def traverse(trees, matrix, classes):
    """Descend every tree for every row using NumPy only. No scikit-learn at this stage."""
    rows = np.arange(matrix.shape[0])
    votes = np.zeros((matrix.shape[0], len(classes)))
    for children_left, children_right, feature, threshold, value in trees:
        node = np.zeros(matrix.shape[0], dtype=np.int32)
        while True:
            leaf = children_left[node] < 0
            if leaf.all():
                break
            column = np.where(leaf, 0, feature[node])
            left = matrix[rows, column] <= threshold[node]
            nxt = np.where(left, children_left[node], children_right[node])
            node = np.where(leaf, node, nxt)
        votes += value[node]
    return np.asarray(classes)[votes.argmax(axis=1)]


def median_time(function, repeats=REPEATS):
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        times.append(time.perf_counter() - started)
    return float(np.median(times))


def measure(name, spec, train):
    preprocessor = build_preprocessor(spec)
    preprocessor.fit(train)
    matrix = preprocessor.transform(train, **spec['transform'])
    columns = len(getattr(preprocessor, 'columns_', [])) or matrix.shape[1]

    hold = train.iloc[:BATCH_ROWS]
    batch = preprocessor.transform(hold, **spec['transform'])

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, criterion='gini', bootstrap=True,
        class_weight=None, oob_score=False, n_jobs=1, random_state=MODEL_SEED,
        **spec['params'])
    model.fit(matrix, train['label'].to_numpy())
    trees = flatten(model)

    sklearn_predictions = model.predict(batch)
    traversal_predictions = traverse(trees, batch, model.classes_)
    if not np.array_equal(sklearn_predictions, traversal_predictions):
        differing = int((sklearn_predictions != traversal_predictions).sum())
        raise AssertionError(f'{name}: traversal differs on {differing} of {len(batch)} rows')

    node_counts = [len(tree[0]) for tree in trees]
    depths = []
    for children_left, children_right, _, _, _ in trees:
        depth = 0
        stack = [(0, 0)]
        while stack:
            node, level = stack.pop()
            if children_left[node] < 0:
                depth = max(depth, level)
            else:
                depth = max(depth, level + 1)
                stack.append((int(children_left[node]), level + 1))
                stack.append((int(children_right[node]), level + 1))
        depths.append(depth)

    single = preprocessor.transform(hold.iloc[:1], **spec['transform'])
    everything = preprocessor.transform(train, **spec['transform'])
    return {
        'version': name,
        'features': int(columns),
        'params': spec['params'],
        'trees': len(trees),
        'nodes_total': int(sum(node_counts)),
        'nodes_max_tree': int(max(node_counts)),
        'depth_max': int(max(depths)),
        'depth_mean': float(np.mean(depths)),
        'predict_132s': median_time(lambda: model.predict(batch)),
        'traverse_132s': median_time(lambda: traverse(trees, batch, model.classes_)),
        'traverse_1s': median_time(lambda: traverse(trees, single, model.classes_)),
        'traverse_606s': median_time(lambda: traverse(trees, everything, model.classes_)),
        'preprocess_132s': median_time(lambda: preprocessor.transform(
            hold, **spec['transform']), repeats=3),
        'predictions_checked': int(len(batch)),
        'classes': [str(name) for name in model.classes_],
    }


def main():
    train = pd.read_csv(TRAIN_CSV, dtype=str, keep_default_na=False)
    print(f'train split {len(train)} rows; timing batch {BATCH_ROWS} rows; '
          f'{N_ESTIMATORS} trees; budget {BUDGET_SECONDS:.1f}s\n')

    results = []
    for name, spec in VERSIONS.items():
        results.append(measure(name, spec, train))
        print(f'  measured {name}', flush=True)

    print(f'\n{"version":8s} {"cols":>5s} {"nodes":>8s} {"depth":>6s} '
          f'{"sklearn":>9s} {"traverse":>9s} {"prep":>8s} {"total":>8s} {"per row":>9s}')
    for row in results:
        total = row['traverse_132s'] + row['preprocess_132s']
        print(f'{row["version"]:8s} {row["features"]:5d} {row["nodes_total"]:8d} '
              f'{row["depth_max"]:6d} {row["predict_132s"] * 1e3:8.2f}m '
              f'{row["traverse_132s"] * 1e3:8.2f}m {row["preprocess_132s"] * 1e3:7.2f}m '
              f'{total * 1e3:7.2f}m {total / BATCH_ROWS * 1e6:8.1f}u')

    print('\nscaling of the hand-written traversal')
    print(f'{"version":8s} {"1 row":>9s} {"132 rows":>10s} {"606 rows":>11s}')
    for row in results:
        print(f'{row["version"]:8s} {row["traverse_1s"] * 1e6:8.0f}u '
              f'{row["traverse_132s"] * 1e3:9.2f}m {row["traverse_606s"] * 1e3:10.2f}m')

    worst = max(results, key=lambda row: row['traverse_132s'] + row['preprocess_132s'])
    worst_total = worst['traverse_132s'] + worst['preprocess_132s']
    print(f'\nslowest version: {worst["version"]} at {worst_total * 1e3:.2f} ms for '
          f'{BATCH_ROWS} rows')
    print(f'budget {BUDGET_SECONDS:.1f}s: {BUDGET_SECONDS / worst_total:,.0f}x headroom, '
          f'about {BUDGET_SECONDS / (worst_total / BATCH_ROWS):,.0f} rows could be scored '
          f'in budget')
    print(f'\nthe traversal agreed with sklearn on all {BATCH_ROWS} rows of every version')

    out = ROOT / 'random_forest_deliverables' / 'inference_time.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'batch_rows': BATCH_ROWS, 'n_estimators': N_ESTIMATORS,
        'model_seed': MODEL_SEED, 'repeats': REPEATS,
        'budget_seconds': BUDGET_SECONDS,
        'timing_batch': 'the first 132 rows of the training split',
        'traversal_matches_sklearn': True,
        'versions': results,
        'slowest': {'version': worst['version'],
                    'seconds_for_batch': worst_total,
                    'headroom': BUDGET_SECONDS / worst_total},
        'test_accessed': False, 'validation_accessed': False,
        'splits_read': ['data/splits/train.csv'],
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name)
                     for name in ('numpy', 'pandas', 'scikit-learn')},
    }, indent=2), encoding='utf-8')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()

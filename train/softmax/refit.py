"""Fit the selected softmax on train+validation and evaluate the canonical local test."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from threadpoolctl import threadpool_limits

import pred
from train import (
    CLASSES, HERE, ROOT, Representation, benchmark_and_verify, fit, fit_parameters,
    read, score, write_json,
)


ARCHIVES = {
    'PyTorch MLP': ROOT / 'MLP_training/out/test_eval',
    'Random forest v1': ROOT / 'random_forest_deliverables/forest_final_v1',
}


def compare_archives(groups, actual, predictions):
    """Verify the published baselines were scored on these exact respondent/label rows."""
    comparison = {}
    for name, folder in ARCHIVES.items():
        archived = pd.read_csv(folder / 'test_predictions.csv', dtype=str, keep_default_na=False)
        np.testing.assert_array_equal(archived['id'].to_numpy(), groups)
        np.testing.assert_array_equal(archived['actual'].to_numpy(), actual)
        metrics = json.loads((folder / 'test_metrics.json').read_text())
        archived_labels = archived['predicted'].to_numpy()
        assert set(archived_labels) <= set(CLASSES)
        correct = int(np.sum(archived_labels == actual))
        np.testing.assert_allclose(correct / len(actual), metrics['accuracy'], rtol=0, atol=1e-12)
        comparison[name] = {
            'correct': correct, 'rows': len(actual), 'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'],
            'prediction_source': str((folder / 'test_predictions.csv').relative_to(ROOT)),
            'respondent_and_label_order_matches': True,
        }
    comparison['Softmax'] = {'correct': int(np.sum(predictions == actual)), 'rows': len(actual)}
    return comparison


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New output directory')
    out = parser.parse_args().out.resolve()
    config_path = HERE / 'results/selected_config.json'
    config = json.loads(config_path.read_text())
    representation = Representation(**config['preprocessing'])
    assert config['model'] == 'multinomial_softmax_regression'
    assert config['optimizer'] == 'L-BFGS' and config['regularization'] == 'L2'
    assert config['tfidf_min_df'] == 2 and config['tolerance'] == 1e-4
    assert config['max_iterations'] == 2000 and config['retry_max_iterations'] == 10000

    train_path, val_path, test_path = [ROOT / 'data/splits' / f'{split}.csv' for split in ('train', 'val', 'test')]
    train, y_train, g_train = read(train_path)
    val, y_val, g_val = read(val_path)
    assert len(train) == 606 and len(val) == 132
    assert set(g_train).isdisjoint(g_val)
    fitting, y_fit = train + val, np.concatenate((y_train, y_val))
    fitting_groups = set(g_train) | set(g_val)
    assert len(fitting) == 738 and len(fitting_groups) == 246
    out.mkdir(parents=True, exist_ok=False)

    with threadpool_limits(limits=1):
        parameters = fit_parameters(fitting, representation)
        X = pred.features(fitting, parameters)
        model, warnings, retries, _ = fit(X, y_fit, config['C'])
        parameters.update(weights=model.coef_.T.tolist(), bias=model.intercept_.tolist(),
                          classes=model.classes_.tolist())
        train_metrics = score(model, X, y_fit)
        latency = benchmark_and_verify(out, train_path, val_path, model, parameters, train, val)
        print(f'Frozen refit complete: C={config["C"]:g}, {len(fitting)} rows, {X.shape[1]} features', flush=True)

        # Open the local test after fitting and export are complete.
        test, actual, test_groups = read(test_path)
        assert len(test) == 132 and len(set(test_groups)) == 44
        assert fitting_groups.isdisjoint(test_groups)
        test_matrix = pred.features(test, parameters)
        predictions = model.predict(test_matrix)
        test_metrics = score(model, test_matrix, actual)
        spec = importlib.util.spec_from_file_location('refitted_softmax_export', out / 'pred.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        np.testing.assert_array_equal(module.predict_all(test_path), predictions)

    comparison = compare_archives(test_groups, actual, predictions)
    comparison['Softmax'].update(accuracy=test_metrics['accuracy'], macro_f1=test_metrics['macro_f1'])
    cm = confusion_matrix(actual, predictions, labels=CLASSES)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out / 'test_confusion.csv')
    report = {
        'configuration': config,
        'fit': {'split': 'train+validation', 'rows': len(fitting), 'respondents': len(fitting_groups),
                'features': X.shape[1], **train_metrics},
        'local_test': {'rows': len(test), 'respondents': len(set(test_groups)),
                       'correct': comparison['Softmax']['correct'], **test_metrics,
                       'per_class': classification_report(actual, predictions, labels=CLASSES,
                                                          output_dict=True, zero_division=0)},
        'same_split_comparison': comparison,
        'optimizer': {'iterations': int(model.n_iter_[0]), 'retries': retries, 'warnings': warnings},
        'export': {**latency, 'parity_rows_local_test': len(test),
                   'parameter_sha256': digest(out / 'softmax_params.json')},
        'source_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'python': platform.python_version(),
        'sha256': {str(path.relative_to(ROOT)): digest(path)
                   for path in (Path(__file__).resolve(), HERE / 'train.py', HERE / 'pred.py', config_path,
                                train_path, val_path, test_path)},
    }
    write_json(out / 'test_metrics.json', report)
    print(json.dumps(comparison, indent=2), flush=True)
    print(f'Slowest first call for 60 rows: {latency["first_call_max_seconds"]:.3f}s', flush=True)


if __name__ == '__main__':
    main()

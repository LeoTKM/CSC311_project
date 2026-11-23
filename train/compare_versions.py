"""Put the four forest versions on one table, with the paired statistics.

The versions are `report` (the model in `analysis/report/proposal.tex`), `v1`, `v2` and
`v3`. They are comparable because every one of them was scored on the same 606 training
rows and the same 132 validation rows, using the same fold constructor
(`GroupKFold(n_splits=5, shuffle=True, random_state=seed)`), the same tree count and the
same model seed. The folds therefore pair up exactly and the validation rows pair up by
row.

Two numbers must not be mixed. The report quotes a five-fold CV of 0.9127; the other three
quote fifteen folds. The like-for-like number for `report` is its fifteen-fold reading of
0.9082, which `train/random_forest_report/reproduce.py` computes.

Three comparisons are produced, because they answer different questions:

    1  selected configuration   what each version would ship, on its own terms
    2  one fixed configuration  the same hyperparameters in all four feature spaces
    3  paired against report    per-fold and per-row differences from the report's forest

Read-only: nothing here fits a model. It reads the artifacts each version already wrote.

Run: python compare_versions.py
"""
import json
import math
from pathlib import Path

import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
OUT = ROOT / 'random_forest_deliverables'

REPORT_OUT = OUT / 'report_forest'
V1_SEARCH = OUT / 'forest_search_v1'
V1_VALIDATION = OUT / 'forest_validation_v1'
V2 = OUT / 'v2_forest'
V3 = OUT / 'v3_forest'

REPORT_PARAMS = {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1}
FOLD_COLUMNS = ['fold_seed', 'fold', 'accuracy', 'macro_f1', 'train_accuracy']


def table_path(directory, *names):
    for name in names:
        path = Path(directory) / name
        if path.exists():
            return path
    raise FileNotFoundError(f'none of {names} in {directory}')


def selected_records(path, params, columns=None):
    """The per-fold rows for one configuration, ordered by fold seed then fold."""
    frame = pd.read_csv(path)
    frame = frame[[json.loads(text) == params for text in frame['params']]].copy()
    if frame.empty:
        raise ValueError(f'{params} is not in {path}')
    wanted = list(columns or FOLD_COLUMNS)
    extra = [name for name in ('n_features',) if name in frame.columns]
    frame = frame.sort_values(['fold_seed', 'fold'])
    return frame[wanted + extra].reset_index(drop=True)


def load_predictions(path):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return frame[['id', 'actual', 'predicted']]


def load_versions():
    versions = {}

    numbers = json.loads((REPORT_OUT / 'reproduction.json').read_text())['reproduced_numbers']
    versions['report'] = {
        'params': REPORT_PARAMS,
        'records': pd.read_csv(REPORT_OUT / 'cv15_records.csv')[FOLD_COLUMNS]
                          .sort_values(['fold_seed', 'fold']).reset_index(drop=True),
        'n_features': int(numbers['n_features']),
        'validation': load_predictions(REPORT_OUT / 'validation_predictions.csv'),
        'validation_accuracy': float(numbers['validation_accuracy']),
        'validation_macro_f1': float(numbers['validation_macro_f1']),
        'selection': "given by the report; its selector used validation to choose "
                     "between three families, not among forest configurations",
    }

    rationale = json.loads(table_path(V1_SEARCH, 'selection_rationale.json').read_text())
    metrics = json.loads(table_path(V1_VALIDATION, 'validation_metrics.json').read_text())
    records = selected_records(table_path(V1_SEARCH, 'cv_folds.csv'),
                               rationale['chosen_params'])
    versions['v1'] = {
        'params': rationale['chosen_params'],
        'records': records,
        'n_features': int(records['n_features'].max()) if 'n_features' in records else None,
        'validation': load_predictions(table_path(V1_VALIDATION,
                                                  'validation_predictions.csv')),
        'validation_accuracy': float(metrics['accuracy']),
        'validation_macro_f1': float(metrics['macro_f1']),
        'selection': 'one-standard-error band, simplest configuration inside it',
    }

    for name, directory in (('v2', V2), ('v3', V3)):
        chosen = json.loads(table_path(directory, 'selection.json').read_text())
        metrics = json.loads(table_path(directory, 'validation_metrics.json').read_text())
        versions[name] = {
            'params': chosen['chosen_params'],
            'records': selected_records(table_path(directory, 'search_records.csv'),
                                        chosen['chosen_params']),
            'n_features': int(chosen['chosen_n_features']),
            'validation': load_predictions(table_path(directory,
                                                      'validation_predictions.csv')),
            'validation_accuracy': float(metrics['accuracy']),
            'validation_macro_f1': float(metrics['macro_f1']),
            'selection': 'one-standard-error band, simplest configuration inside it',
        }
    return versions


def sign_test(wins, losses):
    """Two-sided exact binomial p for wins against losses, ties already removed."""
    trials = wins + losses
    if trials == 0:
        return 1.0
    extreme = sum(math.comb(trials, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * extreme / 2 ** trials)


def paired_folds(left, right, column='accuracy'):
    merged = left.merge(right, on=['fold_seed', 'fold'], suffixes=('_a', '_b'))
    if len(merged) != len(left) or len(left) != len(right):
        raise ValueError('the fold sets do not pair up')
    return merged[f'{column}_a'].to_numpy(), merged[f'{column}_b'].to_numpy()


def summary_row(name, data):
    records = data['records']
    accuracy = records['accuracy'].to_numpy()
    in_fold = records['train_accuracy'].to_numpy()
    return {
        'version': name,
        'features': data['n_features'],
        'params': json.dumps(data['params'], sort_keys=True),
        'cv': accuracy.mean(),
        'cv_sd': accuracy.std(ddof=1),
        'train_in_fold': in_fold.mean(),
        'gap': in_fold.mean() - accuracy.mean(),
        'val': data['validation_accuracy'],
        'val_macro_f1': data['validation_macro_f1'],
        'val_errors': int((data['validation']['actual']
                           != data['validation']['predicted']).sum()),
    }


FORMAT = {'cv': '{:.4f}'.format, 'cv_sd': '{:.4f}'.format,
          'train_in_fold': '{:.4f}'.format, 'gap': '{:+.4f}'.format,
          'val': '{:.4f}'.format, 'val_macro_f1': '{:.4f}'.format}


def main():
    versions = load_versions()
    names = list(versions)
    print(f'{len(names)} versions, {len(versions["report"]["records"])} folds each, '
          f'sharing one training split and one validation split\n')

    # 1. what each version would ship
    print('1. selected configuration, fifteen folds on the 606 training rows')
    table = pd.DataFrame([summary_row(name, versions[name]) for name in names])
    print(table.to_string(index=False, formatters=FORMAT))
    for name in names:
        print(f'   {name:7s} selected by {versions[name]["selection"]}')

    # 2. one configuration, four feature spaces: this isolates the feature work
    print(f'\n2. the report\'s configuration in every feature space '
          f'{json.dumps(REPORT_PARAMS, sort_keys=True)}')
    isolated = []
    for name in names:
        directories = {'v1': V1_SEARCH, 'v2': V2, 'v3': V3}
        if name == 'report':
            records = versions['report']['records']
        else:
            records = selected_records(
                table_path(directories[name], 'cv_folds.csv', 'search_records.csv'),
                REPORT_PARAMS)
        accuracy = records['accuracy'].to_numpy()
        isolated.append({'version': name, 'features': versions[name]['n_features'],
                         'cv': accuracy.mean(), 'cv_sd': accuracy.std(ddof=1),
                         'train_in_fold': records['train_accuracy'].mean()})
    print(pd.DataFrame(isolated).to_string(
        index=False, formatters={'cv': '{:.4f}'.format, 'cv_sd': '{:.4f}'.format,
                                 'train_in_fold': '{:.4f}'.format}))

    # 3. paired differences against the report
    print('\n3. paired against report: 15 folds and 132 validation rows')
    reference = versions['report']
    for name in names[1:]:
        other = versions[name]
        mine, theirs = paired_folds(reference['records'], other['records'])
        difference = mine - theirs
        wins = int((difference > 1e-9).sum())
        losses = int((difference < -1e-9).sum())
        ties = int(len(difference) - wins - losses)
        standard_error = (difference.std(ddof=1) / math.sqrt(len(difference))
                          if wins + losses else float('nan'))
        joined = reference['validation'].merge(other['validation'], on=['id', 'actual'],
                                               suffixes=('_r', '_o'))
        right_reference = (joined['actual'] == joined['predicted_r']).to_numpy()
        right_other = (joined['actual'] == joined['predicted_o']).to_numpy()
        only_reference = int((right_reference & ~right_other).sum())
        only_other = int((~right_reference & right_other).sum())
        print(f'   report - {name}:  CV {difference.mean():+.4f} '
              f'(paired SE {standard_error:.4f}, t p {stats.ttest_rel(mine, theirs).pvalue:.2f})'
              f'   {wins} better / {ties} tied / {losses} worse'
              f'   sign p {sign_test(wins, losses):.2f}')
        print(f'   {"":15s}val: only report right {only_reference}, '
              f'only {name} right {only_other}, '
              f'McNemar p {sign_test(only_reference, only_other):.2f}')

    print('\n   Soccer recall on the 44 validation rows')
    for name in names:
        part = versions[name]['validation']
        soccer = part[part['actual'] == 'Soccer']
        right = int((soccer['actual'] == soccer['predicted']).sum())
        print(f'   {name:7s} {right:2d}/{len(soccer)} = {right / len(soccer):.4f}')

    print('\nnone of the four versions reads the test split')


if __name__ == '__main__':
    main()

"""Does dropping Q4 hurt Baseball?

Diagnostic only: writes nothing. Reads the training split only, 606 rows from 202
respondents, and scores with respondent-grouped cross-validation.

The earlier pass measured Soccer and Basketball recall only. In a three-class model a
column can be useless for one pair and essential for another, so this version keeps the
full confusion matrix and reports every class.

Part 1 measures where Q4 sits on each pair of classes.
Part 2 drops the column and reports all three recalls plus the confusion counts.
Part 3 does the same for the two-class problems that involve Baseball.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT / 'train' / 'random_forest_v2', ROOT / 'data_preprocessing'):
    sys.path.insert(0, str(path))

import features   # noqa: E402
import selection  # noqa: E402

COLUMN = 'individual'
CLASSES = selection.CLASSES
SEEDS = (311, 312, 313)
N_SPLITS = 5
PARAMS = {'max_depth': 8, 'min_samples_leaf': 2, 'max_features': 'log2'}

train = pd.read_csv(ROOT / 'data/splits/train.csv', dtype=str, keep_default_na=False)


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    ranks = pd.Series(np.concatenate([pos, neg])).rank(method='average').to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


pre = features.TextPreprocessor().fit(train)
full = pd.DataFrame(pre.transform(train), columns=pre.columns_, index=train.index)
colour_pre = features.TextPreprocessor(text_fields=('colour',)).fit(train)
colour = pd.DataFrame(colour_pre.transform(train), columns=colour_pre.columns_,
                      index=train.index)
struct_pre = features.TextPreprocessor(text_fields=(), keep_structured=True).fit(train)
structured = pd.DataFrame(struct_pre.transform(train), columns=struct_pre.columns_,
                          index=train.index)
closed = pd.concat([structured, colour], axis=1)
closed = closed.loc[:, ~closed.columns.duplicated()]

print(f'training split: {len(train)} rows, {train.id.nunique()} respondents')
print('Q4 mean by class: '
      + ', '.join(f'{c} {structured.loc[train.label == c, COLUMN].mean():.3f}'
                  for c in CLASSES))

# ---------------------------------------------------------------- 1. where Q4 sits
print()
print('=' * 96)
print('1. WHERE Q4 SITS: RANK SEPARATION ON EVERY PAIR')
print('=' * 96)
values = structured[COLUMN].to_numpy(dtype=float)
labels = train.label.to_numpy()
print(f'{"comparison":34s} {"AUC":>6s} {"|AUC-.5|":>9s}')
for name, pos, neg in (
    ('Soccer vs Basketball', 'Soccer', 'Basketball'),
    ('Soccer vs Baseball', 'Soccer', 'Baseball'),
    ('Baseball vs Basketball', 'Baseball', 'Basketball'),
    ('Baseball vs Soccer+Basketball', 'Baseball', None),
    ('Soccer vs Baseball+Basketball', 'Soccer', None),
    ('Basketball vs Soccer+Baseball', 'Basketball', None),
):
    if neg is None:
        mask = np.isin(labels, [c for c in CLASSES if c != pos])
        value = auc(values[labels == pos], values[mask])
    else:
        value = auc(values[labels == pos], values[labels == neg])
    print(f'{name:34s} {value:6.3f} {abs(value - 0.5):9.3f}')

# ---------------------------------------------------------------- 2. ablation
print()
print('=' * 96)
print('2. DROP THE COLUMN, KEEP EVERYTHING ELSE, REPORT EVERY CLASS')
print('=' * 96)


def run(matrix, keep, used):
    frame = train.reset_index(drop=True)
    data = matrix.reset_index(drop=True)
    subset = frame.label.isin(used).to_numpy()
    frame, data = frame[subset].reset_index(drop=True), data[subset].reset_index(drop=True)
    y = frame.label.to_numpy()
    columns = list(data.columns) if keep else [c for c in data.columns if c != COLUMN]
    X = data[columns].to_numpy(dtype=float)
    records = []
    for seed in SEEDS:
        for fold, (fit_index, held_index) in enumerate(
                selection.folds_of(frame, seed, N_SPLITS)):
            model = selection.make_forest(PARAMS, seed).fit(X[fit_index], y[fit_index])
            predicted = model.predict(X[held_index])
            truth = y[held_index]
            row = {'seed': seed, 'fold': fold, 'n_columns': len(columns),
                   'accuracy': float((predicted == truth).mean())}
            for cls in used:
                mask = truth == cls
                row['recall_' + cls] = (float((predicted[mask] == cls).mean())
                                        if mask.any() else np.nan)
            for actual in used:
                for guess in used:
                    row[actual + ' called ' + guess] = int(
                        ((truth == actual) & (predicted == guess)).sum())
            records.append(row)
    return pd.DataFrame(records)


def compare(name, matrix, used):
    without, with_ = run(matrix, False, used), run(matrix, True, used)
    paired = with_.accuracy.to_numpy() - without.accuracy.to_numpy()
    print(f'\n--- {name}   ({with_.n_columns.iloc[0]} columns with Q4, '
          f'{without.n_columns.iloc[0]} without)')
    print(f'{"variant":14s} {"accuracy":>9s}'
          + ''.join(f'{"recall " + cls:>18s}' for cls in used))
    for label, table in (('without Q4', without), ('with Q4', with_)):
        line = f'{label:14s} {table.accuracy.mean():9.4f}'
        for cls in used:
            line += f'{table["recall_" + cls].mean():18.4f}'
        print(line)
    line = f'{"difference":14s} {paired.mean():+9.4f}'
    for cls in used:
        line += f'{(with_["recall_" + cls].mean() - without["recall_" + cls].mean()):+18.4f}'
    print(line)
    print(f'  folds where keeping Q4 is better: {(paired > 0).sum()}/{len(paired)}, '
          f'worse {(paired < 0).sum()}, tied {(paired == 0).sum()}')
    print(f'  {"confusion":30s} {"without Q4":>11s} {"with Q4":>9s} {"diff":>7s}')
    for actual in used:
        for guess in used:
            if actual == guess:
                continue
            key = actual + ' called ' + guess
            a, b = without[key].mean(), with_[key].mean()
            flag = '   <-' if abs(b - a) >= 0.5 else ''
            print(f'  {actual + " called " + guess:30s} {a:11.2f} {b:9.2f} '
                  f'{b - a:+7.2f}{flag}')
    return without, with_


compare('three classes, full matrix', full, CLASSES)
compare('three classes, closed-form block', closed, CLASSES)
compare('three classes, structured only', structured, CLASSES)

print()
print('=' * 96)
print('3. THE TWO-CLASS PROBLEMS THAT INVOLVE BASEBALL')
print('=' * 96)
compare('Baseball vs Basketball, closed-form', closed, ('Baseball', 'Basketball'))
compare('Baseball vs Soccer, closed-form', closed, ('Soccer', 'Baseball'))
compare('Baseball vs Basketball, structured', structured, ('Baseball', 'Basketball'))
compare('Baseball vs Soccer, structured', structured, ('Soccer', 'Baseball'))

"""Do the multiple-choice answers separate Soccer from Basketball?

Diagnostic only: trains nothing that is kept, writes nothing. Reads the training split
only, 606 rows from 202 respondents.

The three-class model loses almost all of its accuracy on Soccer rows that get called
Basketball: 5 of its 10 validation errors, with Basketball recovered perfectly. This
script asks whether the closed-form questions carry anything that separates those two
classes specifically:

  1. per column, the rate or level in each class and the rank separation (AUC) on the
     Soccer-versus-Basketball sub-problem,
  2. how far a two-class model gets from the structured columns alone, from the colour
     block alone, and from both,
  3. the same for the multi-select questions one at a time, so the cost of each is visible.

AUC is the probability that a randomly drawn Soccer row ranks above a randomly drawn
Basketball row on that column; 0.5 means the column cannot tell them apart at all.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT / 'train' / 'random_forest_v2', ROOT / 'data_preprocessing'):
    sys.path.insert(0, str(path))

import features   # noqa: E402
import selection  # noqa: E402

PAIR = ('Soccer', 'Basketball')
SEEDS = (311, 312, 313)
N_SPLITS = 5
KW = dict(max_depth=8, min_samples_leaf=2, max_features='log2')


def auc(pos, neg):
    """Probability that a random positive outranks a random negative, ties counted half."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    order = np.concatenate([pos, neg])
    ranks = pd.Series(order).rank(method='average').to_numpy()
    rank_pos = ranks[:len(pos)].sum()
    return float((rank_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


train = pd.read_csv(ROOT / 'data/splits/train.csv', dtype=str, keep_default_na=False)
print(f'training split: {len(train)} rows, {train.id.nunique()} respondents')
print('labels: ' + ', '.join(f'{k} {v}' for k, v in train.label.value_counts().items()))

pre = features.TextPreprocessor(text_fields=(), keep_structured=True).fit(train)
structured = pd.DataFrame(pre.transform(train), columns=pre.columns_, index=train.index)
labels = train.label.to_numpy()

pair = train.label.isin(PAIR)
sub = structured[pair]
y_pair = labels[pair]
print(f'\n{PAIR[0]} versus {PAIR[1]}: {len(sub)} rows, '
      f'{train.loc[pair, "id"].nunique()} respondents')
print(f'  {PAIR[0]} {int((y_pair == PAIR[0]).sum())} rows, '
      f'{PAIR[1]} {int((y_pair == PAIR[1]).sum())} rows')

# ---------------------------------------------------------------- 1. per column
print()
print('=' * 96)
print('1. EVERY STRUCTURED COLUMN ON THE SOCCER-VERSUS-BASKETBALL SUB-PROBLEM')
print('=' * 96)
print('sorted by separation; AUC 0.5 means the column cannot tell them apart')
print(f'{"column":26s} {"kind":10s} {"Soccer":>8s} {"Basketb":>8s} {"AUC":>6s} '
      f'{"|AUC-.5|":>9s} {"V(3class)":>10s}')
rows = []
for column in structured.columns:
    soccer = sub.loc[y_pair == PAIR[0], column].to_numpy(dtype=float)
    basket = sub.loc[y_pair == PAIR[1], column].to_numpy(dtype=float)
    value = auc(soccer, basket)
    kind = 'indicator' if set(np.unique(structured[column])) <= {0.0, 1.0} else 'level'
    three = pd.Series(labels)
    table = np.zeros((len(np.unique(structured[column])), 3))
    rows.append({'column': column, 'kind': kind,
                 'soccer': soccer.mean(), 'basketball': basket.mean(),
                 'auc': value, 'off': abs(value - 0.5)})
ordered = pd.DataFrame(rows).sort_values('off', ascending=False)
for row in ordered.itertuples():
    marker = '  <-' if row.off > 0.15 else ''
    print(f'{row.column:26s} {row.kind:10s} {row.soccer:8.3f} {row.basketball:8.3f} '
          f'{row.auc:6.3f} {row.off:9.3f}{marker}')
print(f'\ncolumns with |AUC - 0.5| > 0.15: {int((ordered.off > 0.15).sum())} of {len(ordered)}')
print(f'columns with |AUC - 0.5| > 0.10: {int((ordered.off > 0.10).sum())} of {len(ordered)}')
print(f'columns with |AUC - 0.5| < 0.05: {int((ordered.off < 0.05).sum())} of {len(ordered)}')

# ---------------------------------------------------------------- 2. blocks
print()
print('=' * 96)
print('2. HOW FAR DO THE CLOSED-FORM ANSWERS GET ON THIS SUB-PROBLEM')
print('=' * 96)
colour_pre = features.TextPreprocessor(text_fields=('colour',), keep_structured=False).fit(train)
colour = pd.DataFrame(colour_pre.transform(train), columns=colour_pre.columns_, index=train.index)
multi = [column for column in structured.columns
         if column.startswith(('location_', 'season_', 'music_'))]
ordinal = ['ease', 'strategy', 'injury', 'individual', 'payment_amount',
           'payment_missing']

BLOCKS = {
    'everything closed-form (34)': list(structured.columns),
    'structured only, no flags (27)': [c for c in structured.columns if 'missing' not in c],
    'Q5 venue (7)': [c for c in multi if c.startswith('location_')],
    'Q6 season (4)': [c for c in multi if c.startswith('season_')],
    'Q7 music (10)': [c for c in multi if c.startswith('music_')],
    'Q5+Q6+Q7 multi-select (21)': multi,
    'Q1-Q4 ordinal (4)': ordinal[:4],
    'Q12 payment (2)': ordinal[4:],
    'colour block (20)': list(colour.columns),
    'colour + structured (54)': list(colour.columns) + list(structured.columns),
}

sub_frame = train[pair].reset_index(drop=True)
sub_y = labels[pair]
closed = pd.concat([structured, colour], axis=1)
for name, columns in BLOCKS.items():
    matrix = closed.loc[pair, columns].reset_index(drop=True)
    scores = []
    for seed in SEEDS:
        splitter = GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fit_index, held_index in splitter.split(sub_frame, sub_y, sub_frame['id']):
            model = RandomForestClassifier(n_estimators=300, criterion='gini',
                                           bootstrap=True, random_state=seed, n_jobs=1,
                                           oob_score=False, **KW)
            model.fit(matrix.iloc[fit_index], sub_y[fit_index])
            scores.append((model.predict(matrix.iloc[held_index]) == sub_y[held_index]).mean())
    scores = np.asarray(scores)
    print(f'  {name:32s} {matrix.shape[1]:3d} cols   '
          f'accuracy {scores.mean():.4f} +- {scores.std():.4f}')

print('\nfor reference, always guessing one class on this sub-problem: 0.5000')

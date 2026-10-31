"""Q9 colour: fixed multi-hot over a colour lexicon, versus a selected vocabulary.

Both options produce a multi-hot vector - the question is only which columns it has and
who decides them. Three candidates:

  chi2_perfold   the plan's proposal: pool min_df=3, ranked by chi2 inside each fold,
                 top 10 kept. The column set changes from fold to fold.
  chi2_fixed     the same ranking computed once on all training rows, then frozen.
  lexicon        a hand-written colour lexicon, every colour that appears at all,
                 with spelling variants merged. 17 columns, always the same.

Plus a count-weighting variant of the lexicon, and each option with and without the
item-count column, so the comparison answers "multi-hot or term frequency" too.

Diagnostic only: nothing is written, no v2 artifact is produced. Grouped 5-fold over the
606-row training split, seeded 311/312/313, the same protocol the model comparison uses.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'train' / 'random_forest_v2'))
sys.path.insert(0, str(ROOT / 'data_preprocessing'))

import text as v2text                                # noqa: E402
from encode_all import TEXT, clean, raw_headers      # noqa: E402

CLASSES = ['Soccer', 'Baseball', 'Basketball']
KW = dict(hyphen='space', min_len=2, require_letter=True)
MIN_DF = 3
TOPK = 10
SEEDS = (311, 312, 313)

STOP = {'and', 'the', 'of', 'in', 'at', 'on', 'for', 'with', 'or', 'is', 'are', 'be',
        'as', 'it', 'this', 'that', 'a', 'an', 'some', 'maybe', 'would', 'which', 'one',
        'itself', 'also', 'all', 'no', 'not', 'most', 'to', 'my', 'i', 'because', 'sport',
        'sports', 'colour', 'color', 'colors', 'colours', 'associate', 'team', 'teams',
        'player', 'players', 'field', 'rainbow', 'major', 'none', 'local', 'toronto'}

LEXICON = {
    'blue': {'blue', 'blues'},
    'red': {'red'},
    'orange': {'orange'},
    'green': {'green'},
    'white': {'white'},
    'black': {'black', 'balck'},
    'brown': {'brown'},
    'yellow': {'yellow', 'yellowish'},
    'gold': {'gold', 'golden'},
    'purple': {'purple'},
    'navy': {'navy'},
    'gray': {'gray', 'grey'},
    'beige': {'beige'},
    'maroon': {'maroon'},
    'burgundy': {'burgundy'},
    'cream': {'cream'},
    'tan': {'tan'},
}

train = pd.read_csv(ROOT / 'data/splits/train.csv', dtype=str, keep_default_na=False)
frame = raw_headers(train)
column = TEXT['colour']
labels = frame.label.to_numpy()
groups = frame.id.to_numpy()
answered = np.array([bool(clean(v)) for v in frame[column]])
docs = [v2text.tokens(v, **KW) for v in frame[column]]
presence = [set(d) for d in docs]
n = len(frame)
counts = {}
for s in presence:
    for token in s:
        counts[token] = counts.get(token, 0) + 1


def chi2_scores(matrix, target):
    scores = np.zeros(matrix.shape[1])
    for j in range(matrix.shape[1]):
        present = matrix[:, j] == 1
        table = np.zeros((2, 3))
        for index, cls in enumerate(CLASSES):
            mask = target == cls
            table[1, index] = (present & mask).sum()
            table[0, index] = (~present & mask).sum()
        expected = table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True) / len(target)
        if (expected == 0).any():
            continue
        scores[j] = ((table - expected) ** 2 / expected).sum()
    return scores


def fit_pool(fit_index):
    pool_counts = {}
    for i in fit_index:
        for token in presence[i]:
            pool_counts[token] = pool_counts.get(token, 0) + 1
    return sorted(t for t, c in pool_counts.items() if c >= MIN_DF and t not in STOP)


def rank(fit_index, pool):
    matrix = np.zeros((len(fit_index), len(pool)), dtype=np.int8)
    for row, i in enumerate(fit_index):
        for token in presence[i]:
            if token in pool:
                matrix[row, pool.index(token)] = 1
    scores = chi2_scores(matrix, labels[fit_index])
    return [pool[j] for j in np.argsort(-scores)]


def project(indices, columns, mode='binary'):
    """Binary presence or token count, for a named list of columns.

    Each column is either a token name or, for the lexicon, a tuple of tokens that all
    mean the same colour.
    """
    out = np.zeros((len(indices), len(columns)), dtype=np.int8)
    for row, i in enumerate(indices):
        for j, spec in enumerate(columns):
            members = spec if isinstance(spec, (set, tuple)) else (spec,)
            hits = sum(1 for token in presence[i] if token in members)
            out[row, j] = int(hits > 0) if mode == 'binary' else hits
    return out


def item_count(indices):
    return np.array([[len(docs[i])] for i in indices], dtype=np.int8)


def evaluate(name, build_columns, mode='binary', with_count=False):
    """build_columns(fit_index) -> list of columns fitted on that fold only."""
    records = []
    for seed in SEEDS:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fit_index, held_index in splitter.split(frame, labels, groups):
            columns = build_columns(fit_index)
            X_fit = project(fit_index, columns, mode)
            X_held = project(held_index, columns, mode)
            if with_count:
                X_fit = np.hstack([X_fit, item_count(fit_index)])
                X_held = np.hstack([X_held, item_count(held_index)])
            y_fit = pd.Series(labels[fit_index]).map(
                {c: i for i, c in enumerate(CLASSES)}).to_numpy()
            y_held = pd.Series(labels[held_index]).map(
                {c: i for i, c in enumerate(CLASSES)}).to_numpy()
            for model_name, model in (
                ('forest', RandomForestClassifier(
                    n_estimators=300, max_depth=6, min_samples_leaf=3,
                    max_features='log2', random_state=seed, n_jobs=1)),
                ('logreg', LogisticRegression(max_iter=2000, C=1.0)),
            ):
                model.fit(X_fit, y_fit)
                predicted = model.predict(X_held)
                records.append(dict(
                    seed=seed, model=model_name, columns=len(columns),
                    accuracy=accuracy_score(y_held, predicted),
                    macro_f1=f1_score(y_held, predicted, average='macro'),
                    all_zero=np.mean(X_held.sum(axis=1) == 0),
                ))
    table = pd.DataFrame(records)
    grouped = table.groupby(['model', 'seed']).agg(
        accuracy=('accuracy', 'mean'), macro_f1=('macro_f1', 'mean'))
    print(f'\n--- {name}   ({table.columns[0] and ""}{len(columns)} columns, mode={mode}'
          f'{", +item_count" if with_count else ""})')
    print(f'{"model":8s} {"seed":>5s} {"accuracy":>10s} {"macro F1":>10s}')
    for (model, seed), row in grouped.iterrows():
        print(f'{model:8s} {seed:5d} {row.accuracy:10.4f} {row.macro_f1:10.4f}')
    mean = grouped.groupby('model').mean()
    for model, row in mean.iterrows():
        print(f'{model:8s} {"mean":>5s} {row.accuracy:10.4f} {row.macro_f1:10.4f}   '
              f'(columns {table.columns[0] and ""}{len(columns)}, '
              f'rows with an all-zero colour block {table.all_zero.mean():.1%})')
    return mean


print(f'{n} training rows, {answered.sum()} answered Q9, {len(counts)} distinct tokens')
print(f'lexicon: {len(LEXICON)} colours, from '
      f'{sum(len(v) for v in LEXICON.values())} tokens')

fixed_pool = fit_pool(np.arange(n))
fixed_ranked = rank(np.arange(n), fixed_pool)

results = {}
results['chi2_perfold'] = evaluate(
    'chi2 per fold, pool min_df=3, top 10',
    lambda fit_index: rank(fit_index, fit_pool(fit_index))[:TOPK])
results['chi2_fixed'] = evaluate(
    'chi2 ranked once on all 606 rows, top 10',
    lambda fit_index: fixed_ranked[:TOPK])
results['lexicon'] = evaluate(
    'colour lexicon, 17 columns, binary',
    lambda fit_index: list(LEXICON.values()))
results['lexicon_count'] = evaluate(
    'colour lexicon, 17 columns, token counts',
    lambda fit_index: list(LEXICON.values()), mode='count')
results['chi2_perfold_count'] = evaluate(
    'chi2 per fold, top 10, token counts',
    lambda fit_index: rank(fit_index, fit_pool(fit_index))[:TOPK], mode='count')
results['lexicon_item'] = evaluate(
    'colour lexicon + item count',
    lambda fit_index: list(LEXICON.values()), with_count=True)
results['chi2_item'] = evaluate(
    'chi2 per fold top 10 + item count',
    lambda fit_index: rank(fit_index, fit_pool(fit_index))[:TOPK], with_count=True)

print()
print('=' * 96)
print('SUMMARY, forest accuracy over the three seeds')
print('=' * 96)
for name, mean in sorted(results.items(), key=lambda kv: -kv[1].loc['forest', 'accuracy']):
    print(f'  {name:22s} forest {mean.loc["forest", "accuracy"]:.4f}   '
          f'logreg {mean.loc["logreg", "accuracy"]:.4f}')

print()
print('=' * 96)
print('THE LEXICON COLUMNS AND WHAT THEY COVER')
print('=' * 96)
print(f'{"column":12s} {"tokens":28s} {"df":>4s} {"cover":>7s}')
for name, tokens in LEXICON.items():
    df = sum(1 for s in presence if s & tokens)
    print(f'{name:12s} {", ".join(sorted(tokens)):28s} {df:4d} {df / n:6.1%}')
print(f'\nfixed chi2 top 10 on all rows: {", ".join(fixed_ranked[:TOPK])}')
print(f'tokens the lexicon merges that the token pool keeps apart: '
      f'gray/grey, gold/golden, blue/blues, black/balck, yellow/yellowish')

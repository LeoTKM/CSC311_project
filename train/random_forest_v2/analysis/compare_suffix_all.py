"""Do the suffix merges that helped feeling also help food and song?

Diagnostic only: nothing is written. Computes each block alone under grouped 5-fold,
seeds 311/312/313, at the K already chosen for that field, with and without plural
folding. Also applies the same plural fold to feeling for a like-for-like comparison.

The plural fold is the simple one: -ies -> -y, -ches/-shes/-sses/-xes/-zes -> strip -es,
otherwise strip a final -s unless the word ends in -ss, -us, -is or -as.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'train' / 'random_forest_v2'))
sys.path.insert(0, str(ROOT / 'data_preprocessing'))

import text as v2text                                # noqa: E402
from encode_all import TEXT, clean, raw_headers       # noqa: E402

CLASSES = ['Soccer', 'Baseball', 'Basketball']
KW = dict(hyphen='space', min_len=2, require_letter=True)
MIN_DF = 3
SEEDS = (311, 312, 313)
K_BY_FIELD = {'food': 30, 'song': 30, 'feeling': 60}

train = pd.read_csv(ROOT / 'data/splits/train.csv', dtype=str, keep_default_na=False)
frame = raw_headers(train)
labels = frame.label.to_numpy()
groups = frame.id.to_numpy()
n = len(frame)


def singular(token):
    if len(token) < 4 or not token.isalpha():
        return token
    if token.endswith('ies') and len(token) > 4:
        return token[:-3] + 'y'
    for suffix in ('ches', 'shes', 'sses', 'xes', 'zes'):
        if token.endswith(suffix):
            return token[:-2]
    if token.endswith('s') and not token.endswith(('ss', 'us', 'is', 'as')):
        return token[:-1]
    return token


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


def evaluate(field, docs, k):
    records = []
    for seed in SEEDS:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fit_index, held_index in splitter.split(frame, labels, groups):
            presence = [set(docs[i]) for i in range(n)]
            pool_counts = {}
            for i in fit_index:
                for token in presence[i]:
                    pool_counts[token] = pool_counts.get(token, 0) + 1
            pool = sorted(t for t, c in pool_counts.items() if c >= MIN_DF)
            matrix = np.zeros((len(fit_index), len(pool)), dtype=np.int8)
            for row, i in enumerate(fit_index):
                for token in presence[i]:
                    if token in pool:
                        matrix[row, pool.index(token)] = 1
            order = np.argsort(-chi2_scores(matrix, labels[fit_index]))
            chosen = [pool[j] for j in order[:min(k, len(pool))]]
            position = {t: j for j, t in enumerate(chosen)}

            def project(indices):
                out = np.zeros((len(indices), len(chosen)), dtype=np.int8)
                for row, i in enumerate(indices):
                    for token in presence[i]:
                        if token in position:
                            out[row, position[token]] = 1
                return out

            X_fit, X_held = project(fit_index), project(held_index)
            y_fit = pd.Series(labels[fit_index]).map(
                {c: i for i, c in enumerate(CLASSES)}).to_numpy()
            y_held = pd.Series(labels[held_index]).map(
                {c: i for i, c in enumerate(CLASSES)}).to_numpy()
            model = RandomForestClassifier(n_estimators=300, max_depth=6,
                                           min_samples_leaf=3, max_features='log2',
                                           random_state=seed, n_jobs=1)
            model.fit(X_fit, y_fit)
            predicted = model.predict(X_held)
            records.append((accuracy_score(y_held, predicted),
                            f1_score(y_held, predicted, average='macro'),
                            len(chosen), np.mean(X_held.sum(axis=1) == 0)))
    table = pd.DataFrame(records, columns=['acc', 'f1', 'cols', 'empty'])
    return table.mean()


print(f'{n} rows\n')
print(f'{"field":9s} {"variant":22s} {"K":>4s} {"cols":>5s} {"forest acc":>11s} '
      f'{"forest F1":>10s} {"no term":>8s}')
for field, k in K_BY_FIELD.items():
    column = TEXT[field]
    docs = [v2text.tokens(v, **KW) for v in frame[column]]
    folded = [[singular(t) for t in doc] for doc in docs]
    for name, variant in (('plain tokens', docs), ('plural folded', folded)):
        mean = evaluate(field, variant, k)
        print(f'{field:9s} {name:22s} {k:4d} {int(mean.cols):5d} {mean.acc:11.4f} '
              f'{mean.f1:10.4f} {mean.empty:7.1%}')
    print()

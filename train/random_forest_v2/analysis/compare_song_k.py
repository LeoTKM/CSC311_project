"""Q10 song: how many terms does the song block need, and does refusal handling help?

Diagnostic only: nothing is written. Grouped 5-fold over the 606-row training split,
seeds 311/312/313, song block alone, so the comparison isolates the song encoding.

Variants:
  K sweep        K = 10, 20, 30, 40, 60, 100, and the whole pool, binary presence
  item count     the best-looking K plus the reply-length column
  widened        refusals blanked before tokenising (none, no idea, no particular song,
                 idk, ...) instead of surviving as fake content
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
from encode_all import TEXT, MISSING, raw_headers     # noqa: E402

CLASSES = ['Soccer', 'Baseball', 'Basketball']
KW = dict(hyphen='space', min_len=2, require_letter=True)
MIN_DF = 3
SEEDS = (311, 312, 313)

# exact refusal forms seen in the song answers, plus the analysis-side set
REFUSAL = {
    'none', 'nothing', 'unsure', 'not sure', 'no idea', 'nope', 'nil', 'none at all',
    'n.a.', '--', 'no song', 'no songs', 'no particular song', 'no particular song.',
    'no particular song comes to mind.', 'none in particular', 'none in particular.',
    'nothing really', 'idk', "don't have", 'dont have', 'no', 'not sure.', 'no.',
}


def clean_plain(value):
    text = ' '.join(str(value).split()).lower()
    return '' if text in MISSING else text


def clean_wide(value):
    text = ' '.join(str(value).split()).lower()
    if text in MISSING or text in REFUSAL:
        return ''
    if text.rstrip('.!') in {r.rstrip('.!') for r in REFUSAL}:
        return ''
    return text


train = pd.read_csv(ROOT / 'data/splits/train.csv', dtype=str, keep_default_na=False)
frame = raw_headers(train)
column = TEXT['song']
labels = frame.label.to_numpy()
groups = frame.id.to_numpy()
n = len(frame)


def build(clean_fn):
    cleaned = [clean_fn(v) for v in frame[column]]
    docs = [v2text.tokens(v, **KW) for v in cleaned]
    return cleaned, docs


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


def evaluate(name, docs, k_values, with_count_for=None, mode='binary'):
    print(f'\n--- {name}')
    print(f'{"K":>5s} {"cols":>5s} {"forest acc":>11s} {"forest F1":>10s} '
          f'{"logreg acc":>11s} {"no term":>8s}')
    for k in k_values:
        records = []
        for seed in SEEDS:
            splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            for fit_index, held_index in splitter.split(frame, labels, groups):
                pool_counts = {}
                for i in fit_index:
                    for token in set(docs[i]):
                        pool_counts[token] = pool_counts.get(token, 0) + 1
                pool = sorted(t for t, c in pool_counts.items() if c >= MIN_DF)
                matrix = np.zeros((len(fit_index), len(pool)), dtype=np.int8)
                for row, i in enumerate(fit_index):
                    for token in set(docs[i]):
                        if token in pool:
                            matrix[row, pool.index(token)] = 1
                order = np.argsort(-chi2_scores(matrix, labels[fit_index]))
                take = len(pool) if k is None else min(k, len(pool))
                chosen = [pool[j] for j in order[:take]]
                position = {t: j for j, t in enumerate(chosen)}

                def project(indices):
                    out = np.zeros((len(indices), len(chosen)), dtype=np.int8)
                    for row, i in enumerate(indices):
                        for token in set(docs[i]):
                            if token in position:
                                out[row, position[token]] = 1
                    return out

                X_fit, X_held = project(fit_index), project(held_index)
                if with_count_for is not None and k == with_count_for:
                    X_fit = np.hstack([X_fit, [[len(docs[i])] for i in fit_index]])
                    X_held = np.hstack([X_held, [[len(docs[i])] for i in held_index]])
                y_fit = pd.Series(labels[fit_index]).map(
                    {c: i for i, c in enumerate(CLASSES)}).to_numpy()
                y_held = pd.Series(labels[held_index]).map(
                    {c: i for i, c in enumerate(CLASSES)}).to_numpy()
                for model_name, model in (
                    ('forest', RandomForestClassifier(
                        n_estimators=300, max_depth=6, min_samples_leaf=3,
                        max_features='log2', random_state=seed, n_jobs=1)),
                    ('logreg', LogisticRegression(max_iter=2000)),
                ):
                    model.fit(X_fit, y_fit)
                    predicted = model.predict(X_held)
                    records.append((model_name, accuracy_score(y_held, predicted),
                                    f1_score(y_held, predicted, average='macro'),
                                    np.mean(X_held[:, :len(chosen)].sum(axis=1) == 0)))
        table = pd.DataFrame(records, columns=['model', 'acc', 'f1', 'empty'])
        grouped = table.groupby('model').mean()
        label = 'all' if k is None else str(k)
        print(f'{label:>5s} {len(chosen):5d} {grouped.loc["forest", "acc"]:11.4f} '
              f'{grouped.loc["forest", "f1"]:10.4f} {grouped.loc["logreg", "acc"]:11.4f} '
              f'{grouped.loc["forest", "empty"]:7.1%}')


plain_clean, plain_docs = build(clean_plain)
wide_clean, wide_docs = build(clean_wide)
print(f'{n} rows')
print(f'plain clean(): {sum(1 for v in plain_clean if v)} answered')
print(f'widened refusal set: {sum(1 for v in wide_clean if v)} answered '
      f'({sum(1 for v in plain_clean if v) - sum(1 for v in wide_clean if v)} more blanked)')

evaluate('min_df=3, binary presence, plain clean()', plain_docs,
         [20, 30, 40, 60, 100, None])
evaluate('min_df=3, binary presence, widened refusals', wide_docs,
         [20, 30, 40, 60, 100, None])
evaluate('K=30 plus the reply-length column, plain clean()', plain_docs,
         [30], with_count_for=30)

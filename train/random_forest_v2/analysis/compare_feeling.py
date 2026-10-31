"""Q11 feeling: how many terms, and which clean-ups actually pay?

Diagnostic only: nothing is written. Grouped 5-fold over the 606-row training split,
seeds 311/312/313, the feeling block alone, so every comparison isolates this field.

Sections, each answering one question:
  1  how many terms K are needed (plain tokens)
  2  the same sweep with suffix variants folded onto one standard form
  3  does a flat-reply flag help, i.e. a reply made only of
     nothing / neutral / indifferent / fine / normal / unsure
  4  does the reply-length column help
  5  does a stopword list help, plain and with the standard form
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
from encode_all import TEXT, clean, raw_headers       # noqa: E402

CLASSES = ['Soccer', 'Baseball', 'Basketball']
KW = dict(hyphen='space', min_len=2, require_letter=True)
MIN_DF = 3
SEEDS = (311, 312, 313)

# the hand-written version of the merge, kept for the like-for-like comparison; the
# rule-based standard form lives in compare_feeling_standard_form.py
MERGE = {
    'excited': {'excited', 'excitement', 'exciting', 'excite', 'excitedly', 'exhilarated'},
    'bored': {'bored', 'boring', 'boredom', 'bore', 'boredness'},
    'relaxed': {'relaxed', 'relaxing', 'relaxation', 'relax'},
    'confused': {'confused', 'confusing', 'confusion', 'confusions', 'confuse'},
    'tense': {'tense', 'tension'},
    'energetic': {'energetic', 'energized', 'energize', 'energy', 'energetized', 'energies'},
    'interested': {'interested', 'interesting', 'interest'},
    'entertained': {'entertained', 'entertaining', 'entertainment'},
    'enjoyable': {'enjoy', 'enjoyable', 'enjoyment', 'enjoying'},
    'nervous': {'nervous', 'nervously', 'nervousness'},
    'anxious': {'anxious', 'anxiety'},
    'frustrated': {'frustrated', 'frustrating', 'frustration'},
    'intense': {'intense', 'intensity', 'intensely'},
    'sad': {'sad', 'sadness'},
}
LOOKUP = {word: canonical for canonical, group in MERGE.items() for word in group}

FLAT_TOKENS = {'nothing', 'neutral', 'indifferent', 'none', 'normal', 'fine', 'unsure',
               'whatever', 'okay', 'no', 'not'}
STOP = {'and', 'the', 'of', 'in', 'at', 'on', 'for', 'with', 'or', 'is', 'are', 'be',
        'as', 'it', 'this', 'that', 'a', 'an', 'to', 'my', 'i', 'but', 'when', 'there',
        'very', 'really', 'good', 'sport', 'sports', 'game', 'games', 'watch',
        'watching', 'feel', 'because', 'much', 'get', 'im', 'just', 'can', 'would'}

train = pd.read_csv(ROOT / 'data/splits/train.csv', dtype=str, keep_default_na=False)
frame = raw_headers(train)
column = TEXT['feeling']
labels = frame.label.to_numpy()
groups = frame.id.to_numpy()
n = len(frame)
answered = np.array([bool(clean(v)) for v in frame[column]])
plain_docs = [v2text.tokens(v, **KW) for v in frame[column]]
merged_docs = [[LOOKUP.get(t, t) for t in doc] for doc in plain_docs]
stopped_plain = [[t for t in doc if t not in STOP] for doc in plain_docs]
stopped_merged = [[t for t in doc if t not in STOP] for doc in merged_docs]

plain_presence = [set(d) for d in plain_docs]
flat = np.array([bool(s) and s <= FLAT_TOKENS and 'bored' not in s for s in plain_presence])
sizes = np.array([len(d) for d in plain_docs])


def cramers_v(indicator):
    table = np.zeros((2, 3))
    for index, cls in enumerate(CLASSES):
        mask = labels == cls
        table[1, index] = indicator[mask].sum()
        table[0, index] = mask.sum() - table[1, index]
    expected = table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True) / n
    if (expected == 0).any():
        return 0.0
    return float(np.sqrt(((table - expected) ** 2 / expected).sum() / n))


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


def evaluate(name, docs, k_values, extra=None):
    print(f'\n--- {name}')
    print(f'{"K":>5s} {"cols":>5s} {"forest acc":>11s} {"forest F1":>10s} '
          f'{"logreg acc":>11s} {"no term":>8s}')
    for k in k_values:
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
                take = len(pool) if k is None else min(k, len(pool))
                chosen = [pool[j] for j in order[:take]]
                position = {t: j for j, t in enumerate(chosen)}

                def project(indices):
                    out = np.zeros((len(indices), len(chosen)), dtype=np.int8)
                    for row, i in enumerate(indices):
                        for token in presence[i]:
                            if token in position:
                                out[row, position[token]] = 1
                    return out

                X_fit, X_held = project(fit_index), project(held_index)
                if extra is not None:
                    X_fit = np.hstack([X_fit, extra[fit_index].reshape(-1, 1).astype(np.int8)])
                    X_held = np.hstack([X_held, extra[held_index].reshape(-1, 1).astype(np.int8)])
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


print(f'{n} rows, answered {answered.sum()}')
print(f'flat-reply flag: {flat.sum()} rows ({flat.mean():.1%}), V={cramers_v(flat):.3f}')
for token in sorted(FLAT_TOKENS):
    mask = np.array([token in s for s in plain_presence])
    if mask.sum():
        print(f'  {token:12s} df={mask.sum():3d} V={cramers_v(mask):.3f}')

evaluate('1. plain tokens, min_df=3', plain_docs, [10, 20, 30, 40, 60, 120, None])
evaluate('2. suffix variants merged (hand-written map)', merged_docs,
         [10, 20, 30, 40, 60, 120, None])
evaluate('3. plain tokens, K=40 plus the flat-reply flag', plain_docs, [40], extra=flat)
evaluate('4. plain tokens, K=40 plus the reply-length column', plain_docs, [40], extra=sizes)
evaluate('5a. plain tokens plus a stopword list', stopped_plain, [60])
evaluate('5b. suffix variants merged plus a stopword list', stopped_merged, [60, 120])

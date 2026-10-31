"""Q8 food, part 3: what actually happens when an answer contains a word the frozen
vocabulary has never seen.

Diagnostic only: no model is trained, nothing is written. Grouped 5-fold CV over the
canonical 606-row training split, so every held-out number below is a legitimate
train-only quantity.

The vocabulary is fitted at fit time and frozen. A token outside it produces a zero
everywhere, exactly as if the respondent had written nothing. This script measures how
often that happens, and separates the two reasons it can happen:
  * the token was never in the training folds                (true out-of-vocabulary)
  * the token was in them but did not survive min_df or chi2  (known but unselected)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'train' / 'random_forest_v2'))
sys.path.insert(0, str(ROOT / 'data_preprocessing'))

import text as v2text                                # noqa: E402
from encode_all import TEXT, clean, raw_headers      # noqa: E402

CLASSES = ['Soccer', 'Baseball', 'Basketball']
FIELD = 'food'
KW = dict(hyphen='keep', min_len=2, require_letter=True)
MIN_DF = 2
TOP_K = 30

train = pd.read_csv(ROOT / 'data/splits/train.csv', dtype=str, keep_default_na=False)
frame = raw_headers(train)
column = TEXT[FIELD]
labels = frame.label.to_numpy()
groups = frame.id.to_numpy()
docs = [v2text.tokens(value, **KW) for value in frame[column]]
answered = np.array([bool(clean(value)) for value in frame[column]])
n = len(frame)
print(f'{n} training rows, {len(set(groups))} respondents, '
      f'{answered.sum()} answered Q8\n')


def chi2_scores(binary_matrix, target):
    """chi2 per column, computed from the 2x3 contingency table."""
    scores = np.zeros(binary_matrix.shape[1])
    for j in range(binary_matrix.shape[1]):
        present = binary_matrix[:, j] == 1
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


rows = []
for fold, (fit_index, held_index) in enumerate(
        GroupKFold(n_splits=5).split(frame, labels, groups), start=1):
    fit_docs = [docs[i] for i in fit_index]
    counts = {}
    for doc in fit_docs:
        for token in set(doc):
            counts[token] = counts.get(token, 0) + 1
    kept_by_df = {t for t, c in counts.items() if c >= MIN_DF}

    vocabulary = sorted(kept_by_df)
    position = {t: j for j, t in enumerate(vocabulary)}
    matrix = np.zeros((len(fit_index), len(vocabulary)), dtype=np.int8)
    for row, i in enumerate(fit_index):
        for token in set(docs[i]):
            if token in position:
                matrix[row, position[token]] = 1
    scores = chi2_scores(matrix, labels[fit_index])
    order = np.argsort(-scores)
    selected = {vocabulary[j] for j in order[:TOP_K] if scores[j] > 0}

    held_answered = [i for i in held_index if answered[i]]
    unseen = list(kept_by_df)
    held_tokens = [set(docs[i]) for i in held_answered]
    flat = [t for s in held_tokens for t in s]
    oov_tokens = [t for t in flat if t not in kept_by_df]
    rows_all_oov = [i for i, s in zip(held_answered, held_tokens)
                    if s and not (s & kept_by_df)]
    rows_no_selected = [i for i, s in zip(held_answered, held_tokens)
                        if not (s & selected)]
    rows.append(dict(
        fold=fold,
        fit_rows=len(fit_index),
        vocab_df=len(kept_by_df),
        held_answered=len(held_answered),
        token_oov=len(oov_tokens) / max(1, len(flat)),
        rows_with_oov=np.mean([bool(s - kept_by_df) for s in held_tokens]),
        rows_all_oov=len(rows_all_oov) / max(1, len(held_answered)),
        rows_no_selected=len(rows_no_selected) / max(1, len(held_answered)),
        oov_types=len({t for t in oov_tokens}),
        oov_examples=', '.join(sorted({t for t in oov_tokens})[:6]),
    ))

table = pd.DataFrame(rows)
print('per fold:')
print(table.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

print()
print('mean over folds:')
for key in ('token_oov', 'rows_with_oov', 'rows_all_oov', 'rows_no_selected'):
    label = {'token_oov': 'content tokens that are out of vocabulary',
             'rows_with_oov': 'answered rows containing at least one unseen word',
             'rows_all_oov': 'answered rows whose every word is unseen',
             'rows_no_selected': f'answered rows with no selected top-{TOP_K} term'
             }[key]
    print(f'  {label:56s} {table[key].mean():.1%}')

# ------------------------------------------------------------------ deployment vocab
print()
print('=' * 96)
print('VOCABULARY POOL (training split only, 606 rows)')
print('=' * 96)
both = frame
both_docs = [v2text.tokens(value, **KW) for value in both[column]]
counts = {}
for doc in both_docs:
    for token in set(doc):
        counts[token] = counts.get(token, 0) + 1
total = len(counts)
print(f'rows {len(both)}   distinct tokens {total}')
for threshold in (2, 3, 5, 10):
    kept = sum(1 for c in counts.values() if c >= threshold)
    print(f'  min_df={threshold:<2d} -> {kept:3d} tokens '
          f'({kept / total:.0%} of the raw vocabulary)')

# how much of a fresh answer survives
raw_only = sum(counts.values())
print(f'\nshare of all token occurrences covered by min_df=2 terms: '
      f'{sum(c for c in counts.values() if c >= 2) / raw_only:.1%}')

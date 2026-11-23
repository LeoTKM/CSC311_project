"""Q9 colour: how many terms does a closed concept need, and does it fragment like food?

Diagnostic only: no model is trained, nothing is written.

Colour is the opposite case to food: few distinct answers, almost all of them a single
colour word. The questions here are
  1. how many terms K are needed before a held-out answer stops landing on nothing,
  2. whether modifiers (light blue, dark blue, sky blue) fragment the vocabulary the way
     hyphenated compounds did for food,
  3. whether multi-colour answers (blue, white) carry a signal of their own,
  4. what the tunable parts should be set to.
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
FIELD = 'colour'
KW = dict(hyphen='space', min_len=2, require_letter=True)
MIN_DF = 3

train = pd.read_csv(ROOT / 'data/splits/train.csv', dtype=str, keep_default_na=False)
frame = raw_headers(train)
column = TEXT[FIELD]
labels = frame.label.to_numpy()
groups = frame.id.to_numpy()
n = len(frame)
answered = np.array([bool(clean(v)) for v in frame[column]])
docs = [v2text.tokens(v, **KW) for v in frame[column]]
print(f'{n} training rows, answered Q9 {answered.sum()}\n')


def show(s):
    return str(s).encode('ascii', 'backslashreplace').decode('ascii')


def cramers_v(indicator):
    table = np.zeros((2, 3))
    for index, cls in enumerate(CLASSES):
        mask = labels == cls
        table[1, index] = indicator[mask].sum()
        table[0, index] = mask.sum() - table[1, index]
    expected = table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True) / n
    if (expected == 0).any():
        return 0.0
    chi2 = ((table - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi2 / n))


counts = {}
presence = []
for doc in docs:
    seen = set(doc)
    presence.append(seen)
    for token in seen:
        counts[token] = counts.get(token, 0) + 1

print('=' * 96)
print('1. VOCABULARY CENSUS, top 40 by document frequency')
print('=' * 96)
print(f'{"token":20s} {"df":>4s} {"cover":>7s} {"V":>6s}   per-class rate  S/B/Bb')
for token, count in sorted(counts.items(), key=lambda kv: -kv[1])[:40]:
    mask = np.array([token in s for s in presence])
    rates = '/'.join(f'{mask[labels == c].mean():.2f}' for c in CLASSES)
    print(f'{show(token):20s} {count:4d} {count / n:6.1%} {cramers_v(mask):6.3f}   {rates}')

print()
print('=' * 96)
print('2. TOP 25 BY CLASS ASSOCIATION')
print('=' * 96)
ranked = sorted(((cramers_v(np.array([t in s for s in presence])), t, c)
                 for t, c in counts.items()), reverse=True)
print(f'{"rank":>4s} {"token":20s} {"df":>4s} {"V":>6s}   per-class rate  S/B/Bb')
for index, (value, token, count) in enumerate(ranked[:25], start=1):
    mask = np.array([token in s for s in presence])
    rates = '/'.join(f'{mask[labels == c].mean():.2f}' for c in CLASSES)
    print(f'{index:4d} {show(token):20s} {count:4d} {value:6.3f}   {rates}')

print()
print('=' * 96)
print('3. MIN_DF AND THE CANDIDATE POOL')
print('=' * 96)
total = len(counts)
for threshold in (2, 3, 5, 8):
    kept = sum(1 for c in counts.values() if c >= threshold)
    print(f'  min_df={threshold:<2d} -> {kept:3d} tokens ({kept / total:3.0%} of {total})')

print()
print('=' * 96)
print('4. MODIFIERS: does "light blue" fragment the vocabulary?')
print('=' * 96)
MODIFIERS = ['light', 'dark', 'bright', 'sky', 'navy', 'royal', 'neon', 'baby', 'pale',
             'deep', 'hot', 'random', 'any', 'the', 'of', 'and', 'is', 'it', 'a',
             'like', 'major', 'colours', 'colors', 'rainbow', 'sort', 'kind']
print(f'{"token":12s} {"df":>4s} {"V":>6s}   present?')
for token in MODIFIERS:
    if token in counts:
        mask = np.array([token in s for s in presence])
        print(f'{token:12s} {counts[token]:4d} {cramers_v(mask):6.3f}   yes')
    else:
        print(f'{token:12s} {"-":>4s} {"-":>6s}   no')
print('\nanswers that carry a modifier (all of them):')
for value in frame.loc[[any(t in MODIFIERS for t in d) for d in docs], column]:
    print(f'  {show(value)}')

print()
print('=' * 96)
print('5. SPELLING VARIANTS (British vs American)')
print('=' * 96)
for pair in (('gray', 'grey'), ('color', 'colour'), ('colors', 'colours')):
    for token in pair:
        mask = np.array([token in s for s in presence])
        print(f'  {token:8s} df={counts.get(token, 0):3d}  '
              f'V={cramers_v(mask) if token in counts else 0:.3f}')
print('  answers mentioning "colour"/"color" at all:',
      int(frame[column].str.lower().str.contains('colo').sum()))

print()
print('=' * 96)
print('6. MULTI-COLOUR ANSWERS')
print('=' * 96)
sizes = np.array([len(s) for s in presence])
print(f'{"tokens per answer":20s} {"rows":>5s}  per-class rate S/B/Bb')
for name, mask in (('0', sizes == 0), ('1', sizes == 1), ('2', sizes == 2),
                   ('3+', sizes >= 3)):
    rates = '/'.join(f'{mask[labels == c].mean():.2f}' for c in CLASSES)
    print(f'{name:20s} {mask.sum():5d}  {rates}')
print(f'\nV of "at least 2 colours": {cramers_v(sizes >= 2):.3f}')

print()
print('=' * 96)
print('7. HOW MANY TERMS ARE NEEDED (grouped 5-fold on the 606 training rows)')
print('=' * 96)
fit_docs = docs
rows = []
for fit_index, held_index in GroupKFold(n_splits=5).split(frame, labels, groups):
    pool_counts = {}
    for i in fit_index:
        for token in set(fit_docs[i]):
            pool_counts[token] = pool_counts.get(token, 0) + 1
    pool = [t for t, c in pool_counts.items() if c >= MIN_DF]
    if not pool:
        continue
    matrix = np.zeros((len(fit_index), len(pool)), dtype=np.int8)
    for row, i in enumerate(fit_index):
        for token in set(fit_docs[i]):
            if token in pool:
                matrix[row, pool.index(token)] = 1
    scores = np.zeros(len(pool))
    for j in range(len(pool)):
        present = matrix[:, j] == 1
        table = np.zeros((2, 3))
        for index, cls in enumerate(CLASSES):
            m = labels[fit_index] == cls
            table[1, index] = (present & m).sum()
            table[0, index] = (~present & m).sum()
        expected = table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True) / len(fit_index)
        if (expected == 0).any():
            continue
        scores[j] = ((table - expected) ** 2 / expected).sum()
    order = np.argsort(-scores)
    held = [i for i in held_index if answered[i]]
    sets = [set(fit_docs[i]) for i in held]
    empty = {k: np.mean([not (s & {pool[j] for j in order[:k]}) for s in sets]) for k in
             (5, 8, 10, 12, 15, 20, 30)}
    rows.append(empty)

print(f'pool size per fold: {[len([t for t, c in {}.items()]) for _ in range(0)] or "see pool min_df=3"}')
print(f'{"K":>4s} {"held-out answered rows with no selected term":>46s}')
for k in (5, 8, 10, 12, 15, 20, 30):
    value = np.mean([r[k] for r in rows])
    print(f'{k:4d} {value:45.1%}')

print()
print('=' * 96)
print('8. NON-ANSWERS AND ODD ANSWERS')
print('=' * 96)
blank = int((~answered).sum())
print(f'blank after clean(): {blank}')
rare = [v for v, d in zip(frame[column], docs) if d and len(d) >= 4]
print(f'answers with 4 or more tokens: {len(rare)}')
for value in rare[:12]:
    print(f'  {show(value)}')

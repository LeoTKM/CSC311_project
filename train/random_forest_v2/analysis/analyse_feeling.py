"""Q11 feeling: the most open of the four answers.

Diagnostic only: no model is trained, nothing is written.

Feeling is not a closed list like colour and it is not a title like song. The same
emotion is written with different endings (`excited`, `exciting`, `excitement`), replies
range from one word to a full paragraph, and a reply can be negated. This script
measures which of those actually matters.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'train' / 'random_forest_v2'))
sys.path.insert(0, str(ROOT / 'data_preprocessing'))

import text as v2text                                # noqa: E402
from encode_all import TEXT, clean, raw_headers       # noqa: E402

CLASSES = ['Soccer', 'Baseball', 'Basketball']
FIELD = 'feeling'
KW = dict(hyphen='space', min_len=2, require_letter=True)
MIN_DF = 3

train = pd.read_csv(ROOT / 'data/splits/train.csv', dtype=str, keep_default_na=False)
frame = raw_headers(train)
both = frame
column = TEXT[FIELD]
labels = frame.label.to_numpy()
groups = frame.id.to_numpy()
n = len(frame)
answered = np.array([bool(clean(v)) for v in frame[column]])
docs = [v2text.tokens(v, **KW) for v in frame[column]]
presence = [set(d) for d in docs]
print(f'{n} training rows, answered Q11={answered.sum()}, '
      f'distinct answers={frame[column].nunique()}\n')


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
    return float(np.sqrt(((table - expected) ** 2 / expected).sum() / n))


counts = {}
for s in presence:
    for token in s:
        counts[token] = counts.get(token, 0) + 1

print('=' * 96)
print('1. VOCABULARY CENSUS, top 45 by document frequency')
print('=' * 96)
print(f'{"token":16s} {"df":>4s} {"cover":>7s} {"V":>6s}   per-class rate  S/B/Bb')
for token, count in sorted(counts.items(), key=lambda kv: -kv[1])[:45]:
    mask = np.array([token in s for s in presence])
    rates = '/'.join(f'{mask[labels == c].mean():.2f}' for c in CLASSES)
    print(f'{show(token):16s} {count:4d} {count / n:6.1%} {cramers_v(mask):6.3f}   {rates}')

print()
print('=' * 96)
print('2. TOP 30 BY CLASS ASSOCIATION')
print('=' * 96)
ranked = sorted(((cramers_v(np.array([t in s for s in presence])), t, c)
                 for t, c in counts.items()), reverse=True)
for index, (value, token, count) in enumerate(ranked[:30], start=1):
    mask = np.array([token in s for s in presence])
    rates = '/'.join(f'{mask[labels == c].mean():.2f}' for c in CLASSES)
    print(f'{index:3d} {show(token):16s} df={count:4d} V={value:6.3f}   {rates}')

print()
print('=' * 96)
print('3. SUFFIX VARIANTS: the same feeling written with different endings')
print('=' * 96)
for stem in ('excit', 'bor', 'relax', 'confus', 'tens', 'nerv', 'happ', 'indiffer',
             'interest', 'neutr', 'stress', 'calm', 'enjoy', 'entertain', 'proud',
             'adren', 'exhaust', 'tir', 'fun', 'loud', 'quiet', 'intens', 'energ',
             'satisf', 'frustrat', 'anxious', 'angry', 'sad', 'disappoint', 'competit'):
    members = sorted(((counts[t], t) for t in counts if t.startswith(stem)), reverse=True)
    if not members:
        continue
    total = sum(1 for s in presence if any(t in s for t in (t for _, t in members)))
    best = max(cramers_v(np.array([t in s for s in presence])) for _, t in members)
    merged = sum(members[0][0] for _ in [0])
    print(f'  {stem:12s} tokens {len(members):2d}  '
          f'{", ".join(f"{t}({c})" for c, t in members[:5]):52s} '
          f'rows {total:3d}  best V {best:.3f}')

print()
print('=' * 96)
print('4. ANSWER LENGTH')
print('=' * 96)
sizes = np.array([len(s) for s in presence])
print(f'answered: mean {sizes[answered].mean():.2f}, median {np.median(sizes[answered]):.0f}, '
      f'max {sizes.max()}')
print(f'{"bucket":8s} {"rows":>5s}  per-class rate S/B/Bb')
for name, mask in (('1', sizes == 1), ('2-3', (sizes >= 2) & (sizes <= 3)),
                   ('4-6', (sizes >= 4) & (sizes <= 6)),
                   ('7-12', (sizes >= 7) & (sizes <= 12)), ('13+', sizes >= 13)):
    rates = '/'.join(f'{mask[labels == c].mean():.2f}' for c in CLASSES)
    print(f'{name:8s} {mask.sum():5d}  {rates}')
print(f'\nV of one-token reply: {cramers_v(sizes == 1):.3f}')
print(f'V of reply with 13+ tokens: {cramers_v(sizes >= 13):.3f}')
print('\nthe 6 longest answers:')
for value in frame.loc[np.argsort(-sizes)[:6], column]:
    print(f'  [{len(value.split()):3d} words] {show(value)[:150]}')

print()
print('=' * 96)
print('5. NEGATION')
print('=' * 96)
NEG = re.compile(r"\b(not|dont|don't|never|no|nothing|nor|hardly|barely)\b", re.I)
negated = np.array([bool(NEG.search(v)) for v in frame[column]])
print(f'replies containing a negation: {negated.sum()} ({negated.mean():.1%})')
for cls in CLASSES:
    print(f'  {cls:11s} {negated[labels == cls].mean():.1%}')
print(f'V of "contains a negation": {cramers_v(negated):.3f}')
print('\nnegated replies that also carry a strong feeling word:')
STRONG = ('excited', 'exciting', 'bored', 'boring', 'happy', 'tense', 'relaxed',
          'nervous', 'interested', 'calm', 'stress')
for value, is_neg, doc in zip(frame[column], negated, docs):
    if is_neg and any(t in STRONG for t in doc):
        print(f'  {show(value)[:110]}')

print()
print('=' * 96)
print('6. LOW-INFORMATION REPLIES ("nothing", "neutral", "indifferent")')
print('=' * 96)
FLAT = {'nothing', 'neutral', 'indifferent', 'none', 'normal', 'fine', 'bored',
        'boring', 'unsure', 'whatever', 'okay'}
flat = np.array([bool(s) and s <= FLAT for s in presence])
print(f'replies made only of flat words: {flat.sum()} ({flat.mean():.1%})')
for cls in CLASSES:
    print(f'  {cls:11s} {flat[labels == cls].mean():.1%}')
print(f'V of "flat reply": {cramers_v(flat):.3f}')
for token in ('nothing', 'neutral', 'indifferent', 'bored', 'boring', 'fine', 'normal'):
    if token in counts:
        print(f'  {token:12s} df={counts[token]:3d} V={cramers_v(np.array([token in s for s in presence])):.3f}')

print()
print('=' * 96)
print('7. HOW MANY TERMS ARE NEEDED (grouped 5-fold on the 606 training rows)')
print('=' * 96)
records = []
for fit_index, held_index in GroupKFold(n_splits=5).split(frame, labels, groups):
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
    sets = [presence[i] for i in held]
    records.append((len(pool),
                    {k: np.mean([not (s & {pool[j] for j in order[:k]}) for s in sets])
                     for k in (10, 20, 30, 40, 60, 80, 120)}))
print(f'pool size per fold (min_df=3): {sorted(r[0] for r in records)}')
print(f'{"K":>5s} {"held-out answered rows with no selected term":>46s}')
for k in (10, 20, 30, 40, 60, 80, 120):
    print(f'{k:5d} {np.mean([r[1][k] for r in records]):45.1%}')

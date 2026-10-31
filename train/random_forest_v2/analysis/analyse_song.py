"""Q10 song: an open field whose answers are song titles.

Diagnostic only: no model is trained, nothing is written.

Song is the opposite of colour and harder than food, for four reasons this script
measures:
  1. the same song is written many ways, and the variants share tokens,
  2. the discriminative tokens are title fragments, so one title can occupy several of
     the K slots,
  3. a large share of replies are semantically "no song", and the ones the current
     clean() does not catch survive as fake content,
  4. replies contain `[this sport]` style anonymisation placeholders, whose words leak
     into the vocabulary as if they were content.
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
sys.path.insert(0, str(ROOT / 'analysis'))

import text as v2text                                # noqa: E402
from encode_all import TEXT, MISSING, clean, raw_headers      # noqa: E402
from survey_common import NON_ANSWER                          # noqa: E402

CLASSES = ['Soccer', 'Baseball', 'Basketball']
FIELD = 'song'
KW = dict(hyphen='space', min_len=2, require_letter=True)
MIN_DF = 3
BRACKET = re.compile(r'\[[^\]]*\]')

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
print(f'{n} training rows, answered Q10={answered.sum()}, '
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
print('1. VOCABULARY CENSUS, top 40 by document frequency')
print('=' * 96)
print(f'{"token":16s} {"df":>4s} {"cover":>7s} {"V":>6s}   per-class rate  S/B/Bb')
for token, count in sorted(counts.items(), key=lambda kv: -kv[1])[:40]:
    mask = np.array([token in s for s in presence])
    rates = '/'.join(f'{mask[labels == c].mean():.2f}' for c in CLASSES)
    print(f'{show(token):16s} {count:4d} {count / n:6.1%} {cramers_v(mask):6.3f}   {rates}')

print()
print('=' * 96)
print('2. TOP 25 BY CLASS ASSOCIATION')
print('=' * 96)
ranked = sorted(((cramers_v(np.array([t in s for s in presence])), t, c)
                 for t, c in counts.items()), reverse=True)
for index, (value, token, count) in enumerate(ranked[:25], start=1):
    mask = np.array([token in s for s in presence])
    rates = '/'.join(f'{mask[labels == c].mean():.2f}' for c in CLASSES)
    print(f'{index:3d} {show(token):16s} df={count:4d} V={value:6.3f}   {rates}')

print()
print('=' * 96)
print('3. THE SAME SONG WRITTEN DIFFERENT WAYS')
print('=' * 96)
normalised = [' '.join(sorted(s)) for s in presence]
series = pd.Series(normalised)
tally = series[answered].value_counts()
print(f'distinct normalised answers among answered rows: {tally.size}')
print('15 most frequent:')
for value, count in tally.head(15).items():
    print(f'  {count:4d}  {value!r}')
print('\nraw variants of the two dominant songs:')
for marker in ('waka', 'take'):
    variants = pd.Series(
        [clean(v).lower() for v in frame[column] if marker in clean(v).lower()])
    print(f'  answers containing {marker!r}: {variants.size}')
    for value, count in variants.value_counts().head(9).items():
        print(f'    {count:4d}  {show(value)!r}')

print()
print('=' * 96)
print('4. HOW MANY K SLOTS ONE TITLE CONSUMES')
print('=' * 96)
SLOTS = 40
top = [t for _, t, _ in ranked[:SLOTS]]
for cluster, members in (
    ('take me out to the ball game', ['take', 'me', 'out', 'to', 'the', 'ball', 'game']),
    ('waka waka', ['waka', 'shakira', 'africa', 'time', 'for', 'this']),
):
    inside = [t for t in members if t in top]
    print(f'  {cluster:30s} occupies {len(inside):2d} of the top {SLOTS}: '
          f'{", ".join(inside) if inside else "none"}')
print(f'\n  rows mentioning "take" AND "ball": '
      f'{sum(1 for s in presence if {"take", "ball"} <= s)}')
print(f'  rows mentioning "waka": {sum(1 for s in presence if "waka" in s)}')
print(f'  rows mentioning "shakira": {sum(1 for s in presence if "shakira" in s)}')
print('\n  candidate bigrams, against their strongest single part:')
bigram = {}
for doc in docs:
    for a, b in zip(doc, doc[1:]):
        pair = f'{a} {b}'
        bigram[pair] = bigram.get(pair, 0) + 1
for pair, count in sorted(bigram.items(), key=lambda kv: -kv[1])[:10]:
    a, b = pair.split()
    print(f'    {pair:22s} df={count:3d}  parts {a}({counts[a]}) {b}({counts[b]})')

print()
print('=' * 96)
print('5. REPLIES THAT MEAN "NO SONG"')
print('=' * 96)
caught = (~answered).sum()
print(f'caught by clean() as a non-answer: {caught}')
print(f'the NON_ANSWER set: {sorted(NON_ANSWER)}')
loose = []
for value in frame[column]:
    plain = clean(value).lower().strip()
    if not plain:
        continue
    if plain.startswith(('no ', 'none', 'nothing', 'not ', "don't", 'dont', 'unsure',
                         'idk', 'n / a', 'n.a')):
        loose.append(value)
print(f'\nanswered but reads like a refusal ({len(loose)} rows):')
for value, count in pd.Series(loose).value_counts().head(14).items():
    print(f'  {count:3d}  {show(value)!r}')
print(f'\ntokens those refusals contribute: '
      + ', '.join(f'{t}({counts.get(t, 0)})'
                  for t in ('no', 'not', 'song', 'particular', 'idea', 'none', 'nothing')))

print()
print('=' * 96)
print('6. PLACEHOLDER LEAK, ALL FOUR FIELDS')
print('=' * 96)
print('replies contain bracketed anonymisation spans like [this sport]; step 4 splits them')
print(f'{"field":10s} {"rows with [ or ]":>18s} {"placeholder words left as tokens":>40s}')
for field in ('food', 'colour', 'song', 'feeling'):
    hits = both[TEXT[field]].map(lambda v: bool(BRACKET.search(v)))
    words = []
    for token in ('this', 'sport', 'sports', 'team', 'teams', 'league', 'teams'):
        pass
    print(f'{field:10s} {hits.sum():18d}')
    for value in both.loc[hits, TEXT[field]].head(3):
        print(f'{"":12s}{show(value)}')

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
                     for k in (20, 30, 40, 50, 60, 80, 100)}))
print(f'pool size per fold (min_df=3): '
      f'{sorted(r[0] for r in records)}')
print(f'{"K":>5s} {"held-out answered rows with no selected term":>46s}')
for k in (20, 30, 40, 50, 60, 80, 100):
    print(f'{k:5d} {np.mean([r[1][k] for r in records]):45.1%}')

print()
print('=' * 96)
print('8. ANSWER LENGTH')
print('=' * 96)
sizes = np.array([len(s) for s in presence])
print(f'answered rows: mean {sizes[answered].mean():.2f} tokens, '
      f'median {np.median(sizes[answered]):.0f}, max {sizes.max()}')
for name, mask in (('1', sizes == 1), ('2-3', (sizes >= 2) & (sizes <= 3)),
                   ('4-6', (sizes >= 4) & (sizes <= 6)), ('7+', sizes >= 7)):
    rates = '/'.join(f'{mask[labels == c].mean():.2f}' for c in CLASSES)
    print(f'  {name:5s} rows {mask.sum():4d}   S/B/Bb {rates}')

"""Q8 food, part 2: edge hyphens across all four fields, cost of dropping generic words,
and how many v1 slots (df-based top 50) are wasted on non-content tokens.

Diagnostic only: nothing is trained, nothing is written. Reads the training split only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'train' / 'random_forest_v2'))
sys.path.insert(0, str(ROOT / 'data_preprocessing'))

import text as v2text                                # noqa: E402
from encode_all import TEXT, raw_headers             # noqa: E402

CLASSES = ['Soccer', 'Baseball', 'Basketball']
KW = dict(min_len=2, require_letter=True)
df = raw_headers(pd.read_csv(ROOT / 'data/splits/train.csv', dtype=str,
                             keep_default_na=False))
labels = df.label.to_numpy()
n = len(df)


def show(s):
    return str(s).encode('ascii', 'backslashreplace').decode('ascii')


def cramers_v(indicator):
    table = np.zeros((2, 3))
    for index, cls in enumerate(CLASSES):
        mask = labels == cls
        table[1, index] = indicator[mask].sum()
        table[0, index] = mask.sum() - table[1, index]
    expected = table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True) / n
    chi2 = ((table - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi2 / (n * min(1, 2))))


# ------------------------------------------------------- a. edge hyphens, all fields
print('=' * 96)
print('a. EVERY TOKEN WITH A LEADING OR TRAILING HYPHEN, ALL FOUR FIELDS')
print('=' * 96)
for field in ('food', 'colour', 'song', 'feeling'):
    docs = [v2text.tokens(v, hyphen='keep', **KW) for v in df[TEXT[field]]]
    counts = {}
    for doc in docs:
        for token in set(doc):
            counts[token] = counts.get(token, 0) + 1
    edge = {t: c for t, c in counts.items() if t.startswith('-') or t.endswith('-')}
    lone = [v for v in df[TEXT[field]] if v.strip() == '-']
    print(f'{field:8s} hyphenated tokens {sum(1 for t in counts if "-" in t):3d}   '
          f'edge hyphen {len(edge):2d}   answers that are just "-": {len(lone)}')
    for token, count in sorted(edge.items(), key=lambda kv: -kv[1]):
        print(f'         {count:4d}  {show(token)!r}')

# ------------------------------------------------------- b. generic-word stoplist
print()
print('=' * 96)
print('b. COST OF ALSO DROPPING THE GENERIC / QUESTION-ECHO WORDS')
print('=' * 96)
CONNECTIVE = {'and', 'with', 'a', 'the', 'or', 'of', 'to', 'at', 'on', 'in', 'for', 'my',
              'i', 'some', 'like', 'just', 'is', 'it', 'this', 'that', 'be'}
GENERIC = {'associate', 'watching', 'watch', 'sport', 'sports', 'game', 'games',
           'food', 'foods', 'drink', 'drinks', 'snack', 'snacks', 'soft', 'when',
           'while', 'also', 'generally', 'stuff', 'things', 'thing'}
for stop, name in ((CONNECTIVE, 'connectives only'),
                   (CONNECTIVE | GENERIC, 'connectives + generic')):
    docs = [v2text.tokens(v, hyphen='keep', **KW) for v in df[TEXT['food']]]
    kept = [{t for t in doc if t not in stop} for doc in docs]
    sizes = np.array([len(s) for s in kept])
    print(f'{name:24s} rows with 0 tokens: {(sizes == 0).sum():3d}   '
          f'mean kept tokens {sizes.mean():.2f}   '
          f'distinct tokens {len({t for s in kept for t in s})}')
    empty_rows = df.loc[sizes == 0, 'label'].value_counts().to_dict()
    if empty_rows:
        print(f'{"":24s} label of the emptied rows: {empty_rows}')
        print(f'{"":24s} their raw answers:')
        for value in df.loc[sizes == 0, TEXT['food']].head(8):
            print(f'{"":26s}{show(value)!r}')

# ------------------------------------------------------- c. wasted v1 slots
print()
print('=' * 96)
print('c. v1 SELECTS BY DOCUMENT FREQUENCY, SO IT WASTES SLOTS ON NON-CONTENT WORDS')
print('=' * 96)
docs = [v2text.tokens(v, hyphen='keep', **KW) for v in df[TEXT['food']]]
counts = {}
presence = []
for doc in docs:
    seen = set(doc)
    presence.append(seen)
    for token in seen:
        counts[token] = counts.get(token, 0) + 1
NOISE = CONNECTIVE | GENERIC
top50 = [t for t, _ in sorted(counts.items(), key=lambda kv: -kv[1]) if _ >= 1][:50]
noise_in_top50 = [t for t in top50 if t in NOISE]
print('v1 food cap is 50 terms chosen by document frequency.')
print(f'  non-content tokens inside that top 50: {len(noise_in_top50)}'
      f' ({len(noise_in_top50) / 50:.0%} of the slots)')
for token in noise_in_top50:
    print(f'    {show(token):16s} df={counts[token]:4d} '
          f'V={cramers_v(np.array([token in s for s in presence])):.3f}')
print(f'\n  v1 also keeps min_df=2, so of the {len(counts)} distinct tokens only '
      f'{sum(1 for c in counts.values() if c >= 2)} survive min_df.')

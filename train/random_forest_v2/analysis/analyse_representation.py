"""Should v2 keep TF-IDF at all? Compare candidate text representations on the v1 rows.

Diagnostic only: no model is trained, nothing is written.

Three questions:
  1. For a random forest, which parts of TF-IDF actually change the information?
  2. Do the tokens the current cap keeps (highest document frequency) carry the class
     signal, or do the discriminative tokens sit below the cap?
  3. What would a small, curated vocabulary cost in coverage?
"""
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'train' / 'random_forest_v2'))
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'data_preprocessing'))

import text as v2text                                     # noqa: E402
from encode_all import TEXT, clean, raw_headers           # noqa: E402

CLASSES = ['Soccer', 'Baseball', 'Basketball']
CAPS = {'food': 50, 'colour': 20, 'song': 50, 'feeling': 50}
KW = dict(hyphen='space', min_len=2, require_letter=True)

train = pd.read_csv(ROOT / 'data/splits/train.csv', dtype=str, keep_default_na=False)
df = raw_headers(train)
labels = df.label.to_numpy()
print(f'rows {len(df)}  respondents {df.id.nunique()}\n')


def cramers_v(present):
    """Association between a 0/1 token indicator and the three classes."""
    table = np.zeros((2, 3))
    for index, cls in enumerate(CLASSES):
        mask = labels == cls
        table[1, index] = present[mask].sum()
        table[0, index] = mask.sum() - table[1, index]
    if table.sum() == 0 or (table.sum(axis=1) == 0).any():
        return 0.0
    n = table.sum()
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    if (expected <= 0).any():
        return 0.0
    chi2 = ((table - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi2 / n))          # min(k-1, r-1) == 1


print('=== 1. the three TF-IDF components, per-feature vs cross-feature ===')
print('  sublinear_tf  1+log(tf)     per-feature monotone -> a tree sees the same splits')
print('  smooth_idf    x idf         per-feature constant -> monotone, same splits')
print("  norm='l2'     x / ||row||   CROSS-feature: the value depends on the other")
print('                              tokens of that field in the same row -> the meaning')
print('                              of a split threshold changes row by row')
print('  => for a forest only the L2 row normalisation changes the information;')
print('     counts or binary presence would carry the same signal.\n')

print('=== 2. document frequency vs class association (presence vs label) ===')
rows = []
for field in TEXT:
    docs = Counter()
    present = {}
    for index, value in enumerate(df[TEXT[field]]):
        got = set(v2text.tokens(clean(value), **KW))
        docs.update(got)
        for token in got:
            if token not in present:
                present[token] = np.zeros(len(df), dtype=bool)
            present[token][index] = True
    scored = [(token, docs[token], cramers_v(present[token]))
              for token in present if docs[token] >= 5]
    by_freq = sorted(scored, key=lambda t: -t[1])[:CAPS[field]]
    by_v = sorted(scored, key=lambda t: -t[2])
    print(f'  -- {field} (cap {CAPS[field]}; df>=5 tokens: {len(scored)}) --')
    print(f'     kept by document frequency: {[(t, c) for t, c, _ in by_freq[:12]]}')
    print(f'     mean V of those {len(by_freq)}: '
          f'{np.mean([v for _, _, v in by_freq]):.3f}')
    print(f'     highest V overall        : {[(t, round(v, 3)) for t, _, v in by_v[:12]]}')
    print(f'     mean V of the top {len(by_freq)} by V: '
          f'{np.mean([v for _, _, v in by_v[:len(by_freq)]]):.3f}')
    print(f'     overlap between the two sets: '
          f'{len({t for t, _, _ in by_freq} & {t for t, _, _ in by_v[:len(by_freq)]})}'
          f' / {len(by_freq)}')
    print()

print('=== 3. curated vocabulary: coverage and per-class hit rate ===')
CURATED = {
    'colour': ['green', 'blue', 'orange', 'red', 'white', 'black', 'yellow', 'brown',
               'grey', 'gray', 'purple', 'pink', 'gold', 'navy', 'maroon'],
    'food': ['beer', 'soda', 'pop', 'cola', 'coke', 'popcorn', 'hot', 'dog', 'hotdog',
             'hotdogs', 'pizza', 'wings', 'fries', 'chips', 'nachos', 'peanuts',
             'burger', 'chicken', 'water', 'coffee', 'pretzel', 'donut'],
    'feeling': ['excited', 'exciting', 'bored', 'boring', 'tense', 'relaxed', 'happy',
                'anxious', 'nervous', 'neutral', 'entertained', 'engaged', 'confused',
                'interested', 'calm'],
    'song': ['waka', 'anthem', 'shakira', 'song', 'music', 'none', 'nothing'],
}
for field, vocab in CURATED.items():
    hits = np.zeros(len(df), dtype=bool)
    per_item = {}
    for token in vocab:
        indicator = np.zeros(len(df), dtype=bool)
        for index, value in enumerate(df[TEXT[field]]):
            if token in v2text.tokens(clean(value), **KW):
                indicator[index] = True
        per_item[token] = indicator
        hits |= indicator
    strong = sorted(((t, int(i.sum()), cramers_v(i)) for t, i in per_item.items()),
                    key=lambda x: -x[2])[:8]
    print(f'  {field:9s} vocab {len(vocab):2d}  rows covered {int(hits.sum()):3d}/{len(df)}'
          f' ({100 * hits.mean():.0f}%)   strongest: '
          f'{[(t, c, round(v, 2)) for t, c, v in strong]}')

print('\n=== 4. dimension budget of each representation ===')
current = sum(CAPS.values()) + 4
print(f'  v1 current                     : {current:3d} text columns '
      f'({sum(CAPS.values())} tf-idf + 4 missing flags)')
print(f'  token counts, same caps        : {current:3d}')
print(f'  discriminative tokens, 20/field: {20 * 4 + 4:3d}')
print(f'  discriminative tokens, 30/field: {30 * 4 + 4:3d}')
print(f'  curated vocabulary + length    : '
      f'{sum(len(v) for v in CURATED.values()) + 4 + 4:3d}')

print('\n=== 5. how much text is there per row? (short answers favour binary presence) ===')
for field in TEXT:
    counts = df[TEXT[field]].map(lambda s: len(v2text.tokens(clean(s), **KW)))
    uniques = df[TEXT[field]].map(lambda s: len(set(v2text.tokens(clean(s), **KW))))
    print(f'  {field:9s} tokens per row: mean {counts.mean():.1f}  median '
          f'{int(counts.median())}  90th pct {int(counts.quantile(.9))}  max {int(counts.max())}'
          f'   rows with a repeated token: {int((counts > uniques).sum())}')

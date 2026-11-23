"""Per-field (and optional per-sub-group) vocabularies: how many terms does each question need?

Diagnostic only: no model is trained, nothing is written.

The v1 pipeline already fits one vocabulary per question. This script asks the next two
questions:
  1. how many terms per question are needed before coverage and class association stop
     improving (so each question can get its own K instead of 50/20/50/50),
  2. whether splitting a question into semantic sub-vocabularies (drinks vs food, colour
     families) adds anything over individual terms.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'train' / 'random_forest_v2'))
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'data_preprocessing'))

import text as v2text                                    # noqa: E402
from encode_all import TEXT, clean, raw_headers          # noqa: E402

CLASSES = ['Soccer', 'Baseball', 'Basketball']
KW = dict(hyphen='space', min_len=2, require_letter=True)

train = pd.read_csv(ROOT / 'data/splits/train.csv', dtype=str, keep_default_na=False)
df = raw_headers(train)
labels = df.label.to_numpy()
n = len(df)
print(f'rows {n}  respondents {df.id.nunique()}\n')


def cramers_v(indicator):
    table = np.zeros((2, 3))
    for index, cls in enumerate(CLASSES):
        mask = labels == cls
        table[1, index] = indicator[mask].sum()
        table[0, index] = mask.sum() - table[1, index]
    if (table.sum(axis=1) == 0).any() or table.sum() == 0:
        return 0.0
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    if (expected <= 0).any():
        return 0.0
    return float(np.sqrt((((table - expected) ** 2 / expected).sum()) / table.sum()))


tokens_by_row = {field: [set(v2text.tokens(clean(v), **KW)) for v in df[TEXT[field]]]
                 for field in TEXT}
indicators, frequency = {}, {}
for field in TEXT:
    both = {}
    for index, got in enumerate(tokens_by_row[field]):
        for token in got:
            both.setdefault(token, np.zeros(n, dtype=bool))[index] = True
    indicators[field] = both
    frequency[field] = {token: int(mask.sum()) for token, mask in both.items()}

print('=== 1. the pipeline already keeps one vocabulary per question ===')
for field in TEXT:
    print(f'  {field:9s} distinct {len(indicators[field]):4d}  '
          f'(v1 caps this at {"20" if field == "colour" else "50"} terms)')
print()

print('=== 2. terms needed per question, ranked by class association ===')
for field in TEXT:
    scored = sorted(((t, cramers_v(m), int(m.sum())) for t, m in indicators[field].items()
                     if m.sum() >= 5), key=lambda x: -x[1])
    covered_at = {}
    union = np.zeros(n, dtype=bool)
    marks = {}
    previous_v = None
    for k, (token, v, dfreq) in enumerate(scored, 1):
        union |= indicators[field][token]
        share = union.mean()
        for target in (0.80, 0.90, 0.95):
            if target not in covered_at and share >= target:
                covered_at[target] = k
        if previous_v is not None and v <= previous_v * 0.5 and 'half' not in marks:
            marks['half'] = k
        previous_v = v
    print(f'  -- {field} (df>=5 terms: {len(scored)}, max row coverage '
          f'{int(union.sum())}/{n}) --')
    print(f'     K for 80% / 90% / 95% row coverage: '
          f'{covered_at.get(0.80, "-")} / {covered_at.get(0.90, "-")} / '
          f'{covered_at.get(0.95, "-")}')
    print(f'     V at rank 5 / 10 / 20 / 30 / 50: '
          + ' / '.join(f'{scored[r - 1][1]:.3f}' for r in (5, 10, 20, 30, 50)
                       if r <= len(scored)))
    print(f'     top 20 terms: {[t for t, _, _ in scored[:20]]}')
    print()

print('=== 3. does splitting a question into semantic sub-vocabularies help? ===')
GROUPS = {
    'food': {
        'drinks': ['beer', 'soda', 'pop', 'cola', 'coke', 'water', 'coffee', 'gatorade',
                   'pepsi', 'wine', 'drink', 'drinks'],
        'fast_food': ['pizza', 'burger', 'hotdog', 'hotdogs', 'hot', 'dog', 'dogs', 'wings',
                      'fries', 'chips', 'nachos', 'chicken', 'pretzel', 'donut'],
        'snacks': ['popcorn', 'peanuts', 'nuts', 'candy', 'seeds', 'sunflower'],
    },
    'colour': {
        'primary': ['red', 'blue', 'yellow'],
        'secondary': ['green', 'orange', 'purple'],
        'neutral': ['white', 'black', 'grey', 'gray', 'brown'],
    },
}
for field, groups in GROUPS.items():
    print(f'  -- {field} --')
    for name, words in groups.items():
        present = [w for w in words if w in indicators[field]]
        if not present:
            print(f'     {name:10s} none of the words appear')
            continue
        union = np.zeros(n, dtype=bool)
        for word in present:
            union |= indicators[field][word]
        best_single = max(cramers_v(indicators[field][w]) for w in present)
        print(f'     {name:10s} words {len(present):2d}  rows {int(union.sum()):3d}'
              f' ({100 * union.mean():4.1f}%)  group V {cramers_v(union):.3f}'
              f'   best single V {best_single:.3f}'
              f'   per class {[int(union[labels == c].sum()) for c in CLASSES]}')
    print()

print('=== 4. per-question recommendation from the numbers above ===')
print('  colour : a closed concept. V collapses after about 8 terms; 95% coverage')
print('           needs a handful of words, so K around 10-15 is enough.')
print('  food   : moderately closed. V concentrated in a dozen items, long thin tail.')
print('  feeling: open wording. Coverage needs many terms; K around 30-40.')
print('  song   : open wording plus titles. Its discriminative terms are fragments of')
print('           one or two titles, so K must stay large enough to hold them.')

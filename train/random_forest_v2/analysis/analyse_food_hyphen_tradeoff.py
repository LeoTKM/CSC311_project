"""Q8 food, part 5: does keeping '-' break the spelling variants apart, and how badly?

`hot dog`, `hotdog`, `hot-dogs` are the same food. This script measures, for each hyphen
policy, how many spellings of the same food survive as separate columns, how much
out-of-vocabulary damage each policy takes on held-out rows, and what happens to a
hyphenated compound that never appeared in training.

Diagnostic only: no model is trained, nothing is written.
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

FIELD = 'food'
MIN_DF = 3
KW = dict(min_len=2, require_letter=True)
SAMPLE = ['hot-dog', 'hot-dogs', 'ice-cream', 'chicken-wings', 'pepsi-cola',
          'deep-fried', 'home-made', 'gluten-free', 'coca-cola', 'salt-dried']

train = pd.read_csv(ROOT / 'data/splits/train.csv', dtype=str, keep_default_na=False)
frame = raw_headers(train)
both = frame
column = TEXT[FIELD]
labels = frame.label.to_numpy()
groups = frame.id.to_numpy()
answered = np.array([bool(clean(v)) for v in frame[column]])


def hybrid(text):
    """keep '-' as one token, but also emit the parts of every hyphenated token."""
    out = []
    for token in v2text.tokens(text, hyphen='keep', **KW):
        out.append(token)
        if '-' in token:
            out.extend(part for part in token.split('-')
                       if len(part) >= 2 and any(c.isalpha() for c in part))
    return out


TOKENISERS = {
    'keep': lambda s: v2text.tokens(s, hyphen='keep', **KW),
    'space': lambda s: v2text.tokens(s, hyphen='space', **KW),
    'empty': lambda s: v2text.tokens(s, hyphen='empty', **KW),
    'keep+parts': hybrid,
}
docs_by_policy = {name: [fn(v) for v in frame[column]] for name, fn in TOKENISERS.items()}
deploy_by_policy = {name: [fn(v) for v in both[column]] for name, fn in TOKENISERS.items()}


def vocab(docs, min_df):
    counts = {}
    for doc in docs:
        for token in set(doc):
            counts[token] = counts.get(token, 0) + 1
    return {t for t, c in counts.items() if c >= min_df}


print('=' * 96)
print('1. THE SAME FOOD UNDER DIFFERENT SPELLINGS (training text, min_df=3)')
print('=' * 96)
for name, docs in deploy_by_policy.items():
    kept = vocab(docs, MIN_DF)
    variants = sorted(t for t in kept if 'hot' in t or 'dog' in t)
    counts = {}
    for doc in docs:
        for token in set(doc):
            counts[token] = counts.get(token, 0) + 1
    print(f'{name:12s} vocabulary {len(kept):3d}   hot/dog columns: '
          + ', '.join(f'{t}({counts[t]})' for t in variants))

print()
print('=' * 96)
print('2. HELD-OUT DAMAGE PER POLICY (grouped 5-fold over the 606 training rows)')
print('=' * 96)
print(f'{"policy":14s} {"vocab":>6s} {"token OOV":>10s} {"rows w/ OOV":>12s} '
      f'{"rows all OOV":>13s}')
for name, docs in docs_by_policy.items():
    rows = []
    for fit_index, held_index in GroupKFold(n_splits=5).split(frame, labels, groups):
        train_vocab = vocab([docs[i] for i in fit_index], MIN_DF)
        held = [i for i in held_index if answered[i]]
        sets = [set(docs[i]) for i in held]
        flat = [t for s in sets for t in s]
        rows.append((
            sum(1 for t in flat if t not in train_vocab) / max(1, len(flat)),
            np.mean([bool(s - train_vocab) for s in sets]),
            np.mean([bool(s) and not (s & train_vocab) for s in sets]),
        ))
    mean = np.mean(rows, axis=0)
    print(f'{name:14s} {len(vocab(docs, MIN_DF)):6d} {mean[0]:10.1%} {mean[1]:12.1%} '
          f'{mean[2]:13.1%}')

print()
print('=' * 96)
print('3. A HYPHENATED COMPOUND THAT IS NOT IN THE TRAINING TEXT')
print('=' * 96)
print('"would this answer reach any known column?"  (training vocabulary, min_df=3)')
header = f'{"answer":18s}' + ''.join(f'{n:>26s}' for n in TOKENISERS)
print(header)
for answer in SAMPLE:
    cells = []
    for name, fn in TOKENISERS.items():
        kept = vocab(deploy_by_policy[name], MIN_DF)
        tokens = fn(answer)
        hit = [t for t in tokens if t in kept]
        cells.append(f'{str(tokens)}{"->" + str(hit) if hit else "-> LOST"}')
    print(f'{answer:18s}' + ''.join(f'{c:>26s}' for c in cells))

"""Q8 food, part 4: how much of the out-of-vocabulary problem is just inflection?

The OOV list in part 3 mixed two very different things: rare function words (harmless,
they carry no signal) and inflected or misspelled variants of common food words
(`donuts`/`donut`, `burgers`/`burger`, `bier`/`beer`). Only the second kind is a real
loss. This script measures how many of them a trivial plural normaliser would recover.

nltk is not installed, so the normaliser below is a deliberately crude plural stripper,
not a real stemmer. It exists only to size the effect. Diagnostic only.
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
from encode_all import TEXT, clean, raw_headers      # noqa: E402

FIELD = 'food'
KW = dict(hyphen='keep', min_len=2, require_letter=True)


def singular(token):
    """Crude English plural fold; keeps tokens that are already short."""
    if len(token) < 4 or not token.isalpha():
        return token
    if token.endswith('ies') and len(token) > 4:
        return token[:-3] + 'y'
    for suffix in ('ches', 'shes', 'sses', 'xes', 'zes'):
        if token.endswith(suffix):
            return token[:-2]
    if token.endswith('s') and not token.endswith(('ss', 'us', 'is', 'as')):
        return token[:-1]
    return token


train = pd.read_csv(ROOT / 'data/splits/train.csv', dtype=str, keep_default_na=False)
frame = raw_headers(train)
column = TEXT[FIELD]
labels = frame.label.to_numpy()
groups = frame.id.to_numpy()
answered = np.array([bool(clean(value)) for value in frame[column]])
base = [v2text.tokens(value, **KW) for value in frame[column]]
folded = [[singular(t) for t in doc] for doc in base]
print(f'{len(frame)} training rows, {answered.sum()} answered Q8\n')

print(f'{"":34s} {"token OOV":>10s} {"rows w/ OOV":>12s} {"rows all OOV":>13s}')
for name, docs in (('as tokenised', base), ('after plural folding', folded)):
    rows = []
    for fit_index, held_index in GroupKFold(n_splits=5).split(frame, labels, groups):
        counts = {}
        for i in fit_index:
            for token in set(docs[i]):
                counts[token] = counts.get(token, 0) + 1
        vocabulary = {t for t, c in counts.items() if c >= 2}
        held = [i for i in held_index if answered[i]]
        sets = [set(docs[i]) for i in held]
        flat = [t for s in sets for t in s]
        rows.append((
            sum(1 for t in flat if t not in vocabulary) / max(1, len(flat)),
            np.mean([bool(s - vocabulary) for s in sets]),
            np.mean([bool(s) and not (s & vocabulary) for s in sets]),
        ))
    mean = np.mean(rows, axis=0)
    print(f'{name:34s} {mean[0]:10.1%} {mean[1]:12.1%} {mean[2]:13.1%}')

print()
print('merged by the plural fold (top 20 by how often they occur):')
merged = {}
for doc_base, doc_fold in zip(base, folded):
    for original, result in zip(doc_base, doc_fold):
        if original != result:
            merged.setdefault(result, set()).add(original)
counts = {}
for doc_base in base:
    for token in set(doc_base):
        counts[token] = counts.get(token, 0) + 1
pairs = sorted(merged.items(),
               key=lambda kv: -sum(counts.get(o, 0) for o in kv[1]))
for result, originals in pairs[:20]:
    print(f'  {result:16s} <- ' + ', '.join(
        f'{o}({counts.get(o, 0)})' for o in sorted(originals, key=lambda o: -counts.get(o, 0))))
print(f'\ntotal distinct tokens {len(counts)} -> {len({singular(t) for t in counts})}')

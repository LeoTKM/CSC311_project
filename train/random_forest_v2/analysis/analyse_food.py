"""Q8 food: can the hyphen be kept, and which tokens are 'meaningless'?

Diagnostic only: nothing is trained, nothing is written. Reads the training split only.

Three questions:
  1. does '-' ever appear outside a compound noun/adjective in the food answers?
  2. what does the food vocabulary look like by document frequency, and which of those
     tokens are grammatical connectives rather than food/drink content?
  3. does dropping the connectives cost class association, and does a multi-item count
     feature carry the same information?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'train' / 'random_forest_v2'))
sys.path.insert(0, str(ROOT / 'data_preprocessing'))

import text as v2text                                # noqa: E402
from encode_all import TEXT, clean, raw_headers      # noqa: E402

CLASSES = ['Soccer', 'Baseball', 'Basketball']
FIELD = 'food'
KW = dict(min_len=2, require_letter=True)

df = raw_headers(pd.read_csv(ROOT / 'data/splits/train.csv', dtype=str,
                             keep_default_na=False))
column = TEXT[FIELD]
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
    return float(np.sqrt(chi2 / (n * min(table.shape[0] - 1, table.shape[1] - 1))))


def frequencies(hyphen):
    docs = [v2text.tokens(value, hyphen=hyphen, **KW) for value in df[column]]
    counter = {}
    presence = []
    for doc in docs:
        seen = set(doc)
        presence.append(seen)
        for token in seen:
            counter[token] = counter.get(token, 0) + 1
    return docs, counter, presence


# ---------------------------------------------------------------- 1. hyphen audit
print('=' * 96)
print('1. HYPHEN AUDIT (hyphen policy = keep, so internal hyphens stay in the token)')
print('=' * 96)
docs_keep, dfreq, presence_keep = frequencies('keep')
hyphenated = {t: c for t, c in dfreq.items() if '-' in t or t.endswith('-') or t.startswith('-')}
print(f'distinct tokens {len(dfreq)}   of which carry a hyphen: {len(hyphenated)}')
raw_hits = [v for v in df[column] if '-' in str(v)]
print(f'raw answers containing "-": {len(raw_hits)} of {n} rows')
print('all hyphenated tokens, by document frequency:')
for token, count in sorted(hyphenated.items(), key=lambda kv: -kv[1]):
    where = ('leading' if token.startswith('-') else '') + \
            ('trailing' if token.endswith('-') else '') or 'internal'
    found = np.array([token in s for s in presence_keep])
    print(f'  {count:4d}  {show(token):28s} {where:10s} V={cramers_v(found):.3f}')
print('\nanswers that contain a hyphen (first 14):')
for value in raw_hits[:14]:
    print(f'  {show(value)}')

# ---------------------------------------------------------------- 2. vocabulary
print()
print('=' * 96)
print('2. FOOD VOCABULARY by document frequency (hyphen policy = keep)')
print('=' * 96)
indicator = {t: np.array([t in s for s in presence_keep]) for t in dfreq}
rows = sorted(dfreq.items(), key=lambda kv: -kv[1])
print(f'{"token":26s} {"df":>4s} {"cover":>7s} {"V":>6s}   per-class rate  S/B/Bb')
for token, count in rows[:70]:
    mask = indicator[token]
    rates = '/'.join(f'{mask[labels == c].mean():.2f}' for c in CLASSES)
    print(f'{show(token):26s} {count:4d} {count / n:6.1%} {cramers_v(mask):6.3f}   {rates}')

# ---------------------------------------------------------------- 3. connectives
print()
print('=' * 96)
print('3. CANDIDATE CONNECTIVES / GENERIC FILLER (not a food or a drink)')
print('=' * 96)
CONNECTIVE = ['and', 'with', 'a', 'the', 'or', 'of', 'to', 'at', 'on', 'in', 'for',
              'my', 'i', 'some', 'like', 'just', 'is', 'it', 'this', 'that', 'be',
              'usually', 'mainly', 'mostly', 'traditional', 'traditionally',
              'typical', 'typically', 'local', 'national', 'common', 'classic',
              'food', 'foods', 'drink', 'drinks', 'snack', 'snacks', 'meal', 'meals',
              'game', 'games', 'watch', 'watching', 'sport', 'sports']
print(f'{"token":18s} {"df":>4s} {"V":>6s}  per-class rate S/B/Bb')
for token in CONNECTIVE:
    if token not in dfreq:
        print(f'{token:18s} {"-":>4s} {"-":>6s}  absent')
        continue
    mask = indicator[token]
    rates = '/'.join(f'{mask[labels == c].mean():.2f}' for c in CLASSES)
    print(f'{token:18s} {dfreq[token]:4d} {cramers_v(mask):6.3f}  {rates}')

# ---------------------------------------------------------------- 4. effect of dropping
print()
print('=' * 96)
print('4. WHAT DROPPING THE CONNECTIVES DOES')
print('=' * 96)
STOP = {'and', 'with', 'a', 'the', 'or', 'of', 'to', 'at', 'on', 'in', 'for', 'my',
        'i', 'some', 'like', 'just', 'is', 'it', 'this', 'that', 'be', 'it\'s'}
dropped = [t for t in STOP if t in dfreq]
print(f'stoplist size {len(STOP)}, present in the food vocabulary {len(dropped)}: '
      f'{show(", ".join(sorted(dropped)))}')
coverage_before = np.mean([bool(s) for s in presence_keep])
presence_after = [s - STOP for s in presence_keep]
coverage_after = np.mean([bool(s) for s in presence_after])
print(f'rows with at least one token: {coverage_before:.1%} -> {coverage_after:.1%}')
kept = sorted(((cramers_v(indicator[t]), t, c) for t, c in dfreq.items() if t not in STOP),
              reverse=True)
for k in (10, 20, 30, 50):
    before = sorted(((cramers_v(indicator[t]), t) for t in dfreq), reverse=True)[:k]
    print(f'  top-{k:<3d} mean V  with connectives {np.mean([v for v, _ in before]):.4f}'
          f'   without {np.mean([v for v, _, _ in kept[:k]]):.4f}')
print('\ntop 20 by V after dropping the connectives:')
for value, token, count in kept[:20]:
    print(f'  {show(token):24s} df={count:4d} V={value:.3f}')
print('\ntokens that leave the top 30 when the connectives are dropped:')
before30 = {t for _, t in sorted(((cramers_v(indicator[t]), t) for t in dfreq), reverse=True)[:30]}
after30 = {t for _, t, _ in kept[:30]}
print('  ' + show(', '.join(sorted(before30 - after30))))

# ---------------------------------------------------------------- 5. item count
print()
print('=' * 96)
print('5. MULTI-ITEM ANSWER STRUCTURE (what the connectives were hinting at)')
print('=' * 96)
counts = np.array([len(s - STOP) for s in presence_after])
print(f'{"kept tokens per row":22s} {"rows":>5s}  per-class rate S/B/Bb')
for label, mask in (('0', counts == 0), ('1', counts == 1), ('2', counts == 2),
                    ('3+', counts >= 3)):
    rates = '/'.join(f'{mask[labels == c].mean():.2f}' for c in CLASSES)
    print(f'{label:22s} {mask.sum():5d}  {rates}')
print(f'\nV of "at least 2 items": {cramers_v(counts >= 2):.3f}')
print(f'V of "at least 3 items": {cramers_v(counts >= 3):.3f}')
print(f'V of the kept-token count itself (3-bin split): '
      f'{cramers_v(counts >= 1):.3f} / {cramers_v(counts >= 2):.3f} / {cramers_v(counts >= 3):.3f}')

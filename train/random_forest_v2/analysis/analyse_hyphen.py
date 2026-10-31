"""Does merging the hyphen pay off? Compare the three hyphen policies on the v1 rows.

Diagnostic only: no model is trained and nothing is written.

For every hyphen-containing answer the script shows what each policy produces and, more
importantly, whether the affected word forms end up sharing a vocabulary entry with forms
that already exist (consolidation) or become new rare entries (fragmentation). The last
section is the decisive one: which tokens make the top-50 document-frequency cut that the
TF-IDF cap uses.
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'train' / 'random_forest_v2'))
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'data_preprocessing'))

import text as v2text                                            # noqa: E402
from encode_all import TEXT, clean, raw_headers                  # noqa: E402

POLICIES = v2text.POLICIES
CAPS = {'food': 50, 'colour': 20, 'song': 50, 'feeling': 50}

train = pd.read_csv(ROOT / 'data/splits/train.csv', dtype=str, keep_default_na=False)
df = raw_headers(train)
print(f'rows {len(df)}  respondents {df.id.nunique()}\n')

print('answers containing a hyphen, per field:')
for field in TEXT:
    print(f'  {field:9s} {int(df[TEXT[field]].astype(str).str.contains("-").sum()):3d}')

print('\n=== what each policy produces for real answers ===')
for value in ['coca-cola/pepsi', 'hot-dogs and fries', 'warm-hearted, high-energy',
              'blue-red', 'Waka Waka (This Time for Africa) - Shakira',
              'fast-paced basketball', 'on-edge, tense']:
    print(f'  raw  : {value!r}')
    for policy in POLICIES:
        print(f'  {policy:6s}: {v2text.tokens(value, hyphen=policy)}')

print('\n=== vocabulary per field and policy (no noise removal yet) ===')
for policy in POLICIES:
    print(f'-- hyphen={policy} --')
    for field in TEXT:
        counts, docs = Counter(), Counter()
        for value in df[TEXT[field]]:
            got = v2text.tokens(clean(value), hyphen=policy)
            counts.update(got)
            docs.update(set(got))
        repeated = sum(1 for c in docs.values() if c >= 2)
        print(f'   {field:9s} tokens {sum(counts.values()):5d}  distinct {len(counts):4d}  '
              f'df>=2 {repeated:4d}  singletons {len(counts) - repeated:4d}')
    print()

print('=== consolidation test: where do the hyphen forms end up? ===')
for field in TEXT:
    per_policy = {p: Counter() for p in POLICIES}
    for policy in POLICIES:
        for value in df[TEXT[field]]:
            per_policy[policy].update(v2text.tokens(clean(value), hyphen=policy))
    forms = [t for t in per_policy['keep'] if '-' in t]
    if not forms:
        print(f'  {field}: no hyphenated tokens')
        continue
    print(f'  -- {field} --')
    for form in sorted(forms, key=lambda t: -per_policy['keep'][t]):
        kept = per_policy['keep'][form]
        joined = per_policy['empty'].get(form.replace('-', ''), 0)
        parts = {p: per_policy['space'].get(p, 0) for p in form.split('-')}
        exists = ''
        if joined:
            exists = 'MERGES INTO AN EXISTING TOKEN' if joined > kept else 'joins a rarer token'
        else:
            exists = 'creates a new token'
        print(f'     {form:18s} keep={kept:2d} | empty -> {form.replace("-", ""):16s} '
              f'df={joined:3d} ({exists}) | space -> {parts}')

print('\n=== decisive check: the top-50 document-frequency vocabulary per policy ===')
for field in TEXT:
    vocab = {}
    for policy in POLICIES:
        docs = Counter()
        for value in df[TEXT[field]]:
            docs.update(set(v2text.tokens(clean(value), hyphen=policy)))
        ranked = [t for t, _ in sorted(docs.items(), key=lambda kv: (-kv[1], kv[0]))]
        vocab[policy] = ranked[:CAPS[field]]
    keep, space, empty = (set(vocab[p]) for p in POLICIES)
    print(f'  -- {field} (cap {CAPS[field]}) --')
    print(f'     keep  vs empty differ by: keep-only {sorted(keep - empty)} '
          f'empty-only {sorted(empty - keep)}')
    print(f'     space vs empty differ by: space-only {sorted(space - empty)} '
          f'empty-only {sorted(empty - space)}')
    print(f'     keep  vs space differ by: keep-only {sorted(keep - space)} '
          f'space-only {sorted(space - keep)}')

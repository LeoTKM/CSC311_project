"""Q11 feeling: can a rule-based suffix normaliser replace the hand-written merge map?

The proposal: fold every derivation of one emotion onto a single standard form, so that
`excitement`, `exciting` and `excited` become one column and their distribution over the
three classes is sharper than three thin columns.

Two ways to do it:
  map     a hand-written dictionary, one entry per word (51 words, 14 emotions)
  rule    a suffix chain plus a silent-e restoration, no dictionary, plus a handful of
          irregular entries for the stems that the rule cannot line up

Both are measured on the feeling block alone, grouped 5-fold over the 606 training rows,
seeds 311/312/313. Nothing is written.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'train' / 'random_forest_v2'))
sys.path.insert(0, str(ROOT / 'data_preprocessing'))

import text as v2text                                # noqa: E402
from encode_all import TEXT, clean, raw_headers       # noqa: E402

CLASSES = ['Soccer', 'Baseball', 'Basketball']
KW = dict(hyphen='space', min_len=2, require_letter=True)
MIN_DF = 3
SEEDS = (311, 312, 313)
VOWEL = set('aeiou')
MIN_STEM = 3

# tried longest first; 'ation' before 'ion' so relaxation lands on relax, not relaxat
SUFFIXES = ('ations', 'ation', 'ments', 'ment', 'ness', 'dom', 'ions', 'ion',
            'ances', 'ance', 'ences', 'ence', 'ities', 'ity', 'ings', 'ing',
            'edly', 'ed', 'ly', 'es')

# stems the suffix chain cannot line up, checked before any stripping
IRREGULAR = {
    'energetic': 'energetic', 'energized': 'energetic', 'energize': 'energetic',
    'energy': 'energetic', 'energetized': 'energetic', 'energies': 'energetic',
    'anxious': 'anxious', 'anxiety': 'anxious',
    'happy': 'happy', 'happiness': 'happy',
    'sad': 'sad', 'sadness': 'sad',
    'tense': 'tense', 'tension': 'tense',
    'relaxation': 'relaxe', 'relaxing': 'relaxe', 'relax': 'relaxe',
}

MAP = {
    'excited': {'excited', 'excitement', 'exciting', 'excite', 'excitedly', 'exhilarated'},
    'bored': {'bored', 'boring', 'boredom', 'bore', 'boredness'},
    'relaxed': {'relaxed', 'relaxing', 'relaxation', 'relax'},
    'confused': {'confused', 'confusing', 'confusion', 'confusions', 'confuse'},
    'tense': {'tense', 'tension'},
    'energetic': {'energetic', 'energized', 'energize', 'energy', 'energetized', 'energies'},
    'interested': {'interested', 'interesting', 'interest'},
    'entertained': {'entertained', 'entertaining', 'entertainment'},
    'enjoyable': {'enjoy', 'enjoyable', 'enjoyment', 'enjoying'},
    'nervous': {'nervous', 'nervously', 'nervousness'},
    'anxious': {'anxious', 'anxiety'},
    'frustrated': {'frustrated', 'frustrating', 'frustration'},
    'intense': {'intense', 'intensity', 'intensely'},
    'sad': {'sad', 'sadness'},
}
MAP_LOOKUP = {word: canonical for canonical, group in MAP.items() for word in group}


def standard_form(token, irregular=True):
    """Fold a derivation onto one standard form: suffix chain, then restore a silent e."""
    if irregular and token in IRREGULAR:
        return IRREGULAR[token]
    stem = token
    for _ in range(3):
        for suffix in SUFFIXES:
            if stem.endswith(suffix) and len(stem) - len(suffix) >= MIN_STEM:
                stem = stem[:-len(suffix)]
                break
        else:
            break
    if (len(stem) >= 4 and stem[-1] not in VOWEL and stem[-2] in VOWEL
            and stem[-3] not in VOWEL and stem[-1] != 'y'):
        stem = stem + 'e'
    return stem


train = pd.read_csv(ROOT / 'data/splits/train.csv', dtype=str, keep_default_na=False)
frame = raw_headers(train)
column = TEXT['feeling']
labels = frame.label.to_numpy()
groups = frame.id.to_numpy()
n = len(frame)
answered = np.array([bool(clean(v)) for v in frame[column]])
plain_docs = [v2text.tokens(v, **KW) for v in frame[column]]

variants = {
    'plain': plain_docs,
    'map': [[MAP_LOOKUP.get(t, t) for t in doc] for doc in plain_docs],
    'rule': [[standard_form(t, irregular=False) for t in doc] for doc in plain_docs],
    'rule+irregular': [[standard_form(t) for t in doc] for doc in plain_docs],
}


def show(s):
    return str(s).encode('ascii', 'backslashreplace').decode('ascii')


counts = {}
for s in (set(d) for d in plain_docs):
    for token in s:
        counts[token] = counts.get(token, 0) + 1
print(f'{n} rows, answered {answered.sum()}\n')

print('=' * 96)
print('WHAT THE RULE DOES TO THE FREQUENT WORDS (df >= 3)')
print('=' * 96)
groups_by_form = {}
for token, count in sorted(counts.items(), key=lambda kv: -kv[1]):
    if count < MIN_DF:
        continue
    form = standard_form(token)
    groups_by_form.setdefault(form, []).append((token, count))
for form, members in sorted(groups_by_form.items(), key=lambda kv: -sum(c for _, c in kv[1])):
    if len(members) > 1:
        print(f'  {form:14s} <- ' + ', '.join(f'{show(t)}({c})' for t, c in members))
print('\ntokens above min_df that stay alone, first 30:')
alone = [f'{show(t)}({c})' for form, members in groups_by_form.items()
         if len(members) == 1 for t, c in members]
print('  ' + ', '.join(alone[:30]))
print(f'\ndistinct tokens above min_df: '
      f'{sum(len(m) for m in groups_by_form.values())} -> '
      f'{len(groups_by_form)} forms')


def chi2_scores(matrix, target):
    scores = np.zeros(matrix.shape[1])
    for j in range(matrix.shape[1]):
        present = matrix[:, j] == 1
        table = np.zeros((2, 3))
        for index, cls in enumerate(CLASSES):
            mask = target == cls
            table[1, index] = (present & mask).sum()
            table[0, index] = (~present & mask).sum()
        expected = table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True) / len(target)
        if (expected == 0).any():
            continue
        scores[j] = ((table - expected) ** 2 / expected).sum()
    return scores


def evaluate(docs, k):
    records = []
    for seed in SEEDS:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fit_index, held_index in splitter.split(frame, labels, groups):
            presence = [set(docs[i]) for i in range(n)]
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
            order = np.argsort(-chi2_scores(matrix, labels[fit_index]))
            chosen = [pool[j] for j in order[:min(k, len(pool))]]
            position = {t: j for j, t in enumerate(chosen)}

            def project(indices):
                out = np.zeros((len(indices), len(chosen)), dtype=np.int8)
                for row, i in enumerate(indices):
                    for token in presence[i]:
                        if token in position:
                            out[row, position[token]] = 1
                return out

            X_fit, X_held = project(fit_index), project(held_index)
            y_fit = pd.Series(labels[fit_index]).map(
                {c: i for i, c in enumerate(CLASSES)}).to_numpy()
            y_held = pd.Series(labels[held_index]).map(
                {c: i for i, c in enumerate(CLASSES)}).to_numpy()
            model = RandomForestClassifier(n_estimators=300, max_depth=6,
                                           min_samples_leaf=3, max_features='log2',
                                           random_state=seed, n_jobs=1)
            model.fit(X_fit, y_fit)
            predicted = model.predict(X_held)
            records.append((accuracy_score(y_held, predicted),
                            f1_score(y_held, predicted, average='macro'),
                            np.mean(X_held.sum(axis=1) == 0)))
    return pd.DataFrame(records, columns=['acc', 'f1', 'empty']).mean()


print()
print('=' * 96)
print('FEELING BLOCK ALONE, min_df=3')
print('=' * 96)
print(f'{"variant":16s} {"K":>4s} {"forest acc":>11s} {"forest F1":>10s} {"no term":>8s}')
for name, docs in variants.items():
    for k in (40, 60, 120):
        mean = evaluate(docs, k)
        print(f'{name:16s} {k:4d} {mean.acc:11.4f} {mean.f1:10.4f} {mean.empty:7.1%}')
    print()

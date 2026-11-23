"""Two checks on the harness behind `compare_text.py`, both on the training split only.

1. `tfidf` fidelity: sklearn's ``TfidfVectorizer`` with v1's exact arguments against the
   hand-written vocabulary plus weighting in `compare_text.py`. If the two agree column by
   column, the v1 side of the comparison is a faithful reproduction and the only remaining
   difference is column order.
2. Term coverage: for each open field, how many of the 606 rows end up with no active
   column at all, at v1's budget (50 terms, min_df 2) and at v2's (30 terms, min_df 3).
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V2 = HERE.parents[1] / 'random_forest_v2'
for path in (str(V2), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import encode as survey                      # noqa: E402
from compare_text import (REGEX_TOKEN, regex_docs, select_terms, weigh,  # noqa: E402
                          COLOUR_NAMES)

TRAIN_CSV = ROOT / 'data' / 'splits' / 'train.csv'
FIELDS = ('food', 'colour', 'song', 'feeling')
V1_CAPS = {'food': 50, 'colour': 20, 'song': 50, 'feeling': 50}


def sklearn_block(texts, cap, min_df=2):
    vectorizer = TfidfVectorizer(
        lowercase=True, token_pattern=r'(?u)\b[a-zA-Z]{2,}\b', ngram_range=(1, 1),
        max_features=cap, min_df=min_df, sublinear_tf=True, smooth_idf=True,
        norm='l2', stop_words=None)
    matrix = vectorizer.fit_transform(texts)
    return np.asarray(matrix.todense()), list(vectorizer.get_feature_names_out())


def check_fidelity(train):
    print('1. tf-idf fidelity: sklearn against the hand-written weighting')
    worst = 0.0
    for field in FIELDS:
        texts = train[survey.TEXT[field]].map(survey.clean).tolist()
        reference, columns = sklearn_block(texts, V1_CAPS[field])
        docs = regex_docs(texts)
        vocabulary = select_terms(docs, train['label'].to_numpy(), 'tf',
                                  V1_CAPS[field], 2)
        mine = weigh(docs, vocabulary, 'tfidf')
        same_names = set(columns) == set(vocabulary)
        shared = sorted(set(columns) & set(vocabulary))
        left = [columns.index(term) for term in shared]
        right = [vocabulary.index(term) for term in shared]
        gap = float(np.abs(reference[:, left] - mine[:, right]).max()) if shared else 0.0
        worst = max(worst, gap)
        print(f'   {field:8s} sklearn {len(columns):3d} cols  mine {len(vocabulary):3d} cols  '
              f'names {"equal" if same_names else "DIFFER"}  '
              f'shared {len(shared):3d}  max |diff| {gap:.2e}')
    print(f'   worst disagreement {worst:.2e}\n')
    return worst


def coverage(train):
    print('2. rows with no active column, per field, at each budget')
    labels = train['label'].to_numpy()
    rows = []
    for field in FIELDS:
        texts = train[survey.TEXT[field]].map(survey.clean).tolist()
        docs = regex_docs(texts)
        empty_text = sum(1 for text in texts if not text)
        for name, cap, min_df, rule in (('v1  50/min_df2/tf', 50, 2, 'tf'),
                                        ('v2  30/min_df3/chi2', 30, 3, 'chi2'),
                                        ('v3  50/min_df2/chi2', 50, 2, 'chi2')):
            if field == 'colour':
                active = np.array([bool(set(doc) & survey.COLOUR_WORDS) for doc in docs])
                columns = len(COLOUR_NAMES)
                label = f'{name} (lexicon, {columns} cols)'
            else:
                vocabulary = select_terms(docs, labels, rule,
                                          cap if field != 'colour' else 20, min_df)
                kept = set(vocabulary)
                active = np.array([bool(set(doc) & kept) for doc in docs])
                label = f'{name} ({len(vocabulary)} cols)'
            missing = int((~active).sum())
            rows.append({'field': field, 'budget': label, 'blank': empty_text,
                         'no_column': missing,
                         'share': missing / len(texts)})
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False,
                          formatters={'share': '{:.3f}'.format}))
    return frame


def main():
    train = pd.read_csv(TRAIN_CSV, dtype=str, keep_default_na=False)
    worst = check_fidelity(train)
    frame = coverage(train)
    out = HERE / 'out'
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / 'coverage.csv', index=False)
    print(f'\nwrote {out}   (fidelity worst gap {worst:.2e})')


if __name__ == '__main__':
    main()

"""v3 feature matrix: 32 structured columns and four short-answer blocks.

Column layout, in order:

    26  structured   ease, strategy, injury (Q4 is dropped), the 21 multi-hot options,
                     payment_amount (mean filled, log1p), payment_missing
     6  flags        q1_missing, q2_missing, q3_missing, q5_missing, q6_missing, q7_missing
    53  food         50 most frequent terms, then food_non_answer, food_no_terms,
                     food_item_count
    20  colour       the 17 fixed lexicon colours, then the three colour_ columns
    53  song         50 most frequent terms, then the three song_ columns
    15  feeling      12 most frequent terms, then the three feeling_ columns
    --
   173  total

Three decisions differ from the two earlier models, and each was measured on the training
split before it was written down here:

1. Term selection ranks by corpus frequency, not by chi-square. Frequency ranking is not
   more informative term by term, but it covers far more respondents: at a 50-term budget
   40 of the 606 rows have no active food column and 162 have no active song column, while
   chi-square selection at the same budget leaves 93 and 211. A random forest cannot use a
   row whose block is entirely zero. Selection is unsupervised, so it also carries no
   fold-to-fold variance of its own.
2. Values are 0/1 presence. The answer-breadth information that a TF-IDF row norm folds
   into every column is instead carried by one explicit `item_count` column per field.
3. The colour answer uses a fixed 17-concept lexicon with no fitted state, so a fold
   cannot change the colour columns. The `grey/gray` and `black/balck` variants merge.

The three `*_` columns per text field are `non_answer` (blank or declined), `no_terms`
(answered, but not one word is in the kept vocabulary) and `item_count`. The middle one is
the signal neither earlier model could express: an all-zero block and an
answered-but-unrecognised block looked identical to the forest.

Every fitted value here - the four term vocabularies and the structured imputers - comes
from the rows passed to `fit`. Only `data/splits/train.csv` is ever passed in.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import encode as survey   # noqa: E402
import text as v3text     # noqa: E402

CLASSES = ['Soccer', 'Baseball', 'Basketball']
TEXT_FIELDS = ('food', 'colour', 'song', 'feeling')

# Q4 is the agreement item 'Individual brilliance can outweigh teamwork in this sport'.
# Both its value column and its missing flag are dropped, so the structured block is 32
# columns instead of 34.
Q4_VALUE = 'individual'
Q4_FLAG = 'q4_missing'
DROP_COLUMNS = (Q4_VALUE, Q4_FLAG)

# Terms kept per open field. colour has no K: its columns are the lexicon.
DEFAULT_K = {'food': 50, 'song': 50, 'feeling': 12}
MIN_DF = 2

# The four agreement questions, in survey order, with the short names used below.
ORDINAL_NAMES = {survey.EASE_COLUMN: 'ease',
                 survey.AGREEMENT_COLUMNS[0]: 'strategy',
                 survey.AGREEMENT_COLUMNS[1]: 'injury',
                 survey.AGREEMENT_COLUMNS[2]: 'individual'}
ORDINAL_KEPT = [name for name in survey.ORDINAL_COLUMNS
                if ORDINAL_NAMES[name] not in DROP_COLUMNS]
ORDINAL_SHORT = [ORDINAL_NAMES[name] for name in ORDINAL_KEPT]
FLAG_QUESTIONS = [(name, column) for name, column
                  in zip(survey.FLAG_COLUMNS,
                         [survey.EASE_COLUMN, *survey.AGREEMENT_COLUMNS, *survey.SCHEMA])
                  if name not in DROP_COLUMNS]


def tokenised(raw):
    """Tokenise all four answers. No standard form, no per-field variation."""
    return {field: v3text.tokenize([survey.clean(value)
                                    for value in raw[survey.TEXT[field]]])
            for field in TEXT_FIELDS}


def select_terms(docs, k, min_df):
    """The `k` most frequent terms of one field, ties broken deterministically.

    Ranking is by total occurrences, then by the number of documents containing the term,
    then alphabetically. The labels are deliberately not an input.
    """
    document, frequency = Counter(), Counter()
    for doc in docs:
        document.update(set(doc))
        frequency.update(doc)
    pool = [term for term, count in document.items() if count >= min_df]
    pool.sort(key=lambda term: (-frequency[term], -document[term], term))
    return pool[:k]


def presence(docs, vocabulary):
    """0/1 presence of each kept term, one row per answer."""
    if not vocabulary:
        return np.zeros((len(docs), 0))
    position = {term: index for index, term in enumerate(vocabulary)}
    block = np.zeros((len(docs), len(vocabulary)))
    for row, doc in enumerate(docs):
        for term in doc:
            index = position.get(term)
            if index is not None:
                block[row, index] = 1.0
    return block


class TextPreprocessor:
    """Fit on the training rows, then transform anything with the same columns."""

    def __init__(self, text_fields=TEXT_FIELDS, keep_structured=True,
                 k_by_field=None, min_df=MIN_DF):
        unknown = set(text_fields) - set(TEXT_FIELDS)
        if unknown:
            raise ValueError(f'unknown text field(s): {sorted(unknown)}')
        unknown = set(k_by_field or {}) - set(DEFAULT_K)
        if unknown:
            raise ValueError(f'K given for a field that has no K: {sorted(unknown)}')
        self.text_fields = tuple(field for field in TEXT_FIELDS if field in set(text_fields))
        self.keep_structured = bool(keep_structured)
        self.k_by_field = {**DEFAULT_K, **(k_by_field or {})}
        self.min_df = int(min_df)

    # ------------------------------------------------------------------ text blocks
    def _open_block(self, docs, field, fitting):
        """The colour lexicon, or the most frequent terms of one open field."""
        if field == 'colour':
            rows = []
            for doc in docs:
                words = set(doc)
                rows.append([float(bool(words & set(survey.COLOUR_LEXICON[name])))
                             for name in survey.COLOUR_LEXICON])
            columns = [f'colour_{name}' for name in survey.COLOUR_LEXICON]
            hit = [bool(set(doc) & survey.COLOUR_WORDS) for doc in docs]
            return np.array(rows, dtype=float), columns, hit
        if fitting:
            self.vocabulary_[field] = select_terms(docs, self.k_by_field[field],
                                                   self.min_df)
        vocabulary = self.vocabulary_[field]
        columns = [f'{field}_{term}' for term in vocabulary]
        hit = [bool(set(doc) & set(vocabulary)) for doc in docs]
        return presence(docs, vocabulary), columns, hit

    def _aux_block(self, texts, docs, hit, field):
        """The three columns that describe the answer rather than its words."""
        non_answer = np.array([
            1.0 if (not text) or text in survey.NON_ANSWER else 0.0 for text in texts
        ])
        no_terms = np.array([
            0.0 if declined else (0.0 if matched else 1.0)
            for declined, matched in zip(non_answer, hit)
        ])
        count = np.array([float(len(doc)) for doc in docs])
        return (np.column_stack([non_answer, no_terms, count]),
                [f'{field}_non_answer', f'{field}_no_terms', f'{field}_item_count'])

    # ------------------------------------------------------------------ structured
    def _structured_block(self, raw, fitting):
        encoded = survey.encode_structured(raw)
        block = pd.DataFrame(
            encoded[ORDINAL_KEPT].to_numpy(dtype=float),
            index=raw.index, columns=ORDINAL_SHORT)
        for name in survey.MULTI_COLUMNS:
            block[name] = encoded[name].to_numpy(dtype=float)
        parsed = [survey.parse_payment(value) for value in raw[survey.PAYMENT]]
        amounts = pd.Series([amount for amount, _ in parsed], index=raw.index, dtype=float)
        if fitting:
            valid = amounts.dropna()
            if valid.empty:
                raise ValueError('No valid training payment amounts to impute from.')
            self.payment_fill_ = float(valid.mean())
        block['payment_amount'] = amounts.fillna(self.payment_fill_)
        block['payment_missing'] = amounts.isna().astype(float)
        flags = pd.DataFrame({
            name: raw[column].map(survey.clean).eq('').astype(float)
            for name, column in FLAG_QUESTIONS
        }, index=raw.index)
        return pd.concat([block, flags], axis=1)

    # ------------------------------------------------------------------ frame
    def _frame(self, raw, fitting):
        raw = survey.sanitize(raw)
        if fitting and 'label' not in raw.columns:
            raise ValueError('the fitting frame needs a label column')
        blocks, names = [], []
        if self.keep_structured:
            structured = self._structured_block(raw, fitting)
            if fitting:
                self.fill_ = {}
                for column in structured.columns:
                    valid = structured[column].dropna()
                    if column in ORDINAL_SHORT:
                        value = valid.median() if len(valid) else 0.0
                    else:
                        modes = valid.mode()
                        value = float(modes.iloc[0]) if len(modes) else 0.0
                    self.fill_[column] = float(value)
            structured = structured.fillna(self.fill_)
            structured['payment_amount'] = np.log1p(structured['payment_amount'])
            blocks.append(structured.to_numpy(dtype=float))
            names.extend(structured.columns)
        if fitting:
            self.vocabulary_ = {}
        docs_by_field = tokenised(raw)
        for field in self.text_fields:
            docs = docs_by_field[field]
            block, columns, hit = self._open_block(docs, field, fitting)
            texts = raw[survey.TEXT[field]].map(survey.clean).tolist()
            aux, aux_columns = self._aux_block(texts, docs, hit, field)
            blocks.append(np.hstack([block, aux]))
            names.extend(columns + aux_columns)
        matrix = np.hstack(blocks)
        if fitting:
            self.columns_ = list(names)
        elif list(names) != self.columns_:
            raise AssertionError('transform produced a different column set than fit')
        if not np.isfinite(matrix).all():
            raise AssertionError('feature matrix contains a non-finite value')
        return matrix

    def fit(self, raw):
        self._frame(raw, fitting=True)
        return self

    def transform(self, raw):
        return self._frame(raw, fitting=False)

    def fit_transform(self, raw):
        return self.fit(raw).transform(raw)

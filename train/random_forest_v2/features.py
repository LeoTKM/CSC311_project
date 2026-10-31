"""v2 feature matrix: the structured block plus the four text blocks of PLAN.md.

Everything that depends on the training rows is fitted inside `fit`: the four text
vocabularies (chi-square selection on the fitting rows only) and the structured
imputation values. The colour block is a fixed lexicon and uses no fitted state at all.

Column layout, in order:

    27  structured   ease, strategy, injury, individual, the 21 multi-hot options,
                     payment_amount (mean filled, log1p), payment_missing
     7  flags        q1_missing .. q7_missing
    33  food         30 selected terms, then food_non_answer, food_no_terms,
                     food_item_count
    20  colour       the 17 lexicon colours, then the three colour_ columns
    33  song         as food
    63  feeling      60 selected terms after the standard form, then three feeling_ columns
    --
   183  total

`keep_structured=False` or a restricted `text_fields` drop blocks; the training script
uses neither, they exist so a single block can be inspected in isolation.
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
import text as v2text     # noqa: E402

CLASSES = ['Soccer', 'Baseball', 'Basketball']
TEXT_FIELDS = ('food', 'colour', 'song', 'feeling')

# PLAN.md section 7. colour has no K: its column set is the lexicon.
DEFAULT_K = {'food': 30, 'song': 30, 'feeling': 60}
MIN_DF = 3
TOKEN_KW = dict(hyphen='space', min_len=2, require_letter=True)

# PLAN.md section 3. The standard form applies to the feeling answer only.
STANDARD_FIELDS = ('feeling',)

ORDINAL_SHORT = ['ease', 'strategy', 'injury', 'individual']
FLAG_QUESTIONS = [survey.EASE_COLUMN, *survey.AGREEMENT_COLUMNS, *survey.SCHEMA]


def tokenised(raw):
    """Tokenise all four answers, applying the standard form where PLAN.md says so."""
    out = {}
    for field in TEXT_FIELDS:
        docs = [v2text.tokens(survey.clean(value), **TOKEN_KW)
                for value in raw[survey.TEXT[field]]]
        if field in STANDARD_FIELDS:
            docs = [[v2text.standard_form(token) for token in doc] for doc in docs]
        out[field] = docs
    return out


def chi2_scores(matrix, labels):
    """Chi-square per column against the 3-class label, from the 2x3 table."""
    scores = np.zeros(matrix.shape[1], dtype=float)
    for index in range(matrix.shape[1]):
        present = matrix[:, index] == 1
        table = np.zeros((2, 3))
        for position, name in enumerate(CLASSES):
            mask = labels == name
            table[1, position] = (present & mask).sum()
            table[0, position] = (~present & mask).sum()
        expected = table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True) / len(labels)
        if (expected == 0).any():
            continue
        scores[index] = ((table - expected) ** 2 / expected).sum()
    return scores


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
    def _select_terms(self, docs, labels, field):
        """Candidate pool at min_df, ranked by chi-square, top K kept."""
        counts = Counter()
        for doc in docs:
            counts.update(set(doc))
        pool = sorted(term for term, count in counts.items() if count >= self.min_df)
        if not pool:
            return []
        position = {term: index for index, term in enumerate(pool)}
        matrix = np.zeros((len(docs), len(pool)), dtype=np.int8)
        for row, doc in enumerate(docs):
            for term in doc:
                if term in position:
                    matrix[row, position[term]] = 1
        order = np.argsort(-chi2_scores(matrix, labels), kind='stable')
        return [pool[index] for index in order[:min(self.k_by_field[field], len(pool))]]

    def _open_block(self, docs, raw, field, fitting):
        texts = raw[survey.TEXT[field]].map(survey.clean)
        if field == 'colour':
            columns = list(survey.COLOUR_LEXICON)
            rows = []
            for doc in docs:
                words = set(doc)
                rows.append([int(bool(words & set(survey.COLOUR_LEXICON[name])))
                             for name in columns])
            block = pd.DataFrame(rows, index=raw.index,
                                 columns=[f'colour_{name}' for name in columns])
            matched = [bool(set(doc) & survey.COLOUR_WORDS) for doc in docs]
        else:
            if fitting:
                labels = raw['label'].to_numpy()
                self.vocabulary_[field] = self._select_terms(docs, labels, field)
            columns = self.vocabulary_[field]
            position = {term: index for index, term in enumerate(columns)}
            rows = np.zeros((len(docs), len(columns)), dtype=np.int8)
            for row, doc in enumerate(docs):
                for term in doc:
                    if term in position:
                        rows[row, position[term]] = 1
            block = pd.DataFrame(rows, index=raw.index,
                                 columns=[f'{field}_{term}' for term in columns])
            matched = [bool(set(doc) & set(columns)) for doc in docs]
        return block, matched

    def _aux_block(self, texts, matched, docs, field):
        non_answer = np.array([
            1.0 if (not text) or text in survey.NON_ANSWER else 0.0 for text in texts
        ])
        no_terms = np.array([
            0.0 if declined else (0.0 if hit else 1.0)
            for declined, hit in zip(non_answer, matched)
        ])
        count = np.array([float(len(doc)) for doc in docs])
        return pd.DataFrame({
            f'{field}_non_answer': non_answer,
            f'{field}_no_terms': no_terms,
            f'{field}_item_count': count,
        }, index=texts.index)

    # ------------------------------------------------------------------ structured
    def _structured_block(self, raw, fitting):
        encoded = survey.encode_structured(raw)
        block = pd.DataFrame(
            encoded[survey.ORDINAL_COLUMNS].to_numpy(dtype=float),
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
            for name, column in zip(survey.FLAG_COLUMNS, FLAG_QUESTIONS)
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
            blocks.append(structured)
            names.extend(structured.columns)
        if fitting:
            self.vocabulary_ = {}
        docs_by_field = tokenised(raw)
        for field in self.text_fields:
            docs = docs_by_field[field]
            texts = raw[survey.TEXT[field]].map(survey.clean)
            block, matched = self._open_block(docs, raw, field, fitting)
            aux = self._aux_block(texts, matched, docs, field)
            blocks.append(pd.concat([block, aux], axis=1))
            names.extend(list(block.columns) + list(aux.columns))
        frame = pd.concat(blocks, axis=1)
        if fitting:
            self.columns_ = list(frame.columns)
        elif list(frame.columns) != self.columns_:
            raise AssertionError('transform produced a different column set than fit')
        matrix = frame.to_numpy(dtype=float)
        if not np.isfinite(matrix).all():
            raise AssertionError('feature matrix contains a non-finite value')
        return frame

    def fit(self, raw):
        self._frame(raw, fitting=True)
        return self

    def transform(self, raw):
        return self._frame(raw, fitting=False).to_numpy(dtype=float)

    def fit_transform(self, raw):
        return self.fit(raw).transform(raw)

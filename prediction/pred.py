"""Predict the sport of each survey response. Standard library, NumPy and pandas only.

    from pred import predict_all
    labels = predict_all('data/splits/new_responses.csv')

`predict_all` returns one class label per row of the given CSV, in row order.

The model is the frozen v1 random forest: 300 trees of maximum depth 6, minimum leaf size 3,
and 7 candidate split features per node, over 208 engineered columns. Everything the model
needs lives in two files beside this script:

    model_data.npz        the 300 trees as flat arrays, the IDF vectors, the imputation values
    preprocessing.json    column order, TF-IDF vocabularies, column names, class order

No scikit-learn, PyTorch or TensorFlow is imported, nothing is downloaded, and nothing is
fitted here. The feature construction below is a line-for-line port of the pipeline that
produced those files; `export_model.py` checks the port against the original on the training
rows before the files are written.

Row order and column order are both preserved: the input CSV may arrive with the original
long question headers or with the short names used by the split files, and either way the
208 columns are assembled in the order recorded in `preprocessing.json`.
"""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# ------------------------------------------------------------------ survey vocabulary
EASE_COLUMN = (
    'How easy do you think it would be for someone with no prior experience '
    'to start playing this sport recreationally? Rate it on a scale from 0 to 10, '
    'where 0 means extremely difficult and 10 means extremely easy.'
)
AGREEMENT_COLUMNS = (
    'Success in this sport depends heavily on strategy and planning.',
    'Players must be comfortable with a high risk of injury.',
    'Individual brilliance can outweigh teamwork in this sport.',
)
AGREEMENT_MAP = {
    'strongly disagree': 0.0,
    'disagree': 0.25,
    'neutral': 0.5,
    'agree': 0.75,
    'strongly agree': 1.0,
}
SCHEMA = {
    'Where might you play this sport?': [
        ('Indoor facility', 'location_indoor'),
        ('Outdoor public space', 'location_outdoor'),
        ('School or university facility', 'location_school'),
        ('Community recreation facility', 'location_community'),
        ('Private sports facility', 'location_private'),
        ('At home or nearby', 'location_home'),
        ('Others', 'location_other'),
    ],
    'During which seasons would you be most likely to play this sport recreationally in Toronto?': [
        ('Winter', 'season_winter'),
        ('Summer', 'season_summer'),
        ('Fall', 'season_fall'),
        ('Spring', 'season_spring'),
    ],
    'What types of music do you most associate with this sport?': [
        ('Pop', 'music_pop'),
        ('Electronic or dance', 'music_electronic_dance'),
        ('Latin', 'music_latin'),
        ('Folk or traditional', 'music_folk_traditional'),
        ('Classical or jazz', 'music_classical_jazz'),
        ('Hip-hop or rap', 'music_hiphop_rap'),
        ('Country', 'music_country'),
        ('Rock', 'music_rock'),
        ('No particular type', 'music_no_particular_type'),
        ('Other', 'music_other'),
    ],
}
TEXT = {
    'food': 'What foods and drinks do you associate with watching this sport in the place or culture you know best?',
    'colour': 'What colour do you associate with this sport?',
    'song': 'Which song do you associate with this sport?',
    'feeling': 'How do you feel when you watch this sport?',
}
PAYMENT = 'How much would you be willing to pay for a decent seat at a professional competition of this sport?'
TEXT_FIELDS = ('food', 'colour', 'song', 'feeling')

SHORT_COLUMNS = ['ease', 'strategy', 'injury', 'individual', 'location', 'season',
                 'music_type', 'food', 'colour', 'song', 'feeling', 'payment']
FULL_COLUMNS = [EASE_COLUMN, *AGREEMENT_COLUMNS, *SCHEMA, *TEXT.values(), PAYMENT]

MISSING = {'', 'n/a', 'na', 'nan', 'null', '-', '[]'}
TOKEN_PATTERN = re.compile(r'(?u)\b[a-zA-Z]{2,}\b')
NUMBER = re.compile(r'(?<![\w.])\d+(?:\.\d+)?(?:\s*k)?(?![\d.])')
DASHES = re.compile(r'[\u2013\u2014]')
SCALE_WORDS = re.compile(
    r'\bscale\b|order of magnitude|\d+s\b|\b(?:hundreds?|thousands?|millions?|billions?)\b')
THOUSANDS = re.compile(r'(?<=\d)[, ](?=\d{3}(?:\D|$))')
CURRENCY_CODES = re.compile(r'(?<![a-z])(?:cad|usd|eur|gbp|cny|rmb)(?![a-z])')
CURRENCY_WORDS = re.compile(r'\b(?:canadian|us|dollars?|euros?|bucks|yuan)\b')
CURRENCY_SIGNS = re.compile('[$\\u20ac\\u00a3\\u00a5]')
SPACE_BETWEEN_AMOUNTS = re.compile(r'\s*(?:-|to)\s*')


# ------------------------------------------------------------------ value cleaning
def clean(value):
    """Collapse whitespace, lowercase, and blank out the refusal markers."""
    if pd.isna(value):
        return ''
    text = ' '.join(str(value).split()).lower()
    return '' if text in MISSING else text


def rename_headers(frame):
    """Accept either the original long question headers or the short split-file names."""
    renamed = {}
    for short, full in zip(SHORT_COLUMNS, FULL_COLUMNS):
        if short in frame.columns and full in frame.columns:
            raise ValueError(f'both {short!r} and its long header are present')
        if full not in frame.columns:
            if short not in frame.columns:
                raise ValueError(f'missing survey column: {full!r}')
            renamed[short] = full
    if renamed:
        frame = frame.rename(columns=renamed)
    missing = [name for name in FULL_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f'missing survey column(s): {missing}')
    return frame


def sanitize(frame):
    """Blank out structured answers that are not on the scale, as the pipeline did."""
    frame = rename_headers(frame).reset_index(drop=True).copy()

    ease = pd.to_numeric(frame[EASE_COLUMN].map(clean), errors='coerce')
    frame.loc[~(ease.between(0, 10) & np.isfinite(ease)), EASE_COLUMN] = ''
    for column in AGREEMENT_COLUMNS:
        frame.loc[~frame[column].map(clean).isin(AGREEMENT_MAP), column] = ''
    for column, choices in SCHEMA.items():
        allowed = {clean(option) for option, _ in choices}

        def valid(value, allowed=allowed):
            selected = {part.strip() for part in clean(value).split(',') if part.strip()}
            return bool(selected) and not (selected - allowed)

        frame.loc[~frame[column].map(valid), column] = ''
    return frame


def parse_payment(value):
    """One amount, or the midpoint of a range. Anything else counts as missing."""
    text = DASHES.sub('-', clean(value))
    if not text:
        return None
    if SCALE_WORDS.search(text):
        return None
    text = THOUSANDS.sub('', text)
    text = CURRENCY_CODES.sub(' ', text)
    text = CURRENCY_WORDS.sub(' ', text)
    text = CURRENCY_SIGNS.sub(' ', text)
    matches = list(NUMBER.finditer(text))

    def amount_of(match):
        token = match.group().strip()
        return float(token[:-1]) * 1000.0 if token.endswith('k') else float(token)

    if len(matches) == 1:
        if text[:matches[0].start()].rstrip().endswith('-'):
            return None
        return amount_of(matches[0])
    if len(matches) == 2 and SPACE_BETWEEN_AMOUNTS.fullmatch(
            text[matches[0].end():matches[1].start()]):
        if text[:matches[0].start()].rstrip().endswith('-'):
            return None
        return (amount_of(matches[0]) + amount_of(matches[1])) / 2.0
    return None


# ------------------------------------------------------------------ feature construction
def tfidf_block(texts, terms, idf, vocabulary):
    """Sublinear term frequency times IDF, then L2 rows. Mirrors TfidfVectorizer."""
    matrix = np.zeros((len(texts), len(terms)), dtype=np.float64)
    for row, text in enumerate(texts):
        for token, count in Counter(TOKEN_PATTERN.findall(text)).items():
            index = vocabulary.get(token)
            if index is not None:
                matrix[row, index] = 1.0 + math.log(count)
    matrix *= idf
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def build_features(frame, spec, arrays):
    """Assemble the 208 columns in the recorded order."""
    values = {}

    ease = pd.to_numeric(frame[EASE_COLUMN].map(clean), errors='coerce') / 10.0
    values[EASE_COLUMN] = ease.to_numpy(dtype=np.float64)
    for column in AGREEMENT_COLUMNS:
        values[column] = np.array(
            [AGREEMENT_MAP[a] if (a := clean(v)) else np.nan for v in frame[column]],
            dtype=np.float64)
    for column, choices in SCHEMA.items():
        names = [name for _, name in choices]
        rows = []
        for value in frame[column]:
            answer = clean(value)
            if not answer:
                rows.append([np.nan] * len(choices))
                continue
            selected = {part.strip() for part in answer.split(',') if part.strip()}
            rows.append([float(clean(option) in selected) for option, _ in choices])
        block = np.array(rows, dtype=np.float64).reshape(len(frame), len(names))
        for position, name in enumerate(names):
            values[name] = block[:, position]

    for field in TEXT_FIELDS:
        texts = frame[TEXT[field]].map(clean).tolist()
        block = tfidf_block(texts, spec['text_terms'][field],
                            arrays[f'idf_{field}'],
                            spec['vocabulary'][field])
        for position, term in enumerate(spec['text_terms'][field]):
            values[f'{field}_tfidf_{term}'] = block[:, position]
        values[f'{field}_text_missing'] = np.array([1.0 if not t else 0.0 for t in texts])

    amounts = np.array([np.nan if (a := parse_payment(v)) is None else a
                        for v in frame[PAYMENT]], dtype=np.float64)
    values['payment_missing'] = np.isnan(amounts).astype(np.float64)
    values['payment_amount'] = np.where(np.isnan(amounts),
                                        spec['payment_fill_value'], amounts)

    for index, column in enumerate([EASE_COLUMN, *AGREEMENT_COLUMNS, *SCHEMA], start=1):
        values[f'q{index}_missing'] = frame[column].map(clean).eq('').astype(float).to_numpy()

    columns = spec['columns']
    matrix = np.empty((len(frame), len(columns)), dtype=np.float64)
    for position, name in enumerate(columns):
        column = values[name]
        matrix[:, position] = np.where(np.isnan(column), arrays['fill'][position], column)
    matrix[:, columns.index('payment_amount')] = np.log1p(
        matrix[:, columns.index('payment_amount')])
    if not np.isfinite(matrix).all():
        raise ValueError('the feature matrix contains a non-finite value')
    return matrix


def traverse(arrays, matrix, classes):
    """Descend all 300 trees with NumPy only and average the leaf distributions."""
    rows = np.arange(matrix.shape[0])
    votes = np.zeros((matrix.shape[0], len(classes)), dtype=np.float64)
    offsets = arrays['offsets']
    for tree in range(len(offsets) - 1):
        start = int(offsets[tree])
        left = arrays['left'][start:offsets[tree + 1]]
        right = arrays['right'][start:offsets[tree + 1]]
        feature = arrays['feature'][start:offsets[tree + 1]]
        threshold = arrays['threshold'][start:offsets[tree + 1]]
        value = arrays['value'][start:offsets[tree + 1]]
        node = np.zeros(matrix.shape[0], dtype=np.int32)
        while True:
            leaf = left[node] < 0
            if leaf.all():
                break
            column = np.where(leaf, 0, feature[node])
            go_left = matrix[rows, column] <= threshold[node]
            step = np.where(go_left, left[node], right[node])
            node = np.where(leaf, node, step)
        votes += value[node]
    return [str(classes[index]) for index in votes.argmax(axis=1)]


# ------------------------------------------------------------------ loading and predicting
_CACHE = {}


def load(model_dir=None):
    """Load the exported arrays and the recorded feature specification."""
    directory = Path(model_dir) if model_dir is not None else HERE
    key = str(directory)
    if key not in _CACHE:
        with np.load(directory / 'model_data.npz') as handle:
            arrays = {name: handle[name] for name in handle.files}
        spec = json.loads((directory / 'preprocessing.json').read_text(encoding='utf-8'))
        vocabulary = {field: {term: index for index, term in enumerate(terms)}
                      for field, terms in spec['text_terms'].items()}
        spec['vocabulary'] = vocabulary
        _CACHE[key] = (arrays, spec)
    return _CACHE[key]


def predict_frame(frame, model_dir=None):
    """Predict from an already-loaded DataFrame. The `id` and `label` columns are ignored."""
    if frame.empty:
        return []
    arrays, spec = load(model_dir)
    sanitized = sanitize(frame)
    matrix = build_features(sanitized, spec, arrays)
    return traverse(arrays, matrix, spec['classes'])


def predict_all(csv_filename, model_dir=None):
    """Return one predicted class label per row of `csv_filename`, in row order."""
    frame = pd.read_csv(csv_filename, dtype=str, keep_default_na=False,
                        encoding='utf-8-sig')
    return predict_frame(frame, model_dir)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__.strip())
        return 1
    source = argv[0]
    labels = predict_all(source)
    target = Path(argv[1]) if len(argv) > 1 else None
    if target is None:
        print(f'{len(labels)} prediction(s) for {source}')
        for index, label in enumerate(labels):
            print(f'{index}\t{label}')
    else:
        pd.DataFrame({'prediction': labels}).to_csv(target, index=False)
        print(f'wrote {len(labels)} prediction(s) to {target}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

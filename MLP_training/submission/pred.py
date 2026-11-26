"""Section 3.4 MLP inference: standard library, NumPy and pandas only.

Keep mlp_params.npz next to this file. Parameters and preprocessing were fitted
on train + validation (738 rows). No fitting or downloads occur at inference.
predict_all(filename) returns one sport label per CSV row in input order.
Run: python pred.py unlabelled.csv
"""
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mlp_params.npz')

# ----------------------------------------------------------------------------- survey schema
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
AGREEMENT_MAP = {'strongly disagree': 0.0, 'disagree': 0.25, 'neutral': 0.5,
                 'agree': 0.75, 'strongly agree': 1.0}
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
FULL_COLUMNS = [EASE_COLUMN, *AGREEMENT_COLUMNS, *SCHEMA, *TEXT.values(), PAYMENT]
SHORT_COLUMNS = ['ease', 'strategy', 'injury', 'individual', 'location', 'season',
                 'music_type', 'food', 'colour', 'song', 'feeling', 'payment']
MISSING = {'', 'n/a', 'na', 'nan', 'null', '-', '[]'}
NUMBER = re.compile(r'(?<![\w.])\d+(?:\.\d+)?(?:\s*k)?(?![\d.])')


# ----------------------------------------------------------------------------- parameters
_PARAMS = None


def load_params(path=PARAMS_PATH):
    global _PARAMS
    if _PARAMS is None:
        with np.load(path, allow_pickle=False) as f:
            p = {k: f[k] for k in f.files}
        p['meta'] = json.loads(str(p.pop('meta_json')))
        p['col_index'] = {str(c): i for i, c in enumerate(p['feature_columns'])}
        _PARAMS = p
    return _PARAMS


# ----------------------------------------------------------------------------- cleaning
def clean(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ''
    text = ' '.join(str(value).split()).lower()
    return '' if text in MISSING else text


def parse_payment(value):
    """One number, or a 'a-b' / 'a to b' range (midpoint); '1k' = 1000; else None."""
    text = clean(value).replace('–', '-').replace('—', '-')
    if not text:
        return None
    if re.search(r'\bscale\b|order of magnitude|\d+s\b|\b(?:hundreds?|thousands?|millions?|billions?)\b', text):
        return None
    text = re.sub(r'(?<=\d)[, ](?=\d{3}(?:\D|$))', '', text)
    text = re.sub(r'(?<![a-z])(?:cad|usd|eur|gbp|cny|rmb)(?![a-z])', ' ', text)
    text = re.sub(r'\b(?:canadian|us|dollars?|euros?|bucks|yuan)\b', ' ', text)
    text = re.sub(r'[$€£¥]', ' ', text)
    matches = list(NUMBER.finditer(text))
    if not matches:
        return None

    def number(match):
        token = match.group().strip()
        return float(token[:-1]) * 1000 if token.endswith('k') else float(token)

    if len(matches) == 1:
        if text[:matches[0].start()].rstrip().endswith('-'):
            return None
        amount = number(matches[0])
    elif len(matches) == 2 and re.fullmatch(r'\s*(?:-|to)\s*', text[matches[0].end():matches[1].start()]):
        if text[:matches[0].start()].rstrip().endswith('-'):
            return None
        amount = (number(matches[0]) + number(matches[1])) / 2
    else:
        return None
    return amount if math.isfinite(amount) else None


def _ease(answer):
    """Valid 0-10 number -> value/10, otherwise None (treated as missing)."""
    answer = clean(answer)
    try:
        value = float(answer)
    except ValueError:
        return None
    return value / 10 if math.isfinite(value) and 0 <= value <= 10 else None


def _multi(answer, choices):
    """Comma-separated options -> 0/1 list; blank or any unknown option -> None (missing)."""
    answer = clean(answer)
    allowed = {clean(option) for option, _ in choices}
    selected = {part.strip() for part in answer.split(',') if part.strip()}
    if not selected or selected - allowed:
        return None
    return [float(clean(option) in selected) for option, _ in choices]


def _tfidf(text, enc_vocab_index, idf, pattern):
    """sklearn TfidfVectorizer(sublinear_tf=True, smooth idf, norm='l2') for one document."""
    counts = {}
    for token in pattern.findall(text.lower()):
        j = enc_vocab_index.get(token)
        if j is not None:
            counts[j] = counts.get(j, 0) + 1
    vec = np.zeros(len(idf))
    for j, c in counts.items():
        vec[j] = (1.0 + math.log(c)) * idf[j]
    norm = np.sqrt((vec ** 2).sum())
    return vec / norm if norm > 0 else vec


# ----------------------------------------------------------------------------- features
def _standardize_headers(df):
    """Accept the original survey headers or the short ones; missing columns -> blank."""
    df = df.copy()
    df.columns = [' '.join(str(c).split()) for c in df.columns]
    for short, full in zip(SHORT_COLUMNS, FULL_COLUMNS):
        if full not in df.columns:
            df[full] = df[short] if short in df.columns else ''
    return df


def build_features(df):
    """Raw survey DataFrame -> (n, 208) model inputs, in the order of feature_columns."""
    p = load_params()
    df = _standardize_headers(df)
    n = len(df)
    X = np.full((n, len(p['feature_columns'])), np.nan)
    col = p['col_index']

    def put(name, values):
        if name in col:
            X[:, col[name]] = values

    # Q1 ease and Q2-Q4 agreement (NaN = missing / invalid) + q1..q4_missing flags
    ease = [_ease(v) for v in df[EASE_COLUMN]]
    put(EASE_COLUMN, [np.nan if v is None else v for v in ease])
    put('q1_missing', [float(v is None) for v in ease])
    for q, column in enumerate(AGREEMENT_COLUMNS, 2):
        vals = [AGREEMENT_MAP.get(clean(v)) for v in df[column]]
        put(column, [np.nan if v is None else v for v in vals])
        put(f'q{q}_missing', [float(v is None) for v in vals])

    # Q5-Q7 multi-hot + q5..q7_missing flags
    for q, (column, choices) in enumerate(SCHEMA.items(), 5):
        rows = [_multi(v, choices) for v in df[column]]
        for k, (_, name) in enumerate(choices):
            put(name, [np.nan if r is None else r[k] for r in rows])
        put(f'q{q}_missing', [float(r is None) for r in rows])

    # Q8-Q11 text: TF-IDF with the training vocabulary + <field>_text_missing
    pattern = re.compile(str(p['text_token_pattern']))
    for field, column in TEXT.items():
        texts = [clean(v) for v in df[column]]
        key = f'text_{field}_vocabulary'
        if key in p:
            vocab = [str(t) for t in p[key]]
            index = {t: j for j, t in enumerate(vocab)}
            idf = p[f'text_{field}_idf']
            block = np.array([_tfidf(t, index, idf, pattern) for t in texts]).reshape(n, len(vocab))
            for j, term in enumerate(vocab):
                put(f'{field}_tfidf_{term}', block[:, j])
        put(f'{field}_text_missing', [float(t == '') for t in texts])

    # Q12 payment: parsed amount (missing -> training mean) + payment_missing
    amounts = [parse_payment(v) for v in df[PAYMENT]]
    fill = float(p['payment_fill_value'])
    put('payment_amount', [fill if a is None else a for a in amounts])
    put('payment_missing', [float(a is None) for a in amounts])

    # Impute remaining NaNs (training medians/modes), log1p(payment), standardize
    nan = np.isnan(X)
    X[nan] = np.broadcast_to(p['fill_values'], X.shape)[nan]
    m = p['log1p_mask']
    X[:, m] = np.log1p(X[:, m])
    return (X - p['scale_mean']) / p['scale_std']


# ----------------------------------------------------------------------------- model
def predict_proba(X):
    p = load_params()
    h = X
    n = int(p['n_linear'])
    for i in range(n - 1):
        h = np.maximum(0.0, h @ p[f'W{i}'] + p[f'b{i}'])
    logits = h @ p[f'W{n - 1}'] + p[f'b{n - 1}']
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def predict_df(df):
    p = load_params()
    probs = predict_proba(build_features(df))
    return [str(c) for c in p['classes'][probs.argmax(axis=1)]], probs


def predict_all(filename):
    """Required by the course: CSV file name -> list of predicted labels, one per row."""
    df = pd.read_csv(filename, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    predictions, _ = predict_df(df)
    return predictions


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python pred.py INPUT.csv')
    print('\n'.join(predict_all(sys.argv[1])))

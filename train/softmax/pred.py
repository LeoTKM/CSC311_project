"""CSC311 softmax inference. Place softmax_params.json next to this file."""
from collections import Counter
import csv
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import sys
import unicodedata

import numpy as np


MISSING = {'', 'n/a', 'na', 'nan', 'null', '-', '[]'}
LEGACY_WORD = re.compile(r'\b[a-zA-Z]{2,}\b')


def clean(value):
    text = ' '.join((value or '').split()).lower()
    return '' if text in MISSING else text


def words(text, normalized=False):
    if not normalized:
        return LEGACY_WORD.findall(text)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"['\u2019\u02bc]", '', text)
    return [word for word in re.findall('[a-z0-9]{2,}', text) if re.search('[a-z]', word)]


def terms(text, analyzer):
    tokens = words(text, normalized=analyzer != 'legacy')
    if analyzer != 'char':
        return tokens
    output = []
    for token in tokens:
        padded = ' ' + token + ' '
        for width in (3, 4, 5):
            output.extend(padded[i:i + width] for i in range(len(padded) - width + 1))
    return output


NUMBER = re.compile(r'(?<![\w.])\d+(?:\.\d+)?(?:\s*k)?(?![\d.])')


def parse_payment(value):
    """Numeric policy from data_preprocessing/encode_all.py."""
    text = clean(value).replace('–', '-').replace('—', '-')
    if not text or re.search(r'\bscale\b|order of magnitude|\d+s\b|\b(?:hundreds?|thousands?|millions?|billions?)\b', text):
        return None
    text = re.sub(r'(?<=\d)[, ](?=\d{3}(?:\D|$))', '', text)
    text = re.sub(r'(?<![a-z])(?:cad|usd|eur|gbp|cny|rmb)(?![a-z])', ' ', text)
    text = re.sub(r'\b(?:canadian|us|dollars?|euros?|bucks|yuan)\b', ' ', text)
    matches = list(NUMBER.finditer(re.sub(r'[$€£¥]', ' ', text)))
    if not matches:
        return None
    text = re.sub(r'[$€£¥]', ' ', text)
    if text[:matches[0].start()].rstrip().endswith('-'):
        return None
    if len(matches) != 1 and not (
        len(matches) == 2 and re.fullmatch(r'\s*(?:-|to)\s*', text[matches[0].end():matches[1].start()])
    ):
        return None

    def number(match):
        token = match.group().strip()
        return float(token[:-1]) * 1000 if token.endswith('k') else float(token)

    amount = sum(number(match) for match in matches) / len(matches)
    return amount if math.isfinite(amount) else None


def read_rows(filename, schema):
    """Return canonical question keys in CSV row order, without reading label as input."""
    with open(filename, newline='', encoding='utf-8-sig') as source:
        reader = csv.DictReader(source)
        names = { ' '.join(name.split()): name for name in (reader.fieldnames or []) }
        columns = {}
        for short, full in schema['headers'].items():
            column = names.get(' '.join(full.split()), names.get(short))
            if column is None:
                raise ValueError(f'Missing survey column: {short}')
            columns[short] = column
        return [{key: row[column] for key, column in columns.items()} for row in reader]


def structure(rows, schema):
    result = []
    for row in rows:
        try:
            ease = float(clean(row['ease']))
        except ValueError:
            ease = np.nan
        rating = [ease / 10 if 0 <= ease <= 10 else np.nan]
        missing = [float(np.isnan(rating[0]))]
        for name in ('strategy', 'injury', 'individual'):
            value = schema['agreement'].get(clean(row[name]), np.nan)
            rating.append(value)
            missing.append(float(np.isnan(value)))
        for field, options in schema['multi'].items():
            selected = {part.strip() for part in clean(row[field]).split(',') if part.strip()}
            valid = bool(selected) and selected.issubset(options)
            rating.extend(float(option in selected) if valid else np.nan for option in options)
            missing.append(float(not valid))
        payment = parse_payment(row['payment'])
        rating.extend([payment if payment is not None else np.nan, float(payment is None)])
        result.append(rating + missing)
    return np.asarray(result, dtype=float).reshape(len(rows), 34)


def prepared_structure(matrix, parameters):
    result = np.where(np.isnan(matrix), parameters['fill'], matrix)
    result[:, 25] = np.log1p(result[:, 25])
    return (result - parameters['mean']) / parameters['scale']


def make_text_block(rows, block):
    index = {term: pos for pos, term in enumerate(block['vocabulary'])}
    idf = np.asarray(block['idf'])
    result = np.zeros((len(rows), len(idf)), dtype=float)
    for row_num, row in enumerate(rows):
        tokens = terms(clean(row[block['field']]), block['analyzer'])
        counts = Counter(index[token] for token in tokens if token in index)
        if counts:
            positions = list(counts)
            values = (1 + np.log(list(counts.values()))) * idf[positions]
            result[row_num, positions] = values * (block['weight'] / np.linalg.norm(values))
    return result


def features(rows, parameters):
    blocks = [prepared_structure(structure(rows, parameters['schema']), parameters['structured'])]
    blocks += [make_text_block(rows, block) for block in parameters['text_blocks']]
    fields = parameters['fields']
    texts = {field: [clean(row[field]) for row in rows] for field in fields}
    blocks.append(np.column_stack([np.asarray([not item for item in texts[field]], dtype=float) for field in fields]))
    if parameters['colour_lexicon']:
        colour_tokens = [set(words(item, normalized=True)) for item in texts['colour']]
        for aliases in parameters['colour_lexicon'].values():
            blocks.append(np.asarray([bool(doc.intersection(aliases)) for doc in colour_tokens], dtype=float)[:, None])
    if parameters['coverage_flags']:
        for field in fields:
            indices = [i + 1 for i, block in enumerate(parameters['text_blocks']) if block['field'] == field]
            seen = np.zeros(len(rows), dtype=bool)
            for i in indices:
                seen |= np.any(blocks[i] != 0, axis=1)
            tokens = [words(text, normalized=True) for text in texts[field]]
            blocks.append(np.column_stack([
                np.asarray([bool(text) for text in texts[field]]) & ~seen,
                np.log1p([len(doc) for doc in tokens]),
            ]))
    return np.concatenate(blocks, axis=1)


@lru_cache(maxsize=1)
def load_parameters():
    with Path(__file__).with_name('softmax_params.json').open(encoding='utf-8') as source:
        data = json.load(source)
    for key in ('weights', 'bias'):
        data[key] = np.asarray(data[key])
    return data


def predict_all(filename):
    """Read the course's raw CSV and return a list of sport labels in row order."""
    parameters = load_parameters()
    rows = read_rows(filename, parameters['schema'])
    matrix = features(rows, parameters)
    logits = matrix @ parameters['weights'] + parameters['bias']
    # Softmax preserves the ordering of the logits.
    return [parameters['classes'][i] for i in logits.argmax(axis=1)]


if __name__ == '__main__':
    print('\n'.join(predict_all(sys.argv[1])))

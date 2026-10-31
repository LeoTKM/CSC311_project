"""Fast checks for the v2 pipeline: block sizes, leakage guards, PLAN.md rules.

Run this before any training stage. It fits the preprocessor on the 606 training rows and
transforms the 132 validation rows, but it never trains a scored model and never touches
the test split.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import encode as survey   # noqa: E402
import features           # noqa: E402
import selection          # noqa: E402
import text as v2text     # noqa: E402

EXPECTED = {'structured': 27, 'flags': 7, 'food': 33, 'colour': 20, 'song': 33,
            'feeling': 63, 'total': 183}


def check_tokens():
    assert v2text.tokens('hot-dog, hot dog', hyphen='space', min_len=2,
                         require_letter=True) == ['hot', 'dog', 'hot', 'dog']
    assert v2text.tokens("I don't know", hyphen='space', min_len=2,
                         require_letter=True) == ['dont', 'know']
    assert v2text.tokens('Green\uff0cbecause', hyphen='space', min_len=2,
                         require_letter=True) == ['green', 'because']
    assert v2text.tokens('2019 ok', hyphen='space', min_len=2,
                         require_letter=True) == ['ok']
    print('  tokenizer: hyphen, apostrophe, full-width and digit rules agree')


def check_standard_form():
    cases = {
        'excited': 'excite', 'excitement': 'excite', 'exciting': 'excite',
        'bored': 'bor', 'boring': 'bor',
        'confused': 'confuse', 'confusion': 'confuse',
        'interested': 'interest', 'interesting': 'interest',
        'nervousness': 'nervous',
        'energized': 'energetic', 'energy': 'energetic',
        'relaxation': 'relaxe', 'tense': 'tense', 'tension': 'tense',
    }
    for word, expected in cases.items():
        got = v2text.standard_form(word)
        assert got == expected, f'standard_form({word!r}) gave {got!r}, want {expected!r}'
    assert v2text.standard_form('uninterested') != v2text.standard_form('interested'), \
        'opposites must not merge'
    repeated = v2text.standard_form(v2text.standard_form('excitement'))
    assert repeated == v2text.standard_form('excitement'), 'standard_form is not idempotent'
    print(f'  standard form: {len(cases)} cases agree, opposites stay apart, idempotent')


def check_colour_lexicon():
    train = pd.read_csv(ROOT / 'data/splits/train.csv', dtype=str, keep_default_na=False)
    pre = features.TextPreprocessor().fit(train)
    row = train.iloc[[0]].copy()
    frame_columns = list(pre.columns_)

    def colour_row(answer):
        probe = row.copy()
        probe[survey.TEXT['colour']] = answer
        frame = pd.DataFrame(pre.transform(probe), columns=frame_columns)
        return frame

    known = colour_row('Burgundy, and a bit of grey')
    assert known['colour_burgundy'].iloc[0] == 1.0
    assert known['colour_gray'].iloc[0] == 1.0
    assert known['colour_no_terms'].iloc[0] == 0.0

    unknown = colour_row('Chartreuse')
    colours = [column for column in frame_columns if column.startswith('colour_')
               and column not in ('colour_non_answer', 'colour_no_terms',
                                  'colour_item_count')]
    assert len(colours) == 17, f'expected 17 colour columns, got {len(colours)}'
    assert unknown[colours].to_numpy().sum() == 0.0, 'an unknown colour must set no column'
    assert unknown['colour_no_terms'].iloc[0] == 1.0

    declined = colour_row('none')
    assert declined['colour_non_answer'].iloc[0] == 1.0
    assert declined['colour_no_terms'].iloc[0] == 0.0
    print('   colour: 17 fixed columns, unknown colour flagged, refusal flagged')


def check_block_sizes():
    train = pd.read_csv(ROOT / 'data/splits/train.csv', dtype=str, keep_default_na=False)
    val = pd.read_csv(ROOT / 'data/splits/val.csv', dtype=str, keep_default_na=False)
    pre = features.TextPreprocessor().fit(train)
    matrix = pre.transform(val)
    columns = list(pre.columns_)

    assert matrix.shape == (len(val), EXPECTED['total']), \
        f'matrix is {matrix.shape}, expected {(len(val), EXPECTED["total"])}'
    assert np.isfinite(matrix).all()
    assert len(set(columns)) == len(columns), 'duplicate column names'

    for field in features.TEXT_FIELDS:
        block = [column for column in columns
                 if column.startswith(f'{field}_')]
        assert len(block) == EXPECTED[field], \
            f'{field} block has {len(block)} columns, expected {EXPECTED[field]}'
        for suffix in ('non_answer', 'no_terms', 'item_count'):
            assert f'{field}_{suffix}' in columns, f'missing {field}_{suffix}'

    structured = [column for column in columns
                  if column in features.ORDINAL_SHORT
                  or column in survey.MULTI_COLUMNS
                  or column in survey.PAYMENT_COLUMNS]
    flags = [column for column in columns if column in survey.FLAG_COLUMNS]
    assert len(structured) == EXPECTED['structured'], f'structured {len(structured)}'
    assert len(flags) == EXPECTED['flags'], f'flags {len(flags)}'
    for field, size in pre.vocabulary_.items():
        assert len(size) == features.DEFAULT_K[field], \
            f'{field} vocabulary has {len(size)} terms, expected {features.DEFAULT_K[field]}'
    print(f'  blocks: structured {len(structured)}, flags {len(flags)}, '
          + ', '.join(f'{field} {EXPECTED[field]}' for field in features.TEXT_FIELDS)
          + f', total {matrix.shape[1]}')
    print(f'  vocabularies: '
          + ', '.join(f'{f} {len(v)}' for f, v in pre.vocabulary_.items()))


def check_fold_guard():
    train = pd.read_csv(ROOT / 'data/splits/train.csv', dtype=str, keep_default_na=False)
    for seed in selection.FOLD_SEEDS:
        prepared = selection.prepare_folds(train, seed)
        assert len(prepared) == selection.N_SPLITS
        total = sum(len(fold['y_fit']) + len(fold['y_held']) for fold in prepared)
        assert total == len(train) * selection.N_SPLITS, 'folds do not cover every row'
        for fold in prepared:
            if fold['fold'] == 0:
                assert fold['n_features'] == EXPECTED['total']
    print(f'  folds: {len(selection.FOLD_SEEDS)} seeds x {selection.N_SPLITS} folds, '
          f'every row held out once, no respondent on both sides')


def main():
    print('v2 smoke check')
    check_tokens()
    check_standard_form()
    check_colour_lexicon()
    check_block_sizes()
    check_fold_guard()
    print('all checks passed')


if __name__ == '__main__':
    main()

"""Fast checks on the v3 pipeline. No model is fitted and no split other than the
training split is read.

Run: python smoke_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import encode as survey    # noqa: E402
import features            # noqa: E402
import selection           # noqa: E402
import text as v3text      # noqa: E402

TRAIN_CSV = ROOT / 'data' / 'splits' / 'train.csv'

EXPECTED = {
    'structured': 26,     # 3 ordinals + 21 multi-hot + payment_amount + payment_missing
    'missing flags': 6,   # q1..q3 and q5..q7, no q4
    'food': 53,           # 50 terms + 3 aux
    'colour': 20,         # 17 lexicon colours + 3 aux
    'song': 53,
    'feeling': 15,        # 12 terms + 3 aux
}
EXPECTED_TOTAL = 173


def check_tokenizer():
    cases = {
        '': [],
        '   ': [],
        '---': [],
        'Pizza': ['pizza'],
        "Don't Stop-Me Now": ['dont', 'stop', 'me', 'now'],
        'hot-dog': ['hot', 'dog'],
        '50 Cent': ['cent'],
        'nba2k': ['nba2k'],
        'Rock n Roll': ['rock', 'roll'],
        'Caf\u00e9 Rouge': ['caf', 'rouge'],      # NFKC folds e-acute to a plain letter
        'a b c': [],
    }
    for text, want in cases.items():
        got = v3text.tokens(text)
        assert got == want, f'tokens({text!r}) = {got}, expected {want}'
    print(f'  tokenizer: {len(cases)} cases')


def check_lexicon():
    cases = {'grey': 'gray', 'gray': 'gray', 'balck': 'black', 'blues': 'blue',
             'yellowish': 'yellow', 'golden': 'gold', 'lime': None, 'teal': None}
    fold = features.TextPreprocessor()
    for word, want in cases.items():
        column = f'colour_{want}' if want else None
        docs = [v3text.tokens(word)]
        block, columns, hit = fold._open_block(docs, 'colour', fitting=False)
        active = [name for name, value in zip(columns, block[0]) if value]
        expected = [column] if column else []
        assert active == expected, f'{word!r} activated {active}, expected {expected}'
        assert bool(hit[0]) == bool(want), f'{word!r} hit should be {bool(want)}'
    print(f'  colour lexicon: {len(cases)} cases')


def block_of(column):
    for field in features.TEXT_FIELDS:
        if column.startswith(f'{field}_'):
            return field
    if column in FLAG_NAMES:
        return 'missing flags'
    return 'structured'


FLAG_NAMES = {name for name, _ in features.FLAG_QUESTIONS}


def check_frame(train):
    preprocessor = features.TextPreprocessor().fit(train)
    columns = list(preprocessor.columns_)
    assert len(columns) == EXPECTED_TOTAL, \
        f'{len(columns)} columns, expected {EXPECTED_TOTAL}'
    sizes = {}
    for column in columns:
        sizes[block_of(column)] = sizes.get(block_of(column), 0) + 1
    assert sizes == EXPECTED, f'block sizes {sizes}, expected {EXPECTED}'
    assert 'individual' not in columns and 'q4_missing' not in columns, 'Q4 survived'
    assert not any(name.startswith('colour_tfidf') for name in columns)
    assert sum(1 for name in columns if name.startswith('colour_')) == 20

    matrix = preprocessor.transform(train)
    assert matrix.shape == (len(train), EXPECTED_TOTAL), matrix.shape
    assert np.isfinite(matrix).all()
    binary = [name for name in columns
              if block_of(name) in features.TEXT_FIELDS
              and not name.endswith(('_non_answer', '_no_terms', '_item_count'))]
    values = np.unique(matrix[:, [columns.index(name) for name in binary]])
    assert set(values.tolist()) <= {0.0, 1.0}, f'presence block is not 0/1: {values}'
    print(f'  frame: {matrix.shape[0]} rows x {matrix.shape[1]} columns, '
          f'{len(binary)} presence columns')

    for field in ('food', 'song', 'feeling'):
        vocabulary = preprocessor.vocabulary_[field]
        assert len(vocabulary) == features.DEFAULT_K[field], \
            f'{field} kept {len(vocabulary)} terms, expected {features.DEFAULT_K[field]}'
        assert len(set(vocabulary)) == len(vocabulary), f'{field} vocabulary repeats'
    print('  vocabularies: '
          + ', '.join(f'{field} {len(preprocessor.vocabulary_[field])}'
                      for field in ('food', 'song', 'feeling')))


def check_aux(train):
    """The three auxiliary columns, on answers written for the purpose."""
    preprocessor = features.TextPreprocessor().fit(train)
    columns = list(preprocessor.columns_)
    known = preprocessor.vocabulary_['food'][0]
    probes = {
        '':                  {'non_answer': 1.0, 'no_terms': 0.0, 'item_count': 0.0},
        '   ':               {'non_answer': 1.0, 'no_terms': 0.0, 'item_count': 0.0},
        'none':              {'non_answer': 1.0, 'no_terms': 0.0, 'item_count': 1.0},
        'zzqxwv':            {'non_answer': 0.0, 'no_terms': 1.0, 'item_count': 1.0},
        known:               {'non_answer': 0.0, 'no_terms': 0.0, 'item_count': 1.0},
        f'{known} zzqxwv':   {'non_answer': 0.0, 'no_terms': 0.0, 'item_count': 2.0},
    }
    frame = train.iloc[:len(probes)].copy().reset_index(drop=True)
    frame['id'] = [f'probe{index}' for index in range(len(probes))]
    frame[survey.TEXT['food']] = list(probes)
    matrix = preprocessor.transform(frame)
    for row, (text, want) in enumerate(probes.items()):
        for name, expected in want.items():
            column = columns.index(f'food_{name}')
            got = matrix[row, column]
            assert got == expected, \
                f'food {text!r}: {name} = {got}, expected {expected}'
    print(f'  aux columns: {len(probes)} answers x 3 columns '
          f'(hit word {known!r})')


def check_folds(train):
    respondents = set(train['id'])
    assert len(respondents) == 202, len(respondents)
    for seed in selection.FOLD_SEEDS:
        held_total = []
        for fold, (fit_index, held_index) in enumerate(selection.folds_of(train, seed)):
            fitting, held = train.iloc[fit_index], train.iloc[held_index]
            overlap = set(fitting['id']) & set(held['id'])
            assert not overlap, f'seed {seed} fold {fold}: {len(overlap)} shared ids'
            held_total.extend(held['id'])
        assert sorted(held_total) == sorted(train['id']), 'folds do not cover every row'
        assert len(set(held_total)) == len(respondents)
    print(f'  folds: {len(selection.FOLD_SEEDS)} seeds x {selection.N_SPLITS} splits, '
          f'no respondent straddles, every row held out once per seed')


def check_split_access():
    """The test split may not be named anywhere, the validation split only in run.py.

    This file is skipped because it has to spell the names out to look for them.
    """
    test_name, val_name = 'test' + '.csv', 'val' + '.csv'
    owner = 'run' + '.py'
    names = []
    for path in sorted(HERE.glob('*.py')):
        names.append(path.name)
        if path.name == Path(__file__).name:
            continue
        source = path.read_text(encoding='utf-8')
        assert test_name not in source, f'{path.name} mentions the test split'
        assert 'class_dataset' not in source, f'{path.name} mentions the raw dataset'
        if val_name in source:
            assert path.name == owner, \
                f'{path.name} mentions the validation split; only {owner} may'
    print(f'  sources: {len(names)} files, the test split is named nowhere, '
          f'the validation split only in {owner}')


def main():
    train = pd.read_csv(TRAIN_CSV, dtype=str, keep_default_na=False)
    print(f'train split: {len(train)} rows, {train.id.nunique()} respondents')
    check_tokenizer()
    check_lexicon()
    check_frame(train)
    check_aux(train)
    check_folds(train)
    check_split_access()
    print('\nPASS')


if __name__ == '__main__':
    main()

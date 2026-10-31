"""Which short-answer design choices actually pay: v1 against v2, one at a time.

Reads `data/splits/train.csv` only (606 rows / 202 respondents). Neither `val.csv` nor
`test.csv` is opened anywhere in this file.

Eight decisions separate the two existing text pipelines:

    decision      v1                                  v2
    ------------  ----------------------------------  -------------------------------
    tokenizer     regex ``\\b[a-zA-Z]{2,}\\b``          NFKC -> apostrophes -> non-alnum
    min_df        2                                   3
    selection     most frequent term wins             highest chi-square wins
    value         tf-idf, L2-normalised rows          0/1 presence
    colour        the same tf-idf rule as every field a fixed 17-concept lexicon
    aux columns   1 per field (``*_text_missing``)     3 per field
    feeling form  none                                suffix standard form
    caps          50 / 20 / 50 / 50                   30 / -- / 30 / 60

The run starts from v1's setting, changes exactly one decision at a time, then
assembles every change that paid off into a candidate for v3 (and Q4 is already gone:
``individual`` and ``q4_missing`` are dropped from the structured block).

Every vocabulary, every chi-square score and every idf value is fitted on the fitting
rows of one fold and merely applied to that fold's held-out rows.
"""
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
TRAIN_DIR = HERE.parents[1]
ROOT = HERE.parents[2]
V2 = TRAIN_DIR / 'random_forest_v2'
for path in (str(V2),):
    if path not in sys.path:
        sys.path.insert(0, path)

import encode as survey          # noqa: E402  v2's copy: clean, NON_ANSWER, colour lexicon
import features as v2features    # noqa: E402  structured block only
import selection                 # noqa: E402  folds, forest factory
import text as v2text            # noqa: E402  the explicit tokenizer and standard form

TRAIN_CSV = ROOT / 'data' / 'splits' / 'train.csv'
OUT_DIR = HERE / 'out'
CLASSES = list(selection.CLASSES)
TEXT_FIELDS = ('food', 'colour', 'song', 'feeling')
COLOUR_NAMES = list(survey.COLOUR_LEXICON)

# v1's tokenizer, verbatim from data_preprocessing/encode_all.py.
REGEX_TOKEN = re.compile(r'(?u)\b[a-zA-Z]{2,}\b')
# v2's tokenizer arguments, verbatim from random_forest_v2/features.py.
EXPLICIT_KW = dict(hyphen='space', min_len=2, require_letter=True)

# v3 drops Q4, so the structured block loses its value column and its flag.
DROP_STRUCTURED = ('individual', 'q4_missing')

FOREST_SETTINGS = {
    'v2_setting': dict(max_depth=8, min_samples_leaf=2, max_features='log2'),
    'v1_setting': dict(max_depth=6, min_samples_leaf=3, max_features='log2'),
}

V1 = dict(tokenizer='regex', min_df=2, select='tf', value='tfidf', colour='tfidf',
          aux='v1', standard=False, drop=(),
          caps={'food': 50, 'colour': 20, 'song': 50, 'feeling': 50})
V2 = dict(tokenizer='explicit', min_df=3, select='chi2', value='binary', colour='lexicon',
          aux='v2', standard=True, drop=(),
          caps={'food': 30, 'colour': 20, 'song': 30, 'feeling': 60})


def variant(**changes):
    """A copy of v1's setting with the named decisions replaced."""
    config = {key: (dict(value) if isinstance(value, dict) else value)
              for key, value in V1.items()}
    config.update(changes)
    return config


V3_CAPS = {'food': 50, 'colour': 20, 'song': 50, 'feeling': 60}
V3 = dict(tokenizer='explicit', select='chi2', value='binary', colour='lexicon',
          aux='v2', standard=False, min_df=2, drop=(), caps=dict(V3_CAPS))

CONFIGS = {
    'v1': V1,
    'v2': V2,
    'v1+tokenizer': variant(tokenizer='explicit'),
    'v1+min_df3': variant(min_df=3),
    'v1+chi2': variant(select='chi2'),
    'v1+binary': variant(value='binary'),
    'v1+lexicon': variant(colour='lexicon'),
    'v1+aux': variant(aux='v2'),
    'v1+standard': variant(standard=True),
    'v3': V3,
    'v3+min_df3': {**V3, 'min_df': 3},
    'v3+standard': {**V3, 'standard': True},
    'v3-no_feeling': {**V3, 'drop': ('feeling',)},
    # Round two: the combinations the first round pointed at.
    'v1+aux+binary': variant(aux='v2', value='binary'),
    'v1+aux+binary+lexicon': variant(aux='v2', value='binary', colour='lexicon'),
    'v1+aux+caps60': variant(aux='v2', caps={'food': 60, 'colour': 20,
                                             'song': 60, 'feeling': 60}),
    'v1+aux-no_feeling': variant(aux='v2', drop=('feeling',)),
    'hybrid': variant(tokenizer='explicit', select='hybrid', value='binary',
                      colour='lexicon', aux='v2',
                      caps={'food': 50, 'colour': 20, 'song': 50, 'feeling': 60}),
    'hybrid-no_feeling': variant(tokenizer='explicit', select='hybrid', value='binary',
                                 colour='lexicon', aux='v2', drop=('feeling',),
                                 caps={'food': 50, 'colour': 20, 'song': 50,
                                       'feeling': 60}),
    # The candidate the two rounds point at: v1's coverage-first vocabulary, v2's aux
    # block, v2's fixed colour lexicon, and a small feeling block.
    'v3c': variant(tokenizer='explicit', select='tf', value='tfidf', colour='lexicon',
                   aux='v2', caps={'food': 50, 'colour': 20, 'song': 50,
                                   'feeling': 12}),
    'v3c-binary': variant(tokenizer='explicit', select='tf', value='binary',
                          colour='lexicon', aux='v2',
                          caps={'food': 50, 'colour': 20, 'song': 50, 'feeling': 12}),
    'v3c-no_feeling': variant(tokenizer='explicit', select='tf', value='tfidf',
                              colour='lexicon', aux='v2', drop=('feeling',),
                              caps={'food': 50, 'colour': 20, 'song': 50,
                                    'feeling': 12}),
}

HYBRID_HEAD = 0.8    # share of the budget given to the most frequent terms

REFERENCE = 'v1'


# ---------------------------------------------------------------- tokenizers and vocabularies
def regex_docs(texts):
    return [REGEX_TOKEN.findall(text) for text in texts]


def explicit_docs(texts, standard=False):
    out = []
    for text in texts:
        doc = v2text.tokens(text, **EXPLICIT_KW)
        if standard:
            doc = [v2text.standard_form(token) for token in doc]
        out.append(doc)
    return out


def chi_square(matrix, labels):
    """Chi-square per column against the 3-class label, vectorised over columns."""
    present = np.zeros((matrix.shape[1], len(CLASSES)))
    totals = np.zeros(len(CLASSES))
    for position, name in enumerate(CLASSES):
        mask = labels == name
        totals[position] = mask.sum()
        present[:, position] = matrix[mask].sum(axis=0)
    table = np.stack([totals[None, :] - present, present])          # (2, columns, 3)
    expected = table.sum(axis=2, keepdims=True) * totals[None, None, :] / len(labels)
    safe = np.where(expected > 0.0, expected, 1.0)
    scores = ((table - expected) ** 2 / safe).sum(axis=(0, 2))
    scores[~(expected > 0.0).all(axis=(0, 2))] = 0.0
    return scores


def select_terms(docs, labels, rule, cap, min_df):
    """The kept column names for one field, given this fold's fitting rows."""
    document, frequency = Counter(), Counter()
    for doc in docs:
        document.update(set(doc))
        frequency.update(doc)
    pool = [term for term, count in document.items() if count >= min_df]
    if not pool or cap <= 0:
        return []
    frequent = sorted(pool, key=lambda term: (-frequency[term], -document[term], term))
    if rule == 'tf':
        return frequent[:cap]
    if rule == 'hybrid':
        head = frequent[:max(1, int(round(cap * HYBRID_HEAD)))]
        tail = [term for term in frequent if term not in set(head)]
        return head + rank_by_chi2(tail, docs, labels, cap - len(head))
    if rule != 'chi2':
        raise ValueError(f'unknown selection rule {rule!r}')
    return rank_by_chi2(pool, docs, labels, cap)


def rank_by_chi2(pool, docs, labels, take):
    """The `take` highest-chi-square terms of `pool`, in descending order."""
    if not pool or take <= 0:
        return []
    position = {term: index for index, term in enumerate(pool)}
    matrix = np.zeros((len(docs), len(pool)), dtype=np.int8)
    for row, doc in enumerate(docs):
        for term in doc:
            index = position.get(term)
            if index is not None:
                matrix[row, index] = 1
    order = np.argsort(-chi_square(matrix, labels), kind='stable')
    return [pool[index] for index in order[:min(take, len(pool))]]


def weigh(docs, vocabulary, how):
    """Turn token lists into the field's term block, using the fitted vocabulary only."""
    if not vocabulary:
        return np.zeros((len(docs), 0))
    index = {term: position for position, term in enumerate(vocabulary)}
    counts = np.zeros((len(docs), len(vocabulary)))
    for row, doc in enumerate(docs):
        for term in doc:
            position = index.get(term)
            if position is not None:
                counts[row, position] += 1.0
    if how == 'binary':
        return (counts > 0.0).astype(float)
    if how != 'tfidf':
        raise ValueError(f'unknown weighting {how!r}')
    document = (counts > 0.0).sum(axis=0)
    idf = np.log((1.0 + len(docs)) / (1.0 + document)) + 1.0
    weights = np.where(counts > 0.0, 1.0 + np.log(np.maximum(counts, 1.0)), 0.0) * idf
    norm = np.linalg.norm(weights, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    return weights / norm


def colour_block(docs):
    rows = []
    for doc in docs:
        words = set(doc)
        rows.append([float(bool(words & set(survey.COLOUR_LEXICON[name])))
                     for name in COLOUR_NAMES])
    return np.array(rows, dtype=float), [f'colour_{name}' for name in COLOUR_NAMES]


def aux_block(texts, docs, hit, style, field):
    """One column per field for v1, three for v2."""
    if style == 'v1':
        column = np.array([[0.0] if text else [1.0] for text in texts])
        return column, [f'{field}_text_missing']
    if style != 'v2':
        raise ValueError(f'unknown aux style {style!r}')
    declined = np.array([1.0 if (not text or text in survey.NON_ANSWER) else 0.0
                         for text in texts])
    unseen = np.array([0.0 if flag else (0.0 if match else 1.0)
                       for flag, match in zip(declined, hit)])
    count = np.array([float(len(doc)) for doc in docs])
    return (np.column_stack([declined, unseen, count]),
            [f'{field}_non_answer', f'{field}_no_terms', f'{field}_item_count'])


# ---------------------------------------------------------------- fold cache
def fold_cache(fitting, held):
    """Everything that does not depend on the config decisions, fitted once per fold."""
    preprocessor = v2features.TextPreprocessor(text_fields=(), keep_structured=True)
    preprocessor.fit(fitting)
    keep = [column for column in preprocessor.columns_
            if column not in DROP_STRUCTURED]

    def structured(frame):
        return preprocessor._frame(frame, False)[keep]

    sides = {'fit': fitting, 'held': held}
    texts = {field: tuple(
        sides[side][survey.TEXT[field]].map(survey.clean).tolist()
        for side in ('fit', 'held')) for field in TEXT_FIELDS}
    docs = {}
    for field in TEXT_FIELDS:
        doc_pair = tuple(regex_docs(pair) for pair in texts[field])
        docs[('regex', False, field)] = doc_pair
        doc_pair = tuple(explicit_docs(pair) for pair in texts[field])
        docs[('explicit', False, field)] = doc_pair
        if field == 'feeling':
            docs[('explicit', True, field)] = tuple(
                explicit_docs(pair, standard=True) for pair in texts[field])
            docs[('regex', True, field)] = tuple(
                [[v2text.standard_form(token) for token in doc] for doc in side]
                for side in docs[('regex', False, field)])
    return dict(structured_fit=structured(fitting), structured_held=structured(held),
                texts=texts, docs=docs)


def build(cache, config, labels):
    """Assemble the full matrix for one config, using this fold's fitted state only."""
    blocks_fit = [cache['structured_fit'].to_numpy(dtype=float)]
    blocks_held = [cache['structured_held'].to_numpy(dtype=float)]
    names = list(cache['structured_fit'].columns)
    dropped = set(config['drop'])

    for field in TEXT_FIELDS:
        if field in dropped:
            continue
        texts_fit, texts_held = cache['texts'][field]
        key = (config['tokenizer'], bool(config['standard']) and field == 'feeling', field)
        docs_fit, docs_held = cache['docs'][key]

        if field == 'colour' and config['colour'] == 'lexicon':
            block_fit, columns = colour_block(docs_fit)
            block_held, _ = colour_block(docs_held)
            hit_fit = [bool(set(doc) & survey.COLOUR_WORDS) for doc in docs_fit]
            hit_held = [bool(set(doc) & survey.COLOUR_WORDS) for doc in docs_held]
        else:
            vocabulary = select_terms(docs_fit, labels, config['select'],
                                      config['caps'][field], config['min_df'])
            block_fit = weigh(docs_fit, vocabulary, config['value'])
            block_held = weigh(docs_held, vocabulary, config['value'])
            columns = [f'{field}_{term}' for term in vocabulary]
            kept = set(vocabulary)
            hit_fit = [bool(set(doc) & kept) for doc in docs_fit]
            hit_held = [bool(set(doc) & kept) for doc in docs_held]

        aux_fit, aux_columns = aux_block(texts_fit, docs_fit, hit_fit,
                                         config['aux'], field)
        aux_held, _ = aux_block(texts_held, docs_held, hit_held, config['aux'], field)
        blocks_fit.append(np.hstack([block_fit, aux_fit]))
        blocks_held.append(np.hstack([block_held, aux_held]))
        names.extend(columns + aux_columns)

    return np.hstack(blocks_fit), np.hstack(blocks_held), names


# ---------------------------------------------------------------- tokenizer report
def tokenizer_report(train):
    """How many answers and tokens each tokenizer sees, per field."""
    rows = []
    examples = []
    for field in TEXT_FIELDS:
        texts = train[survey.TEXT[field]].map(survey.clean).tolist()
        for name, maker in (('regex', regex_docs), ('explicit', explicit_docs)):
            docs = maker(texts)
            sizes = np.array([len(doc) for doc in docs])
            document = Counter()
            for doc in docs:
                document.update(set(doc))
            rows.append({
                'field': field, 'tokenizer': name,
                'tokens': int(sizes.sum()),
                'no_tokens': int((sizes == 0).sum()),
                'distinct': len(document),
                'df_ge_2': sum(1 for count in document.values() if count >= 2),
                'df_ge_3': sum(1 for count in document.values() if count >= 3),
            })
        lost = [(text, regex_docs([text])[0], explicit_docs([text])[0])
                for text in texts
                if text and not regex_docs([text])[0] and explicit_docs([text])[0]]
        examples.extend([(field, *item) for item in lost[:4]])
    print('tokenizer coverage on the 606 training rows')
    print(pd.DataFrame(rows).to_string(index=False))
    if examples:
        print(f'\nanswers the regex tokenizer drops completely ({len(examples)} shown)')
        for field, text, _, got in examples:
            print(f'  {field:8s} {text!r:34s} -> {got}')
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main():
    train = pd.read_csv(TRAIN_CSV, dtype=str, keep_default_na=False)
    for column in ('id', 'label'):
        if column not in train.columns:
            raise ValueError(f'{TRAIN_CSV} is missing {column!r}')
    print(f'train split: {len(train)} rows, {train["id"].nunique()} respondents, '
          f'{len(TEXT_FIELDS)} text fields, Q4 removed\n')
    tokenizer_report(train)
    print()

    folds = []
    for fold_seed in selection.FOLD_SEEDS:
        for fold, (fit_index, held_index) in enumerate(
                selection.folds_of(train, fold_seed)):
            fitting, held = train.iloc[fit_index], train.iloc[held_index]
            if set(fitting['id']) & set(held['id']):
                raise AssertionError(f'fold {fold}: a respondent appears twice')
            folds.append(dict(fold=fold, fold_seed=fold_seed, fitting=fitting, held=held,
                              **fold_cache(fitting, held)))
    print(f'{len(folds)} grouped folds prepared '
          f'({len(selection.FOLD_SEEDS)} seeds x {selection.N_SPLITS} splits)\n')

    records = []
    clock = time.perf_counter()
    wanted = [name for name in sys.argv[1:] if name in CONFIGS]
    configs = {name: config for name, config in CONFIGS.items()
               if not wanted or name in wanted or name == REFERENCE}
    for name, config in configs.items():
        for setting_name, params in FOREST_SETTINGS.items():
            for data in folds:
                y_fit = data['fitting']['label'].to_numpy()
                y_held = data['held']['label'].to_numpy()
                X_fit, X_held, columns = build(data, config, y_fit)
                structured = len(data['structured_fit'].columns)
                model = selection.make_forest(params, 311)
                model.fit(X_fit, y_fit)
                prediction = model.predict(X_held)
                records.append({
                    'config': name, 'setting': setting_name,
                    'fold_seed': data['fold_seed'], 'fold': data['fold'],
                    'accuracy': float(accuracy_score(y_held, prediction)),
                    'macro_f1': float(f1_score(y_held, prediction, labels=CLASSES,
                                               average='macro', zero_division=0)),
                    'train_accuracy': float(accuracy_score(y_fit, model.predict(X_fit))),
                    'n_features': len(columns),
                    'text_features': len(columns) - structured,
                })
        print(f'  scored {name} ({time.perf_counter() - clock:.0f}s)', flush=True)

    frame = pd.DataFrame(records)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_DIR / 'text_variants_folds.csv', index=False)

    summary = frame.groupby(['config', 'setting'], sort=False).agg(
        accuracy=('accuracy', 'mean'), accuracy_sd=('accuracy', 'std'),
        macro_f1=('macro_f1', 'mean'), train_accuracy=('train_accuracy', 'mean'),
        n_features=('n_features', 'max'),
    ).reset_index()

    # Paired, fold by fold, against the reference config in the same forest setting.
    for setting_name in FOREST_SETTINGS:
        print(f'\n=== forest setting: {setting_name} '
              f'{FOREST_SETTINGS[setting_name]} ===')
        block = frame[frame.setting == setting_name]
        base = (block[block.config == REFERENCE]
                .sort_values(['fold_seed', 'fold'])
                .set_index(['fold_seed', 'fold'])['accuracy'])
        rows = []
        for name in configs:
            here = (block[block.config == name].sort_values(['fold_seed', 'fold'])
                    .set_index(['fold_seed', 'fold'])['accuracy'])
            delta = here - base
            rows.append({
                'config': name,
                'features': int(block[block.config == name]['n_features'].max()),
                'accuracy': here.mean(),
                'sd': here.std(),
                'macro_f1': float(block[block.config == name]['macro_f1'].mean()),
                'train_acc': float(block[block.config == name]['train_accuracy'].mean()),
                'vs_v1': float(delta.mean()),
                'better': int((delta > 1e-9).sum()),
                'tied': int((delta.abs() <= 1e-9).sum()),
                'worse': int((delta < -1e-9).sum()),
            })
        table = pd.DataFrame(rows)
        print(table.to_string(index=False,
                              formatters={'accuracy': '{:.4f}'.format,
                                          'sd': '{:.4f}'.format,
                                          'macro_f1': '{:.4f}'.format,
                                          'train_acc': '{:.4f}'.format,
                                          'vs_v1': '{:+.4f}'.format}))
        table.to_csv(OUT_DIR / f'text_variants_{setting_name}.csv', index=False)

    summary.to_csv(OUT_DIR / 'text_variants_summary.csv', index=False)
    print(f'\nwrote {OUT_DIR}')


if __name__ == '__main__':
    main()

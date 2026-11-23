"""Interface smoke check for train/random_forest. Fits no real model, writes nothing.

1) the ablation preprocessor must drop exactly the expected question blocks
2) one tiny 5-tree fit on one fold must run end to end through selection.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path('train/random_forest').resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'data_preprocessing'))

import config
import features
import selection

train = pd.read_csv(config.TRAIN_CSV, dtype=str, keep_default_na=False)
print(f'train records = {len(train)}  respondents = {train.id.nunique()}')

print('\n--- 1. ablation preprocessor: column counts on the full training split ---')
prev = None
for name, text_fields in config.TEXT_VARIANTS.items():
    pre = features.AblationPreprocessor(text_fields, keep_structured=True).fit(train)
    block = {field: sum(c.startswith(f'{field}_') for c in pre.columns_)
             for field in features.TEXT}
    print(f'  {name:20s} features={len(pre.columns_):4d}  per-field columns={block}')
    if prev is not None and len(pre.columns_) > prev:
        raise AssertionError(f'{name} has more columns than the variant before it')
    prev = len(pre.columns_)

print('\n--- 1b. dropping structured questions (text only) ---')
pre = features.AblationPreprocessor(features.TEXT, keep_structured=False).fit(train)
leftover = [c for c in pre.columns_
            if c in features.STRUCTURED_COLUMNS or c in features.STRUCTURED_FLAGS
            or c in features.PAYMENT_COLUMNS]
print(f'  text_only features={len(pre.columns_)}  structured leftovers={leftover}')
assert not leftover

print('\n--- 1c. transform on the validation split must reuse the training columns ---')
val = pd.read_csv(config.VAL_CSV, dtype=str, keep_default_na=False)
pre = features.AblationPreprocessor(('food', 'colour'), keep_structured=True).fit(train)
X = pre.transform(val, scaled=False)
print(f'  validation matrix shape={X.shape}  finite={bool((X == X).all())}')

print('\n--- 1d. dropping a field only removes columns, it never changes the others ---')
full = features.AblationPreprocessor(config.TEXT_FIELDS, True).fit(train)
full_matrix = full.transform(val, scaled=False)
for name in ('no_song', 'no_feeling', 'no_song_no_feeling', 'food_only'):
    part = features.AblationPreprocessor(config.TEXT_VARIANTS[name], True).fit(train)
    extra = [c for c in part.columns_ if c not in full.columns_]
    subset = full_matrix[:, [full.columns_.index(c) for c in part.columns_]]
    same = np.array_equal(part.transform(val, scaled=False), subset)
    print(f'  {name:20s} features={len(part.columns_):4d} '
          f'unexpected_extra={len(extra)} identical_values={same}')
    assert not extra and same, f'{name} is not a clean column subset of the full encoding'

print('\n--- 2. one 5-tree fit through selection.py (interface only) ---')
prepared = selection.prepare_folds(train, ('food', 'colour'), True,
                                   config.FOLD_SEEDS[0], config.N_SPLITS)
print(f'  folds={len(prepared)}  features per fold={[d["n_features"] for d in prepared]}')
params = {'max_depth': 4, 'min_samples_leaf': 5, 'max_features': 'sqrt'}
records = selection.evaluate_config(prepared, params, n_estimators=5)
frame = pd.DataFrame(records)
print(frame[['fold', 'accuracy', 'macro_f1', 'log_loss', 'n_features']].round(4).to_string(index=False))
assert frame.log_loss.between(0, 5).all(), 'log loss out of range: classes may be misaligned'

print('\n--- 3. selection helpers ---')
summary = selection.summarize(records)
print(selection.simplicity({'max_depth': None, 'min_samples_leaf': 1, 'max_features': 'sqrt'}))
print(selection.one_se_select(summary, metric='accuracy', basis='se')['chosen_params'])
band = selection.grouped_bootstrap(val.label.to_numpy(),
                                   ['Soccer'] * len(val), val.id.to_numpy(), n_boot=200)
print('bootstrap on a constant predictor:', {k: round(v['low'], 3) for k, v in band.items()})
print('\nPASS: preprocessing blocks drop correctly and the CV plumbing runs.')

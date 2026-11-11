"""Export the frozen v1 random forest into files that `pred.py` can load without scikit-learn.

Reads `random_forest_deliverables/forest_final_v1/final_model.joblib` and writes two files
next to this script:

    model_data.npz        the trees as plain arrays, the IDF vectors, the imputation values
    preprocessing.json    column order, TF-IDF vocabularies, column names, class order

Nothing is refitted. Every number written here is read out of the artifact, so `pred.py`
scores exactly the model that was evaluated once on the test split.

Finally the export is checked by re-predicting the training rows through the plain-array
path and comparing against the artifact itself.

Run from this directory:  python export_model.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
ARTIFACT = ROOT / 'random_forest_deliverables' / 'forest_final_v1' / 'final_model.joblib'
TRAIN_CSV = ROOT / 'data' / 'splits' / 'train.csv'

# The saved preprocessor is a class defined in these modules, so they must be importable
# before joblib can unpickle it. Nothing from them is used for prediction afterwards.
for path in (ROOT / 'train' / 'random_forest', ROOT / 'analysis', ROOT / 'data_preprocessing'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import joblib                                                        # noqa: E402

TEXT_FIELDS = ('food', 'colour', 'song', 'feeling')


def flatten_forest(model):
    """Concatenate every tree into flat arrays plus the offset of each tree."""
    left, right, feature, threshold, value = [], [], [], [], []
    offsets = [0]
    for estimator in model.estimators_:
        tree = estimator.tree_
        counts = tree.value[:, 0, :].astype(np.float64)
        totals = counts.sum(axis=1, keepdims=True)
        left.append(tree.children_left.astype(np.int32))
        right.append(tree.children_right.astype(np.int32))
        feature.append(tree.feature.astype(np.int32))
        threshold.append(tree.threshold.astype(np.float64))
        value.append(counts / np.where(totals > 0.0, totals, 1.0))
        offsets.append(offsets[-1] + len(tree.children_left))
    return {
        'left': np.concatenate(left),
        'right': np.concatenate(right),
        'feature': np.concatenate(feature),
        'threshold': np.concatenate(threshold),
        'value': np.concatenate(value).astype(np.float64),
        'offsets': np.asarray(offsets, dtype=np.int32),
    }


def main():
    artifact = joblib.load(ARTIFACT)
    preprocessor, model = artifact['preprocessor'], artifact['model']
    text_encoder = preprocessor.encoder_['text_encoder']
    vectorizers = text_encoder['vectorizers']

    forest = flatten_forest(model)
    arrays = dict(forest)
    # Only numeric arrays go in the archive: fixed-width string dtypes would truncate the
    # long question headers. Column names, class order and vocabularies go to the JSON.
    arrays['fill'] = np.asarray([preprocessor.fill_[name] for name in preprocessor.columns_],
                                dtype=np.float64)
    for field in TEXT_FIELDS:
        arrays[f'idf_{field}'] = vectorizers[field].idf_.astype(np.float64)
    np.savez_compressed(HERE / 'model_data.npz', **arrays)

    payload = {
        'model': 'random forest, 300 trees, max_depth 6, min_samples_leaf 3, max_features log2',
        'source_artifact': str(ARTIFACT.relative_to(ROOT)).replace('\\', '/'),
        'n_features': len(preprocessor.columns_),
        'columns': list(preprocessor.columns_),
        'classes': list(model.classes_),
        'payment_fill_value': float(text_encoder['payment_fill_value']),
        'text_terms': {field: list(vectorizers[field].get_feature_names_out())
                       for field in TEXT_FIELDS},
    }
    (HERE / 'preprocessing.json').write_text(json.dumps(payload, indent=1), encoding='utf-8')

    print(f'exported {len(model.estimators_)} trees, {len(forest["left"])} nodes, '
          f'{len(preprocessor.columns_)} features')
    print(f'  model_data.npz      {(HERE / "model_data.npz").stat().st_size:>10,} bytes')
    print(f'  preprocessing.json  {(HERE / "preprocessing.json").stat().st_size:>10,} bytes')
    for field in TEXT_FIELDS:
        print(f'    {field:8s} {len(payload["text_terms"][field]):3d} terms')

    # Check the export against the artifact on rows the model was fitted on. This is an
    # implementation check, not an evaluation: the labels are only used to build the
    # comparison, never to fit anything.
    import pred
    train = pd.read_csv(TRAIN_CSV, dtype=str, keep_default_na=False)
    reference = model.predict(preprocessor.transform(train, scaled=False))
    exported = np.asarray(pred.predict_all(TRAIN_CSV))
    disagree = int((reference != exported).sum())
    print(f'\nexport check on {len(train)} training rows: '
          f'{disagree} disagreement(s) against the artifact')
    if disagree:
        raise SystemExit('the exported arrays do not reproduce the artifact')
    print('export verified')


if __name__ == '__main__':
    main()

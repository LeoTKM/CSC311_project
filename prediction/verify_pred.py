"""Check that `pred.py` reproduces the pipeline it replaces, including on messy input.

Three checks, none of which reads the test split:

    1  predictions on data/splits/train.csv and data/splits/val.csv against the artifact
    2  the full 208-column matrix, element by element, on those same rows
    3  the same two comparisons on a synthetic sheet of deliberately awkward answers -
       invalid scale values, unknown multi-select options, refusals, currency words,
       ranges, "1k", thousands separators, hyphens, apostrophes and digits in the text

Checks 2 and 3 matter more than 1: identical predictions can hide a feature that differs
without crossing a split threshold. If the matrices agree to floating-point noise, the port
is the pipeline.

Run from this directory:  python verify_pred.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
ARTIFACT = ROOT / 'random_forest_deliverables' / 'forest_final_v1' / 'final_model.joblib'

for path in (ROOT / 'train' / 'random_forest', ROOT / 'analysis', ROOT / 'data_preprocessing'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import joblib                                                        # noqa: E402
import pred                                                          # noqa: E402

TOLERANCE = 1e-9

# Each row is (ease, Q2, Q3, Q4, venue, season, music, food, colour, song, feeling, payment).
AWKWARD = [
    ('5', 'agree', 'neutral', 'disagree', 'Indoor facility', 'Winter',
     'Latin', 'pizza and beer', 'green', 'waka waka', 'excited', '100'),
    ('', '', '', '', '', '', '', '', '', '', '', ''),
    ('99', 'agree', 'agree', 'agree', 'Indoor facility', 'Summer', 'Pop',
     'burger', 'red', 'we will rock you', 'happy', '50-100'),
    ('abc', 'strongly agree', 'disagree', 'neutral', 'Others', 'Fall',
     'Hip-hop or rap', 'GREEN', 'BLUE', "don't stop me now", 'EXCITED', '1k'),
    ('-2', 'disagree', 'neutral', 'agree', 'Flying carpet', 'Monsoon', 'Polka',
     'hot-dog and 50 cent soda', 'grey', 'n/a', 'bored', 'CAD 200'),
    ('7.5', 'neutral', 'agree', 'agree', '', '', '',
     '', '', '', '', '$150'),
    ('0', 'strongly disagree', 'strongly disagree', 'disagree',
     'Hip-hop or rap, Pop', 'Winter, Summer', 'Rock, Country',
     'ramen', 'black', 'take me out to the ball game', 'nervous', 'none'),
    ('10', 'agree', 'neutral', 'strongly agree', 'No particular type', 'Spring',
     'Classical or jazz', 'n/a', 'yellow', '', 'relaxed', 'hundreds'),
    ('3', 'Agree', 'AGREE', 'Agree', 'indoor facility', 'winter', 'latin',
     'Pizza, Beer', 'Greenish', 'Waka', 'Excited!', '1,000'),
    ('6', 'somewhat agree', 'neutral', 'agree', 'At home or nearby', 'Summer',
     'Other', 'salad', 'orange', 'song with 2 numbers 1999', 'pumped', 'between 100 and 200'),
    ('6', 'agree', 'neutral', 'agree', 'School or university facility', 'Fall',
     'Electronic or dance', 'water', 'white', 'classical piece', 'calm', '200, 500'),
    ('  ', '  agree  ', ' neutral ', ' disagree ', 'Indoor facility , Others ', ' Winter ',
     ' Folk or traditional ', '  Two  Spaces  ', ' teal ', ' instrumental ', ' okay ', ' 75 '),
]


def known_frame(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')


def awkward_frame():
    columns = [pred.EASE_COLUMN, *pred.AGREEMENT_COLUMNS, *pred.SCHEMA, *pred.TEXT.values(),
               pred.PAYMENT]
    frame = pd.DataFrame(AWKWARD, columns=columns)
    frame.insert(0, 'id', [f'probe{index}' for index in range(len(frame))])
    return frame


def compare(label, frame, preprocessor, model, arrays, spec):
    reference_matrix = preprocessor.transform(frame, scaled=False)
    reference_predictions = model.predict(reference_matrix)

    exported_matrix = pred.build_features(pred.sanitize(frame), spec, arrays)
    exported_predictions = pred.traverse(arrays, exported_matrix, spec['classes'])

    if reference_matrix.shape != exported_matrix.shape:
        raise SystemExit(f'{label}: shape {exported_matrix.shape} vs {reference_matrix.shape}')
    gap = float(np.abs(reference_matrix - exported_matrix).max())
    bad_predictions = int((reference_predictions != exported_predictions).sum())
    rows, columns = exported_matrix.shape
    verdict = 'ok' if (gap <= TOLERANCE and bad_predictions == 0) else 'MISMATCH'
    print(f'  {label:26s} {rows:>4d} rows x {columns:>3d} cols   '
          f'max |diff| {gap:.2e}   prediction disagreements {bad_predictions}   {verdict}')
    return gap <= TOLERANCE and bad_predictions == 0


def main():
    artifact = joblib.load(ARTIFACT)
    preprocessor, model = artifact['preprocessor'], artifact['model']
    arrays, spec = pred.load()

    print(f'artifact   {len(spec["columns"])} columns, {len(model.estimators_)} trees')
    print(f'export     {len(arrays["left"])} nodes across '
          f'{len(arrays["offsets"]) - 1} trees\n')

    ok = True
    for name in ('train', 'val'):
        frame = known_frame(ROOT / 'data' / 'splits' / f'{name}.csv')
        ok &= compare(f'{name}.csv', frame, preprocessor, model, arrays, spec)
    awkward = awkward_frame()
    ok &= compare('synthetic edge cases', awkward, preprocessor, model, arrays, spec)

    # Short column names, as used by the earlier encoded files, must behave identically.
    short = awkward.rename(columns=dict(zip(pred.FULL_COLUMNS, pred.SHORT_COLUMNS)))
    same = pred.sanitize(short).equals(pred.sanitize(awkward))
    print(f'\n  short headers renamed to the long ones: {"ok" if same else "MISMATCH"}')
    ok &= bool(same)

    # A blank label column and a missing one must both be accepted.
    for label, drop in (('with label column', False), ('without label column', True)):
        frame = awkward.copy()
        if drop:
            frame = frame.drop(columns=['id'])
        else:
            frame.insert(1, 'label', 'Soccer')
        try:
            labels = pred.predict_frame(frame)
            print(f'  {label:26s} {len(labels)} predictions, first {labels[0]}')
        except Exception as error:                                  # noqa: BLE001
            print(f'  {label:26s} FAILED: {error}')
            ok = False

    # Timing at the size the course states: about 60 predictions within a minute.
    val = ROOT / 'data' / 'splits' / 'val.csv'
    started = time.perf_counter()
    for _ in range(5):
        pred.predict_all(val)
    elapsed = (time.perf_counter() - started) / 5
    print(f'\n  132 rows in {elapsed * 1e3:.0f} ms '
          f'({elapsed / 132 * 1e3:.2f} ms/row; 60 rows in '
          f'{elapsed / 132 * 60 * 1e3:.0f} ms, budget 60 s)')

    print(f'\n{"PASS" if ok else "FAIL"}: the port matches the artifact')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())

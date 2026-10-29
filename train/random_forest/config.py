"""Configuration for the random-forest training pipeline.

Training may use scikit-learn; only the eventual inference script must be
dependency-free. Nothing here is imported by the inference side.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = ROOT / 'data' / 'splits' / 'train.csv'
VAL_CSV = ROOT / 'data' / 'splits' / 'val.csv'
TEST_CSV = ROOT / 'data' / 'splits' / 'test.csv'
DEFAULT_OUTPUT = ROOT / 'analysis' / 'out' / 'forest'

CLASSES = ['Soccer', 'Baseball', 'Basketball']

# Fixed forest settings. n_estimators is checked separately with a convergence
# curve; class_weight stays None because the three classes are exactly balanced
# (202 responses each) and OOB scores are not used because a row-level bootstrap
# does not keep one respondent's three responses together.
FIXED = dict(n_estimators=300, criterion='gini', bootstrap=True,
             class_weight=None, oob_score=False, n_jobs=1)

# Searched axes: 6 x 5 x 5 = 150 combinations.
# Compared with the earlier comparison run (model_experiments.py, 24 combinations
# over max_depth 4/8/12/None, leaf 1/3/5, max_features sqrt/0.5) this adds the
# log2 and 0.3 settings, the 1.0 setting, leaf 2 and 8, and depth 6 and 16.
GRID = {
    'max_depth': [4, 6, 8, 12, 16, None],
    'min_samples_leaf': [1, 2, 3, 5, 8],
    'max_features': ['sqrt', 'log2', 0.3, 0.5, 1.0],
}

# Reduced grid for the text-field ablation, so that comparing six field sets stays
# cheap: 2 x 2 x 1 = 4 combinations per field set.
ABLATION_GRID = {
    'max_depth': [4, None],
    'min_samples_leaf': [1, 5],
    'max_features': ['sqrt'],
}

TEXT_FIELDS = ('food', 'colour', 'song', 'feeling')

# 'none' keeps every non-text feature (Q1-Q7, payment) and drops all four blocks.
TEXT_VARIANTS = {
    'all': TEXT_FIELDS,
    'no_song': ('food', 'colour', 'feeling'),
    'no_feeling': ('food', 'colour', 'song'),
    'no_song_no_feeling': ('food', 'colour'),
    'food_only': ('food',),
    'none': (),
}

FOLD_SEEDS = (311, 312, 313)
ABLATION_FOLD_SEEDS = (311,)
N_SPLITS = 5

SELECTION_NOTE = (
    'Primary metric on grouped repeated CV, then a one-standard-error band, then '
    'the simplest configuration: shallower depth, then larger leaves, then fewer '
    'candidate split features.'
)

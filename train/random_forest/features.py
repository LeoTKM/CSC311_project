"""Fold-local preprocessing with an optional question-block ablation.

The pipeline of ``analysis/model_experiments.py`` is reused as-is; this module only
restricts which question blocks survive. Everything is still fitted on the fitting
rows of one fold and merely transformed on the held-out rows.
"""
import sys
from pathlib import Path

from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / 'analysis', ROOT / 'data_preprocessing'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from encode_all import (AGREEMENT_COLUMNS, EASE_COLUMN, SCHEMA, TEXT)  # noqa: E402
from model_experiments import SurveyPreprocessor                        # noqa: E402

# Structured columns produced by encode_all for Q1-Q4 and the 21 multi-hot options.
STRUCTURED_COLUMNS = [EASE_COLUMN, *AGREEMENT_COLUMNS,
                      *[name for choices in SCHEMA.values() for _, name in choices]]
STRUCTURED_FLAGS = [f'q{i}_missing' for i in range(1, 8)]
PAYMENT_COLUMNS = ['payment_amount', 'payment_missing']


class AblationPreprocessor(SurveyPreprocessor):
    """A SurveyPreprocessor restricted to a subset of the survey questions.

    A dropped text field is blanked before encoding, so its fitted vocabulary stays
    empty, no TF-IDF block is produced and only its constant missing flag would
    remain - that flag is dropped too. Dropped structured questions and the payment
    column are removed after encoding.
    """

    def __init__(self, text_fields, keep_structured=True):
        super().__init__()
        unknown = set(text_fields) - set(TEXT)
        if unknown:
            raise ValueError(f'unknown text field(s): {sorted(unknown)}')
        keep = set(text_fields)
        self.text_fields = tuple(field for field in TEXT if field in keep)
        self.keep_structured = bool(keep_structured)
        self.dropped_text = tuple(field for field in TEXT if field not in keep)

    def _frame(self, raw, fitting):
        raw = raw.copy()
        for field in self.dropped_text:
            raw[TEXT[field]] = ''
        features = super()._frame(raw, fitting)

        drop = [column for field in self.dropped_text
                for column in features.columns if column.startswith(f'{field}_')]
        if not self.keep_structured:
            drop += [column for column in features.columns
                     if column in STRUCTURED_COLUMNS or column in STRUCTURED_FLAGS
                     or column in PAYMENT_COLUMNS]
        if drop:
            features = features.drop(columns=drop)

        if fitting:
            self.columns_ = list(features.columns)
        elif list(features.columns) != list(self.columns_):
            # The parent already reindexed; this is only a guard.
            features = features.reindex(columns=self.columns_)
        return features

    def fit(self, raw):
        features = self._frame(raw, True)
        self.scale_cols_ = [column for column in
                            (EASE_COLUMN, *AGREEMENT_COLUMNS, 'payment_amount')
                            if column in features.columns]
        self.scaler_ = (StandardScaler().fit(features[self.scale_cols_])
                        if self.scale_cols_ else None)
        return self

    def transform(self, raw, scaled=True):
        if scaled:
            raise ValueError('the forest never standardises; call transform(raw, scaled=False)')
        return self._frame(raw, False).to_numpy(dtype=float)

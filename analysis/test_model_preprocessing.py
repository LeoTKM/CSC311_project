"""Checks against metadata leakage and held-out fitting in the modelling pipeline."""
import pickle
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from model_experiments import SurveyPreprocessor, ROOT
from encode_all import EASE_COLUMN, SCHEMA, TEXT, PAYMENT, parse_payment

class PreprocessingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=pd.read_csv(ROOT/'data/splits/train.csv',dtype=str,keep_default_na=False).iloc[:90].copy()
        cls.pre=SurveyPreprocessor().fit(cls.raw)

    def test_metadata_never_changes_inputs(self):
        changed=self.raw.copy()
        changed['id']='different person'
        changed['label']='not a sport'
        np.testing.assert_array_equal(self.pre.transform(self.raw),self.pre.transform(changed))
        self.assertNotIn('id',self.pre.columns_)
        self.assertNotIn('label',self.pre.columns_)

    def test_transform_does_not_fit_on_new_values(self):
        changed=self.raw.iloc[:3].copy()
        changed[TEXT['food']]='zzheldoutonlyword zzheldoutonlyword'
        changed[PAYMENT]='99999999'
        before=pickle.dumps(self.pre)
        self.pre.transform(changed)
        self.assertEqual(before,pickle.dumps(self.pre))
        self.assertNotIn('zzheldoutonlyword',self.pre.encoder_['text_encoder']['vectorizers']['food'].vocabulary_)
        observed=pd.Series([parse_payment(x)[0] for x in self.raw[PAYMENT]],dtype=float)
        self.assertAlmostEqual(observed.mean(),self.pre.encoder_['text_encoder']['payment_fill_value'])

    def test_unknown_and_missing_structured_values_are_finite(self):
        changed=self.raw.iloc[:2].copy()
        changed[EASE_COLUMN]='999'
        changed[next(iter(SCHEMA))]='unknown venue'
        result=self.pre.transform(changed)
        self.assertTrue(np.isfinite(result).all())
        for flag in ('q1_missing','q5_missing'):
            np.testing.assert_array_equal(result[:,self.pre.columns_.index(flag)],np.ones(2))

if __name__=='__main__': unittest.main()

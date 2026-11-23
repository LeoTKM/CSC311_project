"""Regression checks for the two split entry points and committed partition."""
from pathlib import Path
import sys
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'data_preprocessing'))
from make_splits import build_manifest
from encode_all import grouped_split


class SplitConsistency(unittest.TestCase):
    def test_entry_points_agree(self):
        # Numeric-looking strings catch lexical vs numeric sorting divergence;
        # several group sizes also catch rounding/allocation differences.
        for size in (3, 10, 17, 29, 290):
            for seed in (7, 311):
                ids = [str(i) for i in range(size)]
                expected = build_manifest(ids, seed, .15, .15)
                parts = grouped_split(pd.DataFrame({'id': ids}), seed)
                actual = {i: ('val' if split == 'validation' else split)
                          for split, frame in parts.items() for i in frame.id}
                self.assertEqual(actual, expected)

    def test_committed_manifest_matches(self):
        manifest = pd.read_csv(ROOT / 'data/splits/id_to_split.csv', dtype=str)
        parts = grouped_split(manifest[['id']], 311)
        actual = {i: ('val' if split == 'validation' else split)
                  for split, frame in parts.items() for i in frame.id}
        self.assertEqual(actual, dict(zip(manifest.id, manifest.split)))


if __name__ == '__main__':
    unittest.main()

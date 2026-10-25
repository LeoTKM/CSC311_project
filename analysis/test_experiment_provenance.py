"""Keep newline conversion harmless without accepting changed experiments."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from model_experiments import run, text_sha256


class ProvenanceChecks(unittest.TestCase):
    def test_line_endings_do_not_change_text_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.py'
            path.write_bytes(b'value = 1\nprint(value)\n')
            expected = text_sha256(path)
            path.write_bytes(b'value = 1\r\nprint(value)\r\n')
            self.assertEqual(text_sha256(path), expected)
            path.write_bytes(b'value = 2\r\nprint(value)\r\n')
            self.assertNotEqual(text_sha256(path), expected)

    def test_cache_rejects_real_source_or_runtime_changes(self):
        recorded = {'hash_format': 'sha256-lf-v1',
                    'sha256': {'source.py': 'original'}, 'python': '3.12.14'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'run_manifest.json').write_text(json.dumps(
                {'status': 'complete', 'provenance': recorded}))
            for changed in (
                {**recorded, 'sha256': {'source.py': 'changed'}},
                {**recorded, 'python': '3.12.15'},
                {key: value for key, value in recorded.items() if key != 'hash_format'},
            ):
                with self.subTest(changed=changed), patch(
                        'model_experiments.provenance', return_value=changed):
                    with self.assertRaisesRegex(RuntimeError, 'different source/data/runtime'):
                        run(path)

    def test_matching_complete_cache_does_not_refit(self):
        recorded = {'hash_format': 'sha256-lf-v1', 'sha256': {'source.py': 'same'}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'run_manifest.json').write_text(json.dumps(
                {'status': 'complete', 'provenance': recorded}))
            with patch('model_experiments.provenance', return_value=recorded), \
                    patch('model_experiments.pd.read_csv', side_effect=AssertionError('refit')):
                self.assertEqual(run(path), path)


if __name__ == '__main__':
    unittest.main()

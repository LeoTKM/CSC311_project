"""Submission contains code, preserves the PDF, and excludes datasets/models."""
from pathlib import Path
import tempfile
import unittest
import zipfile

from package_proposal import package


class PackagingChecks(unittest.TestCase):
    def test_archive_scope_and_repeatability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'repo'
            files = {
                'analysis/report/proposal.pdf': b'%PDF-1.7\nexample',
                'analysis/example.py': b'print("analysis")\n',
                'analysis/example.ipynb': b'{"cells": []}',
                'analysis/requirements.txt': b'numpy\n',
                'data/raw.csv': b'secret data',
                'analysis/out/model.joblib': b'model',
                'analysis/out/generated.py': b'not source',
                '.venv/private.py': b'not source',
            }
            for name, data in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            first = package(root, Path(directory) / 'first')
            second = package(root, Path(directory) / 'second')
            self.assertEqual(first[0].read_bytes(), files['analysis/report/proposal.pdf'])
            self.assertEqual(first[1].read_bytes(), second[1].read_bytes())
            with zipfile.ZipFile(first[1]) as archive:
                self.assertEqual(set(archive.namelist()), {
                    'analysis/example.py', 'analysis/example.ipynb', 'analysis/requirements.txt'})
                self.assertIsNone(archive.testzip())
            with self.assertRaises(FileExistsError):
                package(root, Path(directory) / 'first')


if __name__ == '__main__':
    unittest.main()

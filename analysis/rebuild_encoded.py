"""Rebuild encodings from canonical raw splits; never generate a new partition."""
from pathlib import Path
import os
import shutil
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'data_preprocessing'))
from encode_all import process_splits, read_csv, sha256


def rebuild():
    paths = {'train': ROOT / 'data/splits/train.csv',
             'validation': ROOT / 'data/splits/val.csv',
             'test': ROOT / 'data/splits/test.csv'}
    parts = {name: read_csv(path) for name, path in paths.items()}
    provenance = {'mode': 'existing_splits', 'files': {
        name: {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha256(path)}
        for name, path in paths.items()}}
    destination = ROOT / 'data/encoded'
    destination.mkdir(exist_ok=True)
    # Build successfully before replacing any existing generated artifacts.
    with tempfile.TemporaryDirectory(prefix='encoding-', dir=destination.parent) as temp:
        staging = Path(temp)
        process_splits(parts, staging, provenance)
        for name, raw in parts.items():
            encoded = read_csv(staging / f'{name}_encoded.csv')
            assert encoded[['id', 'label']].equals(raw[['id', 'label']])
        for file in staging.iterdir():
            # Moving directly from TemporaryDirectory preserves its private
            # Windows ACL, potentially blocking the user's notebook kernel.
            # Create a new file in the destination to inherit its normal ACL.
            replacement = destination / f'.{file.name}.{uuid.uuid4().hex}.tmp'
            try:
                with file.open('rb') as source, replacement.open('xb') as target:
                    shutil.copyfileobj(source, target)
                os.replace(replacement, destination / file.name)
            finally:
                replacement.unlink(missing_ok=True)
    print('Verified: every encoded row preserves its canonical raw split, ID, label and order.')


if __name__ == '__main__':
    rebuild()

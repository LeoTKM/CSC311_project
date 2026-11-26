"""Verify the frozen MLP and build the two-file section 3.4 archive."""
import ast
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'MLP_training'
FILES = ('pred.py', 'mlp_params.npz')

WORKER = r'''
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
import time

class BlockML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'sklearn', 'torch', 'tensorflow'}:
            raise ImportError('Prohibited inference dependency: ' + fullname)
sys.meta_path.insert(0, BlockML())

import numpy as np
import pandas as pd
bundle, root = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location('submission_pred', bundle / 'pred.py')
pred = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pred)
params = pred.load_params()
assert params['meta']['trained_on'] == 'train+val'
assert set(params['classes']) == {'Soccer', 'Baseball', 'Basketball'}
test = pd.read_csv(root / 'data/splits/test.csv', dtype=str, keep_default_na=False)
archive = pd.read_csv(root / 'MLP_training/out/test_eval/test_predictions.csv', dtype=str)
assert test.id.tolist() == archive.id.tolist()
assert test.label.tolist() == archive.actual.tolist()
raw = bundle / 'unlabelled.csv'
test.drop(columns=['id', 'label']).to_csv(raw, index=False, encoding='utf-8-sig')
labels = pred.predict_all(raw)
assert isinstance(labels, list) and all(isinstance(x, str) for x in labels)
assert labels == archive.predicted.tolist()
test.drop(columns=['id', 'label']).iloc[::-1].to_csv(raw, index=False)
assert pred.predict_all(raw) == labels[::-1]
train = pd.read_csv(root / 'data/splits/train.csv', dtype=str, keep_default_na=False)
batch = train.iloc[:60].drop(columns=['id', 'label'])
batch.to_csv(raw, index=False)
times = []
for _ in range(5):
    pred._PARAMS = None
    start = time.perf_counter()
    result = pred.predict_all(raw)
    times.append(time.perf_counter() - start)
    assert len(result) == 60 and set(result) <= set(params['classes'])
assert max(times) < 60, 'Section 3.4 runtime limit exceeded'
short = batch.rename(columns=dict(zip(pred.FULL_COLUMNS, pred.SHORT_COLUMNS)))
short.to_csv(raw, index=False)
assert pred.predict_all(raw) == result
batch.iloc[:0].to_csv(raw, index=False)
assert pred.predict_all(raw) == []
missing = batch.iloc[:1].copy()
missing.loc[:, :] = ''
missing.to_csv(raw, index=False)
assert len(pred.predict_all(raw)) == 1
assert not any(n.split('.')[0] in {'sklearn', 'torch', 'tensorflow'} for n in sys.modules)
print(json.dumps({'archived_test_labels_matched': len(labels),
                  'archived_test_correct': sum(a == b for a,b in zip(labels,test.label)),
                  'unlabelled_and_reversed_rows': 'passed', 'short_headers': 'passed',
                  'empty_and_missing_inputs': 'passed', 'prohibited_imports_blocked': True,
                  'timing_rows': 60, 'timing_repeats': 5,
                  'max_seconds_excluding_imports': max(times),
                  'numpy': np.__version__, 'pandas': pd.__version__}))
'''

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if sys.version_info < (3, 10):
        raise RuntimeError('Python 3.10 or later is required')
    source = HERE / 'submission'
    predictor = source / 'pred.py'
    allowed = sys.stdlib_module_names | {'numpy', 'pandas'}
    imports = []
    for node in ast.walk(ast.parse(predictor.read_text(encoding='utf-8'))):
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module, 'Local import in submission'
            imports.append(node.module.split('.')[0])
    assert set(imports) <= allowed, f'Unexpected imports: {set(imports) - allowed}'
    assert digest(source / 'mlp_params.npz') == digest(HERE / 'out/final_trainval/mlp_final.npz'), 'Submission weights differ from frozen refit'
    total = sum((source / name).stat().st_size for name in FILES)
    assert total <= 10_000_000, 'Submission exceeds 10 MB'
    release = HERE / 'release'
    release.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mlp_submission_') as folder:
        scratch = Path(folder)
        archive = scratch / 'mlp_submission.zip'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for name in FILES:
                # Fixed archive timestamps make identical input files produce identical ZIPs.
                info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, (source / name).read_bytes())
        extracted = scratch / 'extracted'
        with zipfile.ZipFile(archive) as z:
            assert tuple(z.namelist()) == FILES
            z.extractall(extracted)
        assert all(digest(extracted / n) == digest(source / n) for n in FILES)
        completed = subprocess.run([sys.executable, '-I', '-c', WORKER, str(extracted), str(ROOT)],
                                   cwd=scratch, capture_output=True, text=True, timeout=120)
        if completed.returncode:
            raise RuntimeError(completed.stdout + completed.stderr)
        verification = json.loads(completed.stdout)
        # No-argument execution must exit with usage, not wait for terminal input.
        cli = subprocess.run([sys.executable, '-I', str(extracted / 'pred.py')],
                             cwd=scratch, capture_output=True, text=True, timeout=30)
        assert cli.returncode != 0 and 'Usage:' in cli.stderr
        shutil.copyfile(archive, release / archive.name)
    manifest = {'model': 'MLP', 'fit_split': 'train+validation', 'fit_rows': 738,
                'weights_changed': False, 'total_uncompressed_bytes': total,
                'files': {name: {'sha256': digest(source/name), 'bytes': (source/name).stat().st_size}
                          for name in FILES},
                'archive_sha256': digest(release / 'mlp_submission.zip'),
                'imports': sorted(set(imports)), 'minimum_python': '3.10'}
    verification.update(python=platform.python_version(), platform=platform.platform(),
                        model_retrained=False, no_argument_cli='usage without prompting',
                        archive_sha256=manifest['archive_sha256'])
    for name, data in [('manifest.json', manifest), ('verification.json', verification)]:
        (release / name).write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'manifest': manifest, 'verification': verification}, indent=2))

if __name__ == '__main__':
    main()

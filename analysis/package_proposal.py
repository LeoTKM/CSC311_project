"""Copy the proposal PDF and package source code without data or model artifacts."""
import argparse
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ('analysis', 'data_preprocessing', 'data_visualization')


def source_files(root):
    root = Path(root).resolve()
    files = []
    for directory in SOURCE_DIRS:
        for path in (root / directory).rglob('*'):
            relative = path.relative_to(root)
            if any(part in ('out', '__pycache__', '.venv') for part in relative.parts):
                continue
            if path.suffix not in ('.py', '.ipynb'):
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError(f'Refusing source link outside the package: {relative}')
            if path.is_file():
                files.append(path)
    for name in ('README.md', 'analysis/README.md', 'analysis/requirements.txt'):
        path = root / name
        if path.is_symlink():
            raise ValueError(f'Refusing source link: {name}')
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def package(root, output_dir):
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    pdf = root / 'analysis/report/proposal.pdf'
    if not pdf.is_file() or pdf.is_symlink() or not pdf.read_bytes().startswith(b'%PDF-'):
        raise ValueError('Expected a compiled analysis/report/proposal.pdf.')
    sources = source_files(root)
    if not sources:
        raise ValueError('No source files found.')
    targets = [output_dir / 'proposal.pdf', output_dir / 'code.zip']
    for target in targets:
        if target.exists():
            raise FileExistsError(f'Will not overwrite {target}; choose a new --output-dir.')
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(targets[1], 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            info = zipfile.ZipInfo(path.relative_to(root).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    with zipfile.ZipFile(targets[1]) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP integrity check failed.')
    with pdf.open('rb') as source, targets[0].open('xb') as destination:
        shutil.copyfileobj(source, destination)
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'submission')
    args = parser.parse_args()
    for path in package(ROOT, args.output_dir):
        print(f'{path} ({path.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()

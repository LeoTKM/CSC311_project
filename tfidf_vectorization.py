#!/usr/bin/env python3
"""
Add TF-IDF features for chosen text columns of a CSV.

python tfidf_vectorization.py /data/class_dataset.csv --columns 8 9 10 11 --max-features 30
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def resolve_columns(df, specs):
    """Accept column names or 1-based column numbers."""
    resolved = []
    for spec in specs:
        if spec in df.columns:
            resolved.append(spec)
        elif spec.isdigit() and 1 <= int(spec) <= len(df.columns):
            resolved.append(df.columns[int(spec) - 1])
        else:
            sys.exit(f"Column not found: {spec!r}. Available: {list(df.columns)}")
    return resolved


def clean(series):
    """Fill blanks, lowercase, strip stray whitespace."""
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def safe_name(text):
    return re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_")


def main():
    p = argparse.ArgumentParser(description="Add TF-IDF features for text columns of a CSV.")
    p.add_argument("input", help="input CSV file")
    p.add_argument("--columns", "-c", nargs="+", required=True,
                   help="column names (or 1-based numbers) to vectorize")
    p.add_argument("--output", "-o", help="output path (default: <input>_tfidf.csv in same folder)")
    p.add_argument("--max-features", type=int, default=50,
                   help="max TF-IDF terms kept per column (default 50)")
    p.add_argument("--min-df", type=int, default=2,
                   help="drop terms appearing in fewer than this many rows (default 2)")
    p.add_argument("--ngram-max", type=int, default=2,
                   help="use word n-grams up to this length (default 2)")
    p.add_argument("--stop-words", action="store_true",
                   help="remove English stop words (off by default: 'not', 'very' etc. can carry meaning)")
    p.add_argument("--drop-original", action="store_true",
                   help="drop the original text columns from the output")
    p.add_argument("--encoding", default="utf-8", help="CSV encoding (default utf-8)")
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        sys.exit(f"File not found: {in_path}")

    df = pd.read_csv(in_path, encoding=args.encoding)
    cols = resolve_columns(df, args.columns)

    new_frames = []
    for col in cols:
        texts = clean(df[col])
        if (texts == "").all():
            print(f"[skip] {col!r} is empty")
            continue

        vec = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, args.ngram_max),
            min_df=args.min_df,
            max_features=args.max_features,
            stop_words="english" if args.stop_words else None,
            sublinear_tf=True,
        )
        try:
            matrix = vec.fit_transform(texts)
        except ValueError:
            # e.g. min_df removed every term; retry with min_df=1
            vec.set_params(min_df=1)
            matrix = vec.fit_transform(texts)

        prefix = f"tfidf_{safe_name(col)}_"
        names = [prefix + safe_name(t) for t in vec.get_feature_names_out()]
        frame = pd.DataFrame(matrix.toarray(), columns=names, index=df.index)
        new_frames.append(frame)
        print(f"[ok]   {col!r}: {frame.shape[1]} features")

    base = df.drop(columns=cols) if args.drop_original else df
    out = pd.concat([base] + new_frames, axis=1)

    out_path = Path(args.output) if args.output else in_path.with_name(f"{in_path.stem}_tfidf.csv")
    out.to_csv(out_path, index=False, encoding=args.encoding)
    print(f"Wrote {out_path}  ({out.shape[0]} rows, {out.shape[1]} columns)")


if __name__ == "__main__":
    main()
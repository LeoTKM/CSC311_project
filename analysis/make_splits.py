#!/usr/bin/env python3
"""
Grouped train / val / test split of data/class_dataset.csv.

The split is grouped by `id`: each `id` owns 3 rows (Soccer / Baseball /
Basketball), and all rows of one id stay in the same split.

Reproducible via a seeded numpy RNG

Usage:
    python analysis/make_splits.py                       # 70/15/15, seed 311
    python analysis/make_splits.py --test-size 0.1 --val-size 0.1 --seed 7
    python analysis/make_splits.py --from-manifest data/splits/id_to_split.csv

Outputs (default `data/splits/`)
    train.csv, val.csv, test.csv   row-level splits, original column order
    id_to_split.csv                manifest: id,split,n_rows,labels
    split_summary.json             sizes and class counts per split
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

SPLIT_NAMES = ("train", "val", "test")


def load(path):
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig",
                       keep_default_na=False, na_values=[])


def build_manifest(groups, seed, test_size, val_size):
    """id -> split, by shuffling the sorted ids with a seeded RNG."""
    ordered = sorted(groups, key=str)
    rng = np.random.default_rng(seed)
    shuffled = [ordered[i] for i in rng.permutation(len(ordered))]

    n = len(shuffled)
    n_test = max(1, int(round(n * test_size)))
    n_val = max(1, int(round(n * val_size)))
    if n_test + n_val >= n:
        sys.exit(f"sizes leave nothing for train: n={n}, test={n_test}, val={n_val}")

    manifest = {}
    for g in shuffled[:n_test]:
        manifest[g] = "test"
    for g in shuffled[n_test:n_test + n_val]:
        manifest[g] = "val"
    for g in shuffled[n_test + n_val:]:
        manifest[g] = "train"
    return manifest


def describe(df, mask, group_col, label_col):
    sub = df[mask]
    return {
        "groups": int(sub[group_col].nunique()),
        "rows": int(len(sub)),
        "rows_per_class": {str(k): int(v) for k, v in
                           sub[label_col].value_counts().sort_index().items()},
    }


def main():
    ap = argparse.ArgumentParser(description="grouped train/val/test split")
    ap.add_argument("--data", default="data/class_dataset.csv")
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--group-col", default="id")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--seed", type=int, default=311)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--from-manifest", help="reuse an existing id_to_split.csv")
    args = ap.parse_args()

    df = load(args.data)
    for col in (args.group_col, args.label_col):
        if col not in df.columns:
            sys.exit(f"missing column {col!r}")
    groups = list(df[args.group_col].unique())

    if args.from_manifest:
        man = pd.read_csv(args.from_manifest, dtype=str, encoding="utf-8-sig",
                          keep_default_na=False, na_values=[])
        manifest = dict(zip(man["id"], man["split"]))
    else:
        manifest = build_manifest(groups, args.seed, args.test_size, args.val_size)

    row_split = df[args.group_col].map(manifest)
    if row_split.isna().any():
        sys.exit("manifest does not cover every group in the data")

    os.makedirs(args.out_dir, exist_ok=True)
    for name in SPLIT_NAMES:
        sub = df.loc[row_split == name]
        sub.to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False, encoding="utf-8")

    man_rows = []
    for gid in sorted(manifest, key=lambda g: (0, int(g)) if str(g).isdigit() else (1, str(g))):
        sub = df[df[args.group_col] == gid]
        man_rows.append({"id": gid, "split": manifest[gid], "n_rows": int(len(sub)),
                         "labels": "|".join(sorted(sub[args.label_col].astype(str)))})
    pd.DataFrame(man_rows).to_csv(os.path.join(args.out_dir, "id_to_split.csv"),
                                  index=False, encoding="utf-8")

    summary = {
        "dataset": args.data,
        "group_col": args.group_col,
        "label_col": args.label_col,
        "seed": args.seed,
        "n_rows": int(len(df)),
        "n_groups": len(groups),
        "splits": {name: describe(df, row_split == name, args.group_col, args.label_col)
                   for name in SPLIT_NAMES},
    }
    with open(os.path.join(args.out_dir, "split_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"{args.data}  rows={len(df)}  groups={len(groups)}  seed={args.seed}")
    for name in SPLIT_NAMES:
        s = summary["splits"][name]
        per_class = "  ".join(f"{k}={v}" for k, v in s["rows_per_class"].items())
        print(f"{name:<5} rows={s['rows']:<4} groups={s['groups']:<4} {per_class}")


if __name__ == "__main__":
    main()

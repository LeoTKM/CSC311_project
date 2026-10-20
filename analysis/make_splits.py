#!/usr/bin/env python3
"""
Reproducible grouped train / val / test split of data/class_dataset.csv.

Why grouped: the 870 rows come from 290 respondents (`id`). Each respondent
answered once per sport, i.e. every `id` owns exactly 3 rows (one per class:
Soccer / Baseball / Basketball). Rows sharing an `id` are NOT independent, so
all rows of one respondent must land in the same split -- otherwise the same
person's answers leak from train into val/test and every reported score is
optimistic.

How reproducibility is guaranteed
--------------------------------
Groups are ordered by `sha256("<seed>:<id>")`. That means the split is a pure
function of (seed, id set) -- no RNG state, so it cannot drift with the numpy
version, platform, or the order of rows in the CSV. Re-running with the same
seed produces byte-identical output files.

A manifest (`id_to_split.csv`) is also written. Pass `--from-manifest` to reuse
an existing manifest without recomputing anything, which keeps the split stable
even if this script later changes.

Usage (conda env csc311)
------------------------
    conda activate csc311
    python analysis/make_splits.py                       # 70/15/15, seed 311
    python analysis/make_splits.py --test-size 0.1 --val-size 0.1 --seed 7
    python analysis/make_splits.py --from-manifest data/splits/id_to_split.csv

Outputs (default `data/splits/`)
-------------------------------
    train.csv, val.csv, test.csv   row-level splits, original column order
    id_to_split.csv                manifest: id,split,n_rows,labels
    split_summary.json             sizes, class balance, per-split id hashes
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

import pandas as pd

SPLIT_NAMES = ("train", "val", "test")


def load(path):
    """Read every column as text.

    Two reasons:
      * keep_default_na=False -> real answers such as "None" stay strings
      * dtype=str -> the split files reproduce the source *text* exactly. Without
        it pandas infers Q1 ("how easy", mixing 5 / 7.5 / blank) as float and
        writes "5.0" back instead of "5", i.e. the rows would no longer match the
        source file. Type conversion belongs to the analysis scripts, not here.
    """
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig",
                       keep_default_na=False, na_values=[])


def rank_key(seed, gid):
    """Deterministic ordering key: sha256(f'{seed}:{gid}')."""
    return hashlib.sha256(f"{seed}:{gid}".encode("utf-8")).hexdigest()


def build_manifest(groups, seed, test_size, val_size):
    """id -> split, by sorting ids on their seeded hash."""
    ordered = sorted(groups, key=lambda g: (rank_key(seed, g), str(g)))
    n = len(ordered)
    n_test = max(1, int(round(n * test_size)))
    n_val = max(1, int(round(n * val_size)))
    if n_test + n_val >= n:
        sys.exit(f"sizes leave nothing for train: n={n}, test={n_test}, val={n_val}")

    manifest = {}
    for g in ordered[:n_test]:
        manifest[g] = "test"
    for g in ordered[n_test:n_test + n_val]:
        manifest[g] = "val"
    for g in ordered[n_test + n_val:]:
        manifest[g] = "train"
    return manifest, ordered


def ids_hash(ids):
    return hashlib.sha256(",".join(sorted(str(i) for i in ids)).encode("utf-8")).hexdigest()


def verify(df, manifest, group_col, label_col):
    """Hard checks; returns a list of (check, ok, detail)."""
    checks = []

    unknown = sorted(set(df[group_col]) - set(manifest))
    checks.append(("every group has a split", not unknown, f"{len(unknown)} unassigned"))

    assigned_ids = [i for i, s in manifest.items() if s in SPLIT_NAMES]
    checks.append(("no duplicate/unknown split names", len(assigned_ids) == len(manifest), ""))

    row_split = df[group_col].map(manifest)
    checks.append(("no row left unassigned", row_split.notna().all(), ""))

    for name in SPLIT_NAMES:
        n_groups = sum(1 for s in manifest.values() if s == name)
        n_rows = int((row_split == name).sum())
        checks.append((f"{name}: non-empty", n_groups > 0 and n_rows > 0,
                       f"{n_groups} ids / {n_rows} rows"))

    # group integrity: no id may appear in two splits (true by construction,
    # re-checked against the row->split mapping actually used)
    per_id = row_split.groupby(df[group_col]).nunique()
    checks.append(("each id maps to exactly one split", int(per_id.max()) == 1,
                   f"max distinct splits per id = {int(per_id.max())}"))

    # class balance: every split should happen to hold each class
    for name in SPLIT_NAMES:
        labels = df.loc[row_split == name, label_col]
        missing = sorted(set(df[label_col]) - set(labels))
        checks.append((f"{name}: all {df[label_col].nunique()} classes present",
                       not missing, f"missing {missing}" if missing else ""))

    return checks


def row_multiset(df):
    """Multiset of rows as text tuples -- order-insensitive, byte-exact comparison."""
    return Counter(tuple(str(v) for v in row)
                   for row in df.itertuples(index=False, name=None))


def roundtrip_check(full, out_dir):
    """Re-read the written splits and prove they partition the source exactly."""
    merged = Counter()
    n_rows = 0
    for name in SPLIT_NAMES:
        part = load(os.path.join(out_dir, f"{name}.csv"))
        n_rows += len(part)
        merged.update(row_multiset(part))
    same = merged == row_multiset(full)
    return same, n_rows


def describe_split(df, mask, group_col, label_col):
    sub = df[mask]
    return {
        "groups": int(sub[group_col].nunique()),
        "rows": int(len(sub)),
        "rows_per_class": {str(k): int(v) for k, v in
                           sub[label_col].value_counts().sort_index().items()},
        "rows_per_group": {"min": int(sub.groupby(group_col).size().min()),
                           "max": int(sub.groupby(group_col).size().max())},
        "ids_sha256": ids_hash(sub[group_col].unique()),
    }


def main():
    ap = argparse.ArgumentParser(description="reproducible grouped train/val/test split")
    ap.add_argument("--data", default="data/class_dataset.csv")
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--group-col", default="id", help="grouping column (default: id)")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--seed", type=int, default=311)
    ap.add_argument("--test-size", type=float, default=0.15, help="fraction of GROUPS in test")
    ap.add_argument("--val-size", type=float, default=0.15, help="fraction of GROUPS in val")
    ap.add_argument("--from-manifest", help="reuse an existing id_to_split.csv instead of splitting")
    args = ap.parse_args()

    df = load(args.data)
    for col in (args.group_col, args.label_col):
        if col not in df.columns:
            sys.exit(f"missing column {col!r}")

    groups = list(df[args.group_col].unique())
    print(f"data: {args.data}  rows={len(df)}  groups({args.group_col})={len(groups)}  "
          f"classes={sorted(df[args.label_col].unique())}")

    # ---- build or load the manifest
    if args.from_manifest:
        man = pd.read_csv(args.from_manifest, dtype=str, encoding="utf-8-sig",
                          keep_default_na=False, na_values=[])
        manifest = dict(zip(man["id"], man["split"]))
        print(f"manifest reused from {args.from_manifest} (seed/sizes ignored)")
    else:
        manifest, _ = build_manifest(groups, args.seed, args.test_size, args.val_size)
        print(f"split by seeded sha256: seed={args.seed}  "
              f"test_size={args.test_size}  val_size={args.val_size}")

    row_split = df[args.group_col].map(manifest)
    if row_split.isna().any():
        sys.exit("manifest does not cover every group in the data")

    # ---- verification
    print("\nverification:")
    failed = 0
    for name, ok, detail in verify(df, manifest, args.group_col, args.label_col):
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        failed += 0 if ok else 1
    if failed:
        sys.exit(f"{failed} check(s) failed")

    # ---- write row-level splits
    os.makedirs(args.out_dir, exist_ok=True)
    for name in SPLIT_NAMES:
        sub = df.loc[row_split == name].drop(columns=[c for c in ("_split",) if c in df.columns])
        path = os.path.join(args.out_dir, f"{name}.csv")
        sub.to_csv(path, index=False, encoding="utf-8")
        print(f"\nwrote {path}  ({sub.shape[0]} rows, {sub.shape[1]} cols)")

    # ---- the three files must reproduce the source rows exactly (text-exact)
    same, n_rows = roundtrip_check(df, args.out_dir)
    print(f"\n[{'OK' if same else 'FAIL'}] train+val+test reproduce the source rows exactly "
          f"({n_rows} rows read back)")
    if not same:
        sys.exit("round-trip check failed: split files do not match the source data")

    # ---- write manifest (numeric ids first, so the file reads naturally)
    man_rows = []
    ordered_ids = sorted(manifest, key=lambda g: (0, int(g), "")
                         if str(g).lstrip("-").isdigit() else (1, 0, str(g)))
    for gid in ordered_ids:
        sub = df[df[args.group_col] == gid]
        man_rows.append({
            "id": gid,
            "split": manifest[gid],
            "n_rows": int(len(sub)),
            "labels": "|".join(sorted(sub[args.label_col].astype(str))),
        })
    man_path = os.path.join(args.out_dir, "id_to_split.csv")
    pd.DataFrame(man_rows).to_csv(man_path, index=False, encoding="utf-8")
    print(f"wrote {man_path}  ({len(man_rows)} ids)")

    # ---- summary (deterministic content: no timestamps, so re-runs are stable)
    summary = {
        "dataset": args.data,
        "group_col": args.group_col,
        "label_col": args.label_col,
        "seed": args.seed,
        "requested_sizes": {"test": args.test_size, "val": args.val_size},
        "n_rows": int(len(df)),
        "n_groups": len(groups),
        "splits": {name: describe_split(df, row_split == name, args.group_col, args.label_col)
                   for name in SPLIT_NAMES},
    }
    total = sum(s["groups"] for s in summary["splits"].values())
    summary["splits"]["train"]["fraction_of_groups"] = round(
        summary["splits"]["train"]["groups"] / total, 4)
    for name in ("val", "test"):
        summary["splits"][name]["fraction_of_groups"] = round(
            summary["splits"][name]["groups"] / total, 4)

    sum_path = os.path.join(args.out_dir, "split_summary.json")
    with open(sum_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"wrote {sum_path}")

    # ---- console report
    print("\nsplit sizes (rows / groups / fraction of groups):")
    for name in SPLIT_NAMES:
        s = summary["splits"][name]
        print(f"  {name:<5} {s['rows']:>4} rows  {s['groups']:>3} groups  "
              f"{s['fraction_of_groups']:.3f}  "
              f"class counts={s['rows_per_class']}  ids_sha256={s['ids_sha256'][:12]}...")
    print("\nreproduce this exact split later with:")
    print(f"  python analysis/make_splits.py --seed {args.seed} "
          f"--test-size {args.test_size} --val-size {args.val_size}")
    print(f"  # or reuse the manifest: --from-manifest {man_path}")


if __name__ == "__main__":
    main()

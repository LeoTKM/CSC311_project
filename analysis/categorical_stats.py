#!/usr/bin/env python3
"""
Descriptive statistics for the single-choice (Likert) and multi-select questions.

Reads a split of the survey CSV (train by default) and prints the statistics to
the terminal; the same numbers are also written as tidy CSV tables.

Usage:
    python analysis/categorical_stats.py
    python analysis/categorical_stats.py --data data/splits/val.csv
    python analysis/categorical_stats.py --out-dir analysis/out/train
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csv_output import write_categorical_tables          # noqa: E402
from survey_common import (                              # noqa: E402
    CLASSES,
    LIKERT_SCALE,
    MULTI_SPECS,
    MULTI_VOCAB,
    MULTI_VOCAB_BY_Q,
    QUALITY_KIND_LABEL,
    QUALITY_SPECS,
    SEASON_INDEX,
    SINGLE_SPECS,
    cramers_v,
    col_by_prefix,
    describe,
    eta_sq,
    feature_quality,
    load,
    load_split_summary,
)


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and x != x):
        return "na"
    return f"{x:.{nd}f}"



def quality_table(df):
    """Blank / invalid / valid counts for all 12 feature columns."""
    rows = []
    for q, name, prefix, kind in QUALITY_SPECS:
        blank, invalid, valid, examples = feature_quality(df, prefix, kind, MULTI_VOCAB_BY_Q.get(q))
        rows.append({
            "q": q, "name": name, "kind": kind, "kind_label": QUALITY_KIND_LABEL[kind],
            "rows": len(df), "blank": blank, "invalid": invalid, "valid": valid,
            "share": 100 * valid / len(df), "examples": examples,
        })
    return rows


def single_choice_stats(df, classes):
    """Per-question statistics of the three 5-point Likert questions."""
    results, rows, ranking = [], [], []

    for q, name, prefix in SINGLE_SPECS:
        col = col_by_prefix(df, prefix)
        raw = df[col].astype(str).str.strip()
        score = raw.map(LIKERT_SCALE)

        valid = pd.DataFrame({"score": score, "label": df["label"]}).dropna()
        n_all, mu, var, sd = describe(valid["score"].values)
        per_class = {c: describe(valid.loc[valid.label == c, "score"].values) for c in classes}

        levels = sorted(LIKERT_SCALE.values())
        table = [[int(((valid.label == c) & (valid.score == lv)).sum()) for c in classes]
                 for lv in levels]
        v = cramers_v(table)
        eta2 = eta_sq([valid.loc[valid.label == c, "score"].values for c in classes])

        blank, invalid, valid_n, examples = feature_quality(df, prefix, "likert")
        question = f"{q} {name}"
        ranking.append((question, v, eta2, "single"))

        results.append({
            "question": question, "column": col, "levels": levels,
            "n": n_all, "mean": mu, "var": var, "sd": sd,
            "per_class": {c: {"n": per_class[c][0], "mean": per_class[c][1],
                              "var": per_class[c][2], "sd": per_class[c][3]} for c in classes},
            "table": table, "cramers_v": v, "eta2": eta2,
            "quality": {"blank": blank, "invalid": invalid, "valid": valid_n,
                        "examples": examples},
        })

        for c in classes:
            rows.append(dict(question=question, kind="single_ordinal", class_=c,
                             n=per_class[c][0], mean=per_class[c][1], var=per_class[c][2],
                             sd=per_class[c][3], cramers_v=v, eta2=eta2))
        rows.append(dict(question=question, kind="single_ordinal", class_="ALL",
                         n=n_all, mean=mu, var=var, sd=sd, cramers_v=v, eta2=eta2))

    return results, rows, ranking


def multi_select_stats(df, classes, ranking):
    """Per-option counts of the three checkbox questions (+ selection-count distribution).

    No mean and no variance: for a 0/1 indicator the variance is fully determined by
    the mean, so only the raw counts are reported.
    """
    results, rows, count_rows = [], [], []
    labs = df["label"]

    for (q, name, prefix), (vname, vocab) in zip(MULTI_SPECS, MULTI_VOCAB.items()):
        col = col_by_prefix(df, prefix)
        raw = df[col].astype(str).str.strip().replace("", np.nan)
        tokens = raw.map(lambda s: {t.strip() for t in s.split(",")} if isinstance(s, str) else set())
        valid_mask = raw.notna()
        n_all = int(valid_mask.sum())
        if n_all == 0:
            continue
        question = f"{q} {name}"
        n_by_class = {c: int((labs[valid_mask] == c).sum()) for c in classes}

        # derived feature: how many options each respondent selected -> count distribution
        cnt = tokens[valid_mask].map(len)
        cnt_by_class = {c: cnt[labs[valid_mask] == c].values for c in classes}
        max_cnt = int(max((len(s) for s in tokens[valid_mask]), default=0))
        values = list(range(max_cnt + 1))
        dist = {c: [int((cnt_by_class[c] == k).sum()) for k in values] for c in classes}

        blank, invalid, valid_n, examples = feature_quality(df, prefix, "multi", vocab)

        count_feature = {
            "n": n_all,
            "values": values,
            "dist": dist,
            "by_class": {c: {"n": n_by_class[c]} for c in classes},
        }

        options = []
        for opt in vocab:
            ind = tokens[valid_mask].map(lambda s: 1 if opt in s else 0)
            sel_by_class = {c: int(ind[labs[valid_mask] == c].sum()) for c in classes}
            selected = int(sum(sel_by_class.values()))
            table = [[sel_by_class[c] for c in classes],
                     [n_by_class[c] - sel_by_class[c] for c in classes]]
            v = cramers_v(table)
            ranking.append((f"{question} - {opt}", v, np.nan, "indicator"))

            options.append({
                "option": opt,
                "n": n_all,
                "selected": selected,
                "selected_by_class": sel_by_class,
                "n_by_class": n_by_class,
                "cramers_v": v,
            })
            rows.append(dict(question=question, kind="multi_indicator", option=opt,
                             n=n_all, selected=selected,
                             **{f"selected_{c}": sel_by_class[c] for c in classes},
                             **{f"n_{c}": n_by_class[c] for c in classes},
                             cramers_v=v))

        for k in values:
            count_rows.append(dict(question=question, feature="selection_count", value=k,
                                   n_total=sum(dist[c][k] for c in classes),
                                   **{f"n_{c}": dist[c][k] for c in classes}))

        results.append({
            "question": question, "column": col, "n": n_all,
            "quality": {"blank": blank, "invalid": invalid, "valid": valid_n,
                        "examples": examples},
            "count": count_feature, "options": options,
        })

    return results, rows, count_rows


def analyse_categorical(df, classes=None):
    """Run the whole analysis and return one structured result."""
    classes = classes or [c for c in CLASSES if c in set(df["label"])]
    single, single_rows, ranking = single_choice_stats(df, classes)
    multi, multi_rows, count_rows = multi_select_stats(df, classes, ranking)
    quality = quality_table(df)
    return {
        "classes": classes,
        "n_rows": len(df),
        "class_counts": {c: int((df["label"] == c).sum()) for c in classes},
        "quality": quality,
        "quality_totals": {"blank": sum(r["blank"] for r in quality),
                           "invalid": sum(r["invalid"] for r in quality)},
        "single": single, "multi": multi,
        "single_rows": single_rows, "multi_rows": multi_rows, "count_rows": count_rows,
        "ranking": ranking,
    }



def print_result(result):
    classes = result["classes"]
    print("\ndata quality")
    for r in result["quality"]:
        print(f"  {r['q']:<4} blank={r['blank']:<4} invalid={r['invalid']:<4} valid={r['valid']:<4}")

    for s in result["single"]:
        print(f"\n{s['question']}")
        print(f"  n={s['n']}  mean={fmt(s['mean'])}  var={fmt(s['var'])}  sd={fmt(s['sd'])}")
        print("  " + " | ".join(
            f"{c} n={s['per_class'][c]['n']} mean={fmt(s['per_class'][c]['mean'])} "
            f"var={fmt(s['per_class'][c]['var'])}" for c in classes))
        print(f"  V={fmt(s['cramers_v'])}  eta2={fmt(s['eta2'])}")
        dist = "  ".join(
            f"{lv}=" + "/".join(str(s["table"][i][j]) for j in range(len(classes)))
            for i, lv in enumerate(s["levels"]))
        print(f"  dist {dist}")

    for m in result["multi"]:
        c = m["count"]
        print(f"\n{m['question']}")
        print(f"  n={c['n']}  " + " ".join(f"{cl} n={c['by_class'][cl]['n']}" for cl in classes))
        dist = "  ".join(
            f"{v}=" + "/".join(str(c["dist"][cl][i]) for cl in classes)
            for i, v in enumerate(c["values"]))
        print(f"  #selected {dist}")
        for o in m["options"]:
            print(f"  {o['option']:<34} selected={o['selected']:<4} " +
                  " ".join(f"{cl}={o['selected_by_class'][cl]}" for cl in classes) +
                  f"  V={fmt(o['cramers_v'])}")

    clean = [r for r in result["ranking"] if r[1] == r[1]]
    clean.sort(key=lambda r: -r[1])
    print("\nranking by Cramer's V")
    for i, (name, v, e2, kind) in enumerate(clean, 1):
        print(f"  {i:>2}  {name:<44} V={fmt(v)}  eta2={fmt(e2) if e2 == e2 else '-'}")


def main():
    ap = argparse.ArgumentParser(description="single-choice and multi-select statistics")
    ap.add_argument("--data", default="data/splits/train.csv")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--split-summary", default="data/splits/split_summary.json")
    args = ap.parse_args()

    df = load(args.data)
    if "label" not in df.columns:
        sys.exit("no label column")

    split_info = load_split_summary(args.split_summary)
    print(f"{args.data}  rows={len(df)}")
    if split_info:
        print("  ".join(f"{n}={split_info['splits'][n]['rows']}"
                        for n in ("train", "val", "test")))

    result = analyse_categorical(df)
    print("classes " + "  ".join(f"{c}={result['class_counts'][c]}" for c in result["classes"]))
    print_result(result)

    out_dir = args.out_dir or os.path.join("analysis", "out", Path(args.data).stem)
    paths = write_categorical_tables(result, out_dir)
    print()
    for path in paths.values():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

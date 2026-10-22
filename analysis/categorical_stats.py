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
    """Per-option statistics of the three checkbox questions (+ selection count)."""
    results, rows, count_rows = [], [], []
    labs = df["label"]

    for (q, name, prefix), (vname, vocab) in zip(MULTI_SPECS, MULTI_VOCAB.items()):
        col = col_by_prefix(df, prefix)
        raw = df[col].astype(str).str.strip().replace("", np.nan)
        tokens = raw.map(lambda s: {t.strip() for t in s.split(",")} if isinstance(s, str) else set())
        valid_mask = raw.notna()
        if int(valid_mask.sum()) == 0:
            continue
        question = f"{q} {name}"

        cnt = tokens[valid_mask].map(len)
        cnt_by_class = {c: cnt[labs[valid_mask] == c].values for c in classes}
        n_all, mu, var, sd = describe(cnt.values)
        eta2 = eta_sq(list(cnt_by_class.values()))
        ranking.append((f"{question} #selected", np.sqrt(eta2), eta2, "count"))

        blank, invalid, valid_n, examples = feature_quality(df, prefix, "multi", vocab)

        count_feature = {
            "n": n_all, "mean": mu, "var": var, "sd": sd,
            "by_class": {c: {"n": describe(cnt_by_class[c])[0],
                             "mean": describe(cnt_by_class[c])[1],
                             "var": describe(cnt_by_class[c])[2]} for c in classes},
            "eta2": eta2,
        }

        options = []
        for opt in vocab:
            ind = tokens[valid_mask].map(lambda s: 1 if opt in s else 0)
            by_class = {c: ind[labs[valid_mask] == c].values for c in classes}
            n_o, mu_o, var_o, sd_o = describe(ind.values)
            table = [[int(sum(by_class[c])), int(len(by_class[c]) - sum(by_class[c]))] for c in classes]
            table = [list(x) for x in zip(*table)]
            v = cramers_v(table)
            ranking.append((f"{question} - {opt}", v, np.nan, "indicator"))

            options.append({
                "option": opt, "n": n_o, "rate": mu_o, "var": var_o, "sd": sd_o,
                "by_class": {c: {"n": describe(by_class[c])[0],
                                 "rate": describe(by_class[c])[1],
                                 "var": describe(by_class[c])[2]} for c in classes},
                "cramers_v": v,
            })
            rows.append(dict(question=question, kind="multi_indicator", option=opt, class_="ALL",
                             n=n_o, rate=mu_o, var=var_o, sd=sd_o, cramers_v=v,
                             **{f"rate_{c}": describe(by_class[c])[1] for c in classes},
                             **{f"var_{c}": describe(by_class[c])[2] for c in classes},
                             **{f"n_{c}": describe(by_class[c])[0] for c in classes}))

        season_supplement = None
        if "season" in name:
            idx = {c: [] for c in classes}
            for _, r in df.iterrows():
                s = str(r[col]).strip()
                if not s:
                    continue
                for t in {x.strip() for x in s.split(",") if x.strip()}:
                    if t in SEASON_INDEX:
                        idx[r["label"]].append(SEASON_INDEX[t])
            allv = [v for c in classes for v in idx[c]]
            n_s, mu_s, var_s, _ = describe(allv)
            season_supplement = {
                "n": n_s, "mean": mu_s, "var": var_s,
                "by_class": {c: {"mean": describe(idx[c])[1], "var": describe(idx[c])[2]}
                             for c in classes},
                "eta2": eta_sq([idx[c] for c in classes]),
            }

        results.append({
            "question": question, "column": col, "n": n_all,
            "quality": {"blank": blank, "invalid": invalid, "valid": valid_n,
                        "examples": examples},
            "count": count_feature, "options": options,
            "season_supplement": season_supplement,
        })

        count_rows.append(dict(question=question, feature="selection_count", n=n_all,
                               mean=mu, var=var, sd=sd, eta2=eta2,
                               **{f"mean_{c}": describe(cnt_by_class[c])[1] for c in classes},
                               **{f"var_{c}": describe(cnt_by_class[c])[2] for c in classes}))

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
        print(f"  count n={c['n']} mean={fmt(c['mean'])} var={fmt(c['var'])} "
              f"sd={fmt(c['sd'])} eta2={fmt(c['eta2'])}")
        print("  " + " | ".join(
            f"{cl} n={c['by_class'][cl]['n']} mean={fmt(c['by_class'][cl]['mean'])} "
            f"var={fmt(c['by_class'][cl]['var'])}" for cl in classes))
        for o in m["options"]:
            print(f"  {o['option']:<34} rate={fmt(o['rate'])}  " +
                  " ".join(f"{cl}={fmt(o['by_class'][cl]['rate'])}" for cl in classes) +
                  "  var=" + "/".join(fmt(o["by_class"][cl]["var"]) for cl in classes) +
                  f"  V={fmt(o['cramers_v'])}")
        sup = m.get("season_supplement")
        if sup:
            print(f"  season index mean={fmt(sup['mean'])} var={fmt(sup['var'])} "
                  f"eta2={fmt(sup['eta2'])}  " +
                  " ".join(f"{cl}={fmt(sup['by_class'][cl]['mean'])}" for cl in classes))

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

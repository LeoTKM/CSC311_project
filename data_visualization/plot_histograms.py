#!/usr/bin/env python3
"""
Histograms of the categorical survey columns, grouped by the 3 label classes.

The default input is the TRAIN split (data/splits/train.csv). With the
`id`-grouped splits in place, figures that describe the data must be drawn from
train only, so val/test cannot leak into the picture.

Outputs (PNG) into data_visualization/figures/ by default.

Usage:
    python data_visualization/plot_histograms.py                         # train split
    python data_visualization/plot_histograms.py --data data/splits/val.csv
    python data_visualization/plot_histograms.py --out-dir data_visualization/figures
"""

import argparse
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASS_ORDER = ["Soccer", "Baseball", "Basketball"]
COLORS = {"Soccer": "#2ca02c", "Baseball": "#1f77b4", "Basketball": "#ff7f0e"}

LIKERT_SCALE = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly agree": 5,
}
LEVEL_LABELS = ["Strongly\ndisagree", "Disagree", "Neutral", "Agree", "Strongly\nagree"]

# (question number, full question text, kind-describer)
SINGLE_SPECS = [
    ("Q2",
     "Success in this sport depends heavily on strategy and planning.",
     "Single choice | 5-point Likert (ordinal 1-5)"),
    ("Q3",
     "Players must be comfortable with a high risk of injury.",
     "Single choice | 5-point Likert (ordinal 1-5)"),
    ("Q4",
     "Individual brilliance can outweigh teamwork in this sport.",
     "Single choice | 5-point Likert (ordinal 1-5)"),
]

MULTI_SPECS = [
    ("Q5", "Where might you play this sport?", 7, [
        ("Indoor facility", "Indoor"),
        ("Outdoor public space", "Outdoor\npublic"),
        ("School or university facility", "School/\nuniv."),
        ("Community recreation facility", "Community\nrec."),
        ("Private sports facility", "Private\nsports"),
        ("At home or nearby", "At home/\nnearby"),
        ("Others", "Others"),
    ]),
    ("Q6",
     "During which seasons would you be most likely to play this sport "
     "recreationally in Toronto?", 4, [
        ("Winter", "Winter"),
        ("Spring", "Spring"),
        ("Summer", "Summer"),
        ("Fall", "Fall"),
    ]),
    ("Q7", "What types of music do you most associate with this sport?", 10, [
        ("Pop", "Pop"),
        ("Electronic or dance", "Electronic/\ndance"),
        ("Latin", "Latin"),
        ("Folk or traditional", "Folk/\ntraditional"),
        ("Classical or jazz", "Classical/\njazz"),
        ("Hip-hop or rap", "Hip-hop/\nrap"),
        ("Country", "Country"),
        ("Rock", "Rock"),
        ("No particular type", "No\nparticular"),
        ("Other", "Other"),
    ]),
]


def setup_style():
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["axes.axisbelow"] = True


def wrap(text, width=64):
    return textwrap.fill(text, width=width)


def load(path):
    # keep_default_na=False: real answers such as "None" must not become NaN
    return pd.read_csv(path, encoding="utf-8-sig",
                       keep_default_na=False, na_values=[])


def col_by_prefix(df, prefix):
    hits = [c for c in df.columns if c.startswith(prefix)]
    if len(hits) != 1:
        sys.exit(f"prefix {prefix!r} matched {len(hits)} columns")
    return hits[0]


def grouped_hist(ax, positions, values_by_class, counts_by_class, annotate_pct=False):
    """One bar per (category, class); the class count is written above each bar."""
    n_cls = len(CLASS_ORDER)
    width = 0.8 / n_cls
    for i, cls in enumerate(CLASS_ORDER):
        vals = values_by_class[cls]
        offs = positions - 0.4 + width * (i + 0.5)
        bars = ax.bar(offs, vals, width=width * 0.92, color=COLORS[cls],
                      edgecolor="white", linewidth=0.6,
                      label=f"{cls} (n={counts_by_class[cls]})")
        for rect, v in zip(bars, vals):
            if v <= 0:
                continue
            txt = f"{v:.0f}"
            if annotate_pct and counts_by_class[cls]:
                txt += f"\n{100 * v / counts_by_class[cls]:.0f}%"
            ax.annotate(txt, (rect.get_x() + rect.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=7, linespacing=1.0)
    ax.set_ylabel("Number of respondents (count)")
    return max((max(v) if len(v) else 0) for v in values_by_class.values())


def finish(ax, ymax, legend=True, title_pad=12, ylabel=None):
    ax.set_ylim(0, ymax * 1.22 if ymax > 0 else 1)
    if legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3,
                  frameon=False, fontsize=9)
    ax.set_title(ax.get_title(), pad=title_pad)
    if ylabel:
        ax.set_ylabel(ylabel)


def save(fig, out_path, dpi=200):
    """Save with a small automatic margin so nothing can be clipped."""
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)



def weighted_moments(values, lo):
    """Sample mean / variance / sd of a discrete score given per-level counts."""
    counts = np.asarray(values, dtype=float)
    total = counts.sum()
    if total == 0:
        return float("nan"), float("nan"), float("nan")
    idx = np.arange(lo, lo + len(counts), dtype=float)
    mu = float((idx * counts).sum() / total)
    if total < 2:
        return mu, float("nan"), float("nan")
    var = float(((idx - mu) ** 2 * counts).sum() / (total - 1))
    return mu, var, float(np.sqrt(var))


def compact_stats(values, lo):
    """'3.92 / 3.49 / 3.90' style strings in CLASS_ORDER for mean and variance."""
    means, vars_ = [], []
    for c in CLASS_ORDER:
        mu, var, _ = weighted_moments(values[c], lo)
        means.append(f"{mu:.2f}")
        vars_.append(f"{var:.2f}")
    return " / ".join(means), " / ".join(vars_)



def plot_single(df, spec, out_path):
    q, question, kind = spec
    col = col_by_prefix(df, question[:45])
    raw = df[col].astype(str).str.strip()
    score = raw.map(LIKERT_SCALE)

    values = {c: [int(((df.label == c) & (score == lv)).sum()) for lv in range(1, 6)]
              for c in CLASS_ORDER}
    counts = {c: int(score[df.label == c].notna().sum()) for c in CLASS_ORDER}
    n_all = sum(counts.values())
    missing = len(df) - n_all
    mean_s, var_s = compact_stats(values, 1)

    fig, ax = plt.subplots(figsize=(9.4, 6.4))
    positions = np.arange(5)
    ymax = grouped_hist(ax, positions, values, counts)
    ax.set_xticks(positions)
    ax.set_xticklabels(LEVEL_LABELS, fontsize=9.5)
    ax.set_xlabel("Response level", fontsize=9.5)
    ax.set_title(f"{q} | {wrap(question, 66)}\n"
                 f"Single choice, Likert 1-5 (ordinal score 1-5)   |   "
                 f"Soccer / Baseball / Basketball\n"
                 f"mean: {mean_s}      variance: {var_s}", fontsize=10.5)
    finish(ax, ymax)

    fig.text(0.5, 0.012,
             f"Bar height = count in that class; label on bar = count.  "
             f"n = {n_all} valid answers ({missing} missing).  "
             f"Per-class tables: analysis/out/train/*.csv (see analysis/categorical_stats.py)",
             ha="center", fontsize=8, color="#555")
    save(fig, out_path)
    plt.close(fig)



def tokens_of(series):
    return series.map(lambda s: {t.strip() for t in str(s).split(",")} if str(s).strip() else set())


def plot_multi(df, spec, out_path):
    q, question, n_opt, vocab = spec
    col = col_by_prefix(df, question[:45])
    raw = df[col].astype(str).str.strip().replace("", np.nan)
    toks = tokens_of(raw)
    valid = raw.notna()

    counts = {c: int(((df.label == c) & valid).sum()) for c in CLASS_ORDER}
    values = {c: [int(toks[(df.label == c) & valid].map(lambda s, o=o: o in s).sum())
                  for o, _ in vocab] for c in CLASS_ORDER}
    n_all = sum(counts.values())

    fig, ax = plt.subplots(figsize=(13.4, 6.2))
    positions = np.arange(len(vocab))
    ymax = grouped_hist(ax, positions, values, counts, annotate_pct=True)
    ax.set_xticks(positions)
    ax.set_xticklabels([short for _, short in vocab], fontsize=8.5)
    ax.set_xlabel("Option", fontsize=9.5)
    ax.set_title(f"{q} | {wrap(question, 74)}\n"
                 f"Multi-select, {n_opt} options | bar = respondents selecting the option, "
                 f"label = count (within-class %)", fontsize=10.5)
    finish(ax, ymax)

    fig.text(0.5, 0.012,
             f"Bar height = number of respondents in that class who selected the option.  "
             f"n = {n_all} valid answers ({len(df) - n_all} missing).  "
             f"Per-class tables: analysis/out/train/*.csv (see analysis/categorical_stats.py)",
             ha="center", fontsize=8, color="#555")
    save(fig, out_path)
    plt.close(fig)


def count_values(df, spec):
    q, question, n_opt, vocab = spec
    col = col_by_prefix(df, question[:45])
    raw = df[col].astype(str).str.strip().replace("", np.nan)
    toks = tokens_of(raw)
    valid = raw.notna()
    counts = {c: int(((df.label == c) & valid).sum()) for c in CLASS_ORDER}
    values = {c: [int((toks[(df.label == c) & valid].map(len) == k).sum())
                  for k in range(0, n_opt + 1)] for c in CLASS_ORDER}
    return values, counts, n_opt


def plot_count_feature(df, spec, out_path):
    q, question, n_opt, _ = spec
    values, counts, n_opt = count_values(df, spec)

    fig, ax = plt.subplots(figsize=(9.8, 6.4))
    positions = np.arange(0, n_opt + 1)
    ymax = grouped_hist(ax, positions, values, counts)
    ax.set_xticks(positions)
    ax.set_xlabel(f"Number of options selected by the respondent (0-{n_opt})", fontsize=9.5)
    ax.set_title(f"{q} | {wrap(question, 66)}   (number of options selected)\n"
                 f"Multi-select, {n_opt} options, derived count feature", fontsize=10.5)
    finish(ax, ymax)

    fig.text(0.5, 0.012,
             "0 = no option selected (includes blank answers).  "
             f"n = {sum(counts.values())} valid answers "
             f"({len(df) - sum(counts.values())} missing).  "
             f"Per-class tables: analysis/out/train/*.csv (see analysis/categorical_stats.py)",
             ha="center", fontsize=8, color="#555")
    save(fig, out_path)
    plt.close(fig)


def plot_overview(df, single_vals, count_vals, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.2))

    for ax, (spec, values, counts) in zip(axes[0], single_vals):
        q, question, kind = spec
        positions = np.arange(5)
        grouped_hist(ax, positions, values, counts)
        ax.set_xticks(positions)
        ax.set_xticklabels(LEVEL_LABELS, fontsize=7.5)
        ax.set_xlabel("Response level", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        mean_s, var_s = compact_stats(values, 1)
        ax.set_title(f"{q} | {wrap(question, 44)}\n"
                     f"Single choice, Likert 1-5\n"
                     f"mean: {mean_s}\nvariance: {var_s}", fontsize=7.8)
        finish(ax, max(max(v) for v in values.values()), legend=False, title_pad=8)

    for ax, (spec, values, counts, n_opt) in zip(axes[1], count_vals):
        q, question, _, _ = spec
        positions = np.arange(0, n_opt + 1)
        grouped_hist(ax, positions, values, counts)
        ax.set_xticks(positions)
        ax.set_xlabel("Number of options selected", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.set_title(f"{q} | {wrap(question, 44)}\n"
                     f"Multi-select, count feature (0-{n_opt})", fontsize=7.5)
        finish(ax, max(max(v) for v in values.values()), legend=False, title_pad=8)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Categorical questions: distribution of each feature by class "
                 "(Soccer / Baseball / Basketball)\n"
                 "top row titles also give the mean and variance in the order "
                 "Soccer / Baseball / Basketball", fontsize=12)
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    save(fig, out_path, dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="histograms per class for categorical columns")
    ap.add_argument("--data", default="data/splits/train.csv",
                    help="input CSV (default: the train split; never plot val/test)")
    ap.add_argument("--out-dir", default="data_visualization/figures")
    args = ap.parse_args()

    setup_style()
    df = load(args.data)
    classes = sorted(df["label"].unique())
    missing_cls = [c for c in CLASS_ORDER if c not in classes]
    if missing_cls:
        sys.exit(f"classes missing from data: {missing_cls}")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"rows={len(df)}  " + ", ".join(
        f"{c}={int((df.label == c).sum())}" for c in CLASS_ORDER))

    single_vals = []
    for i, spec in enumerate(SINGLE_SPECS, 1):
        q, question, kind = spec
        col = col_by_prefix(df, question[:45])
        score = df[col].astype(str).str.strip().map(LIKERT_SCALE)
        values = {c: [int(((df.label == c) & (score == lv)).sum()) for lv in range(1, 6)]
                  for c in CLASS_ORDER}
        counts = {c: int(score[df.label == c].notna().sum()) for c in CLASS_ORDER}
        single_vals.append((spec, values, counts))
        path = os.path.join(args.out_dir, f"{i:02d}_single_{q}.png")
        plot_single(df, spec, path)
        print(f"  wrote {path}")

    count_vals = []
    for j, spec in enumerate(MULTI_SPECS, 4):
        q = spec[0]
        path = os.path.join(args.out_dir, f"{j:02d}_multi_{q}.png")
        plot_multi(df, spec, path)
        print(f"  wrote {path}")

        values, counts, n_opt = count_values(df, spec)
        count_vals.append((spec, values, counts, n_opt))
        p = os.path.join(args.out_dir, f"{j + 3:02d}_count_{q}.png")
        plot_count_feature(df, spec, p)
        print(f"  wrote {p}")

    overview = os.path.join(args.out_dir, "00_overview.png")
    plot_overview(df, single_vals, count_vals, overview)
    print(f"  wrote {overview}")


if __name__ == "__main__":
    main()

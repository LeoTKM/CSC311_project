#!/usr/bin/env python3
"""
Histograms of Q12 (willingness to pay), grouped by the 3 label classes.

Q12 is a short answer, so `data_preprocessing/encode_all.py` turns it into two
numeric columns instead of a TF-IDF block:

    payment_amount   float, parsed CAD amount, missing replies filled with the
                     training-split mean (recorded in encoding_summary.json)
    payment_missing  1 when the reply had no parseable amount

Those two columns are what this script reads, so the input is an ENCODED file
under data/encoded/, not the raw split under data/splits/. Only the train split
is ever plotted, so val/test cannot leak into a data-description figure.

Two figures are written:

    10_payment_Q12.png           all parsed amounts, categorical amount buckets
    11_payment_Q12_trimmed.png   the upper tail of each class dropped, one panel
                                 per class with its own continuous x axis

Amounts are far too right-skewed for equal-width bins (median 100, mean 274,
max 10000), so figure 10 uses 8 round-amount buckets on a log-like scale plus a
separate "not stated" bucket. Those buckets are categorical positions, not a
numeric axis: the bins are not equally wide, so a numeric axis would draw the
wrong widths.

The trimmed figure drops, inside each class, every amount above that class's
Q3 + 1.5 * IQR. Only the upper tail is cut: every amount from 0 up to the cut
is kept, so the $0 answers (respondents who would not pay) stay in the picture.
After that trim the within-class skew falls from about 7 to about 0.9 and the
range shrinks to 0-625 / 0-200 / 0-400, so figure 11 can use a real numeric
axis with equal 50 CAD bins (bar width equals bin width). The three classes no
longer share one axis, because their trimmed ranges differ by a factor of 3;
only the height (count) is shared. Replies with no parseable amount have no
position on a numeric axis, so they are reported as text in each panel and are
not trimmed.

Amounts are read as CAD. Currency notation is stripped by the encoder without
any conversion, so a dollar sign, "dollars" and "bucks" all count as CAD.

Outputs (PNG) into data_visualization/figures/ by default.

Usage:
    python data_visualization/plot_payment.py
    python data_visualization/plot_payment.py --data data/encoded/train_encoded.csv
    python data_visualization/plot_payment.py --out-dir data_visualization/figures
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_histograms import CLASS_ORDER, COLORS, finish, grouped_hist, save, setup_style

# Upper bounds of the amount buckets. Index 0 is "exactly 0", index 1 is
# (0, 25], index 2 is (25, 50], ... index 7 is (1000, inf). Index 8 is reserved
# for replies with no parseable amount, which is not an amount bucket.
UPPER = np.array([0.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0, np.inf])
LABELS = ["$0", "$1-25", "$25-50", "$50-100", "$100-200",
          "$200-500", "$500-1000", "$1000+", "not\nstated"]
NOT_STATED = len(LABELS) - 1
DROPPED = -1  # bucket code for a valid amount removed by the trim; never counted

TUKEY = 1.5  # the trim cut is Q3 + TUKEY * IQR, per class
BIN_WIDTH = 50  # equal width of every bin in the trimmed figure, in CAD

QUESTION = ("How much would you be willing to pay for a decent seat "
            "at a professional competition of this sport?")


def load_payment(path):
    """Read the encoded Q12 columns and validate them."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in ("label", "payment_amount", "payment_missing") if c not in df.columns]
    if missing:
        sys.exit(f"{path}: missing encoded columns {missing}; run data_preprocessing/encode_all.py first")
    name = os.path.basename(path).lower()
    if "validation" in name or "test" in name:
        sys.exit(f"{path}: figures that describe the data are drawn from the train split only")
    if df.payment_amount.isna().any():
        sys.exit(f"{path}: payment_amount has unanswered cells; the encoder should have filled them")
    if not df.payment_missing.isin([0, 1]).all():
        sys.exit(f"{path}: payment_missing must be 0 or 1")
    absent = [c for c in CLASS_ORDER if c not in set(df.label)]
    if absent:
        sys.exit(f"{path}: classes missing from data: {absent}")
    return df


def trim_cuts(df):
    """Per-class cut: Q3 + 1.5 * IQR of that class's parsed amounts."""
    cuts = {}
    for cls in CLASS_ORDER:
        amount = df.payment_amount[(df.label == cls) & (df.payment_missing == 0)]
        q1, q3 = amount.quantile([0.25, 0.75])
        cuts[cls] = float(q3 + TUKEY * (q3 - q1))
    return cuts


def kept_mask(df, cuts=None):
    """Rows whose amount enters the histogram. With cuts, the upper tail is dropped."""
    keep = (df.payment_missing == 0).to_numpy(copy=True)
    if cuts is not None:
        amount = df.payment_amount.to_numpy(dtype=float)
        label = df.label.to_numpy()
        for cls in CLASS_ORDER:
            keep &= ~((label == cls) & (amount > cuts[cls]))
    return keep


def bucket_codes(df, keep):
    """Map every row to a bucket code 0..8, or DROPPED for a trimmed amount."""
    codes = np.full(len(df), DROPPED, dtype=int)
    missing = (df.payment_missing == 1).to_numpy()
    codes[missing] = NOT_STATED
    amount = df.payment_amount.to_numpy(dtype=float)
    codes[keep & (amount <= 0)] = 0
    positive = keep & (amount > 0)
    codes[positive] = np.searchsorted(UPPER, amount[positive], side="left")
    return codes


def class_stats(df, keep, cuts=None):
    """Per-class statistics of the amounts that are actually drawn."""
    stats = {}
    for cls in CLASS_ORDER:
        rows = (df.label == cls).to_numpy()
        valid = df.payment_amount[rows & (df.payment_missing == 0).to_numpy()]
        kept = df.payment_amount[rows & keep]
        q1, q3 = valid.quantile([0.25, 0.75])
        stats[cls] = {
            "total": int(rows.sum()),
            "missing": int((rows & (df.payment_missing == 1).to_numpy()).sum()),
            "valid": int(len(valid)),
            "kept": int(len(kept)),
            "dropped": int(len(valid) - len(kept)),
            "cut": float(cuts[cls]) if cuts else float(valid.max()),
            "q1": float(q1),
            "q3": float(q3),
            "min": float(kept.min()),
            "max": float(kept.max()),
            "mean": float(kept.mean()),
            "var": float(kept.var(ddof=1)),
            "full_mean": float(valid.mean()),
            "full_var": float(valid.var(ddof=1)),
        }
    return stats


def fmt_money(value):
    return f"{value:,.0f}"


def fmt_sci(value):
    """1.19e6 style, so a raw-dollar variance does not break the title width."""
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def joined(stats, key, fmt):
    return " / ".join(fmt(s[key]) for s in stats.values())


def title_full(stats, fill_value):
    """Title of figure 10: no trimming."""
    total = stats[CLASS_ORDER[0]]["total"]
    iqr = " / ".join(f"{fmt_money(s['q1'])}-{fmt_money(s['q3'])}" for s in stats.values())
    return "\n".join([
        f"Q12 | {QUESTION}",
        "Short answer parsed to a CAD amount | 8 amount buckets + not stated   |   "
        f"Soccer / Baseball / Basketball (n = {total} per class)",
        'Unit: every amount is read as CAD (the dollar sign, "dollars" and "bucks" '
        "all mean CAD); no currency conversion",
        f"missing: {joined(stats, 'missing', lambda v: f'{v:.0f}')}      "
        f"fill value: {fill_value:.2f}",
        f"IQR (Q1-Q3): {iqr}      max: {joined(stats, 'max', fmt_money)}      "
        f"min: {joined(stats, 'min', fmt_money)}",
        f"mean: {joined(stats, 'mean', lambda v: f'{v:.2f}')}      "
        f"variance: {joined(stats, 'var', fmt_sci)}",
    ])


def trimmed_suptitle():
    """Shared heading of the trimmed figure."""
    return "\n".join([
        f"Q12 | {QUESTION}",
        "Upper tail dropped: inside each class, every amount above Q3 + 1.5 x IQR is "
        "removed; every amount from 0 up to that cut is kept, so the $0 answers stay in",
        "Each class has its own horizontal axis with equal 50 CAD bins, so bar width equals "
        "bin width; the axes are shared only in height (count)",
        "Amounts read as CAD, no currency conversion; the replies with no parseable amount "
        "are not plotted here and not trimmed (see figure 10)",
    ])


def panel_title(cls, s):
    """Per-panel heading, carrying that class's own statistics."""
    return "\n".join([
        f"{cls}  |  n = {s['total']}",
        f"kept {s['kept']} of {s['valid']} parsed, dropped {s['dropped']}, "
        f"no amount {s['missing']}",
        f"cut {fmt_money(s['cut'])} (Q3 + 1.5 x IQR)   |   "
        f"range {fmt_money(s['min'])}-{fmt_money(s['max'])}   |   "
        f"IQR {fmt_money(s['q1'])}-{fmt_money(s['q3'])}",
        f"mean {s['mean']:.2f}   |   variance {fmt_sci(s['var'])}",
    ])


def draw_trimmed(df, out_path, cuts, stats):
    """One panel per class, each with its own continuous x axis and 50 CAD bins."""
    keep = kept_mask(df, cuts)
    panels = {}
    for cls in CLASS_ORDER:
        kept = df.payment_amount[(df.label == cls).to_numpy() & keep]
        top = int(np.ceil(kept.max() / BIN_WIDTH)) * BIN_WIDTH
        edges = np.arange(0, top + BIN_WIDTH, BIN_WIDTH, dtype=float)
        counts, _ = np.histogram(kept, bins=edges)
        panels[cls] = (edges, counts)
    ymax = max(int(counts.max()) for _, counts in panels.values())

    fig, axes = plt.subplots(1, len(CLASS_ORDER), figsize=(14.6, 8.4), sharey=True)
    # top: 4 suptitle lines + 4 panel-title lines above the axes. Verified with
    # text bounding boxes; the old 0.79 let the two blocks touch.
    fig.subplots_adjust(top=0.775, bottom=0.105, left=0.055, right=0.985, wspace=0.10)
    fig.suptitle(trimmed_suptitle(), fontsize=10.5, linespacing=1.45, y=0.995, va="top")

    for ax, cls in zip(axes, CLASS_ORDER):
        edges, counts = panels[cls]
        centers = (edges[:-1] + edges[1:]) / 2
        # np.histogram puts the outermost edge in the last bin, so the class
        # maximum is never lost when it sits exactly on that edge.
        ax.bar(centers, counts, width=BIN_WIDTH * 0.92, color=COLORS[cls],
               edgecolor="white", linewidth=0.6, zorder=2)
        for x, c in zip(centers, counts):
            if c:
                ax.annotate(f"{c}\n{100 * c / stats[cls]['total']:.0f}%", (x, c),
                            ha="center", va="bottom", fontsize=7, linespacing=1.0)
        ax.set_xlim(-BIN_WIDTH * 0.06, edges[-1] + BIN_WIDTH * 0.06)
        ax.set_ylim(0, ymax * 1.24)
        ax.set_xticks(edges)
        ax.set_xticklabels([fmt_money(e) for e in edges], fontsize=8.5)
        ax.set_xlabel("Willingness to pay (CAD)", fontsize=9.5)
        ax.set_title(panel_title(cls, stats[cls]), fontsize=9.0, linespacing=1.4, pad=10)
        ax.annotate(f"not stated: {stats[cls]['missing']}\n(no amount, not trimmed)",
                    xy=(0.985, 0.965), xycoords="axes fraction", ha="right", va="top",
                    fontsize=8, color="#555555", linespacing=1.3,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#cccccc", alpha=0.9))
    axes[0].set_ylabel("Number of respondents (count)", fontsize=9.5)

    save(fig, out_path, dpi=200)
    plt.close(fig)
    return panels


def draw(df, out_path, cuts, stats, title, figsize, top):
    """Draw one categorical-bucket histogram of the kept amounts and save it."""
    keep = kept_mask(df, cuts)
    codes = bucket_codes(df, keep)
    values = {cls: [int(((df.label == cls).to_numpy() & (codes == b)).sum())
                    for b in range(len(LABELS))]
              for cls in CLASS_ORDER}
    # Percent below each bar is a share of the whole class, so a bar that lost
    # its upper tail simply shrinks instead of being renormalised.
    counts = {cls: int((df.label == cls).sum()) for cls in CLASS_ORDER}

    fig, ax = plt.subplots(figsize=figsize)
    # The statistics live in the title, so reserve room instead of relying on
    # the saved-file bounding box to rescue a tall title.
    fig.subplots_adjust(top=top, bottom=0.11, left=0.075, right=0.985)
    positions = np.arange(len(LABELS))
    # Mark the last bucket as "not an amount", drawn under the bars.
    ax.axvspan(NOT_STATED - 0.5, NOT_STATED + 0.5, color="#f2f2f2", zorder=0)
    ax.axvline(NOT_STATED - 0.5, color="#9a9a9a", linewidth=1.0, linestyle="--", zorder=0.5)

    ymax = grouped_hist(ax, positions, values, counts, annotate_pct=True, label_fontsize=6.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(LABELS, fontsize=9.5)
    ax.get_xticklabels()[NOT_STATED].set_color("#555555")
    ax.set_xlabel("Willingness to pay (CAD)", fontsize=9.5)
    ax.set_ylabel("Number of respondents (count)", fontsize=9.5)
    ax.set_title(title, fontsize=10.0, linespacing=1.45)
    finish(ax, ymax, title_pad=14)
    save(fig, out_path, dpi=200)
    plt.close(fig)
    return values, stats


def report(df, stats, path, trimmed):
    print(f"rows={len(df)}  " + ", ".join(f"{c}={int((df.label == c).sum())}" for c in CLASS_ORDER))
    for cls in CLASS_ORDER:
        s = stats[cls]
        cut = f"cut={s['cut']:8.1f} " if trimmed else ""
        print(f"  {cls:<10s} kept={s['kept']:3d} dropped={s['dropped']:2d} missing={s['missing']:2d} "
              f"{cut}mean={s['mean']:7.2f} var={s['var']:.2f}")
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description="Q12 payment histograms per class")
    ap.add_argument("--data", default="data/encoded/train_encoded.csv",
                    help="encoded train split with payment_amount / payment_missing")
    ap.add_argument("--out-dir", default="data_visualization/figures")
    args = ap.parse_args()

    setup_style()
    df = load_payment(args.data)
    filled = df.payment_amount[df.payment_missing == 1]
    fill_value = float(filled.iloc[0]) if len(filled) else float("nan")
    os.makedirs(args.out_dir, exist_ok=True)

    full_path = os.path.join(args.out_dir, "10_payment_Q12.png")
    stats = class_stats(df, kept_mask(df))
    draw(df, full_path, None, stats, title_full(stats, fill_value),
         figsize=(12.4, 8.8), top=0.78)
    report(df, stats, full_path, trimmed=False)

    trim_path = os.path.join(args.out_dir, "11_payment_Q12_trimmed.png")
    cuts = trim_cuts(df)
    stats = class_stats(df, kept_mask(df, cuts), cuts)
    draw_trimmed(df, trim_path, cuts, stats)
    report(df, stats, trim_path, trimmed=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Word counts for the five short-answer questions.

Normalisation: lowercase -> drop apostrophes -> repair thousands separators
("1,000" -> "1000") -> replace every other non-alphanumeric character with a
space -> split on spaces.

Results are printed to the terminal and written as CSV tables.

Usage (conda env csc311):
    conda activate csc311
    python analysis/text_word_counts.py
    python analysis/text_word_counts.py --min-len 2
    python analysis/text_word_counts.py --top 20
"""

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csv_output import write_text_tables          # noqa: E402
from survey_common import (                       # noqa: E402
    CLASSES,
    TEXT_SPECS,
    col_by_prefix,
    feature_quality,
    load,
    load_split_summary,
)

APOSTROPHES = re.compile("['\u2019`]")          # straight quote, curly quote, backtick
THOUSANDS = re.compile(r"(?<=\d),(?=\d)")
NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(text):
    """lowercase + punctuation handling, keeping the words that carry meaning."""
    s = str(text).strip().lower()
    s = APOSTROPHES.sub("", s)          # don't -> dont
    s = THOUSANDS.sub("", s)            # 1,000 -> 1000
    s = NON_ALNUM.sub(" ", s)           # punctuation -> word boundary
    return re.sub(r"\s+", " ", s).strip()


def tokenize(text, min_len=1):
    return [t for t in normalize(text).split() if len(t) >= min_len]


def count_words(df, label_col="label", min_len=1):
    """Word counts for every short-answer question.

    Returns (counts_rows, summary_rows, normalized_texts):
      counts_rows    long format, one row per (question, word)
      summary_rows   one row per question
      normalized     {question number: [normalised text per row]}
    """
    counts_rows, summary_rows = [], []
    normalized = {}

    for q, name, prefix, kind in TEXT_SPECS:
        col = col_by_prefix(df, prefix)
        raw = df[col].astype(str).str.strip()
        question = f"{q} {name}"

        blank_n, invalid_n, _valid_n, _ex = feature_quality(df, prefix, kind)
        per_class_counts = {c: Counter() for c in CLASSES}
        per_class_docs = {c: Counter() for c in CLASSES}
        total = Counter()
        docs = Counter()
        norm_texts = []
        n_tokens = 0
        n_rows_with_tokens = 0

        for idx, value in raw.items():
            norm = normalize(value)
            norm_texts.append(norm)
            toks = [t for t in norm.split() if len(t) >= min_len]
            if not toks:
                continue
            n_rows_with_tokens += 1
            n_tokens += len(toks)
            total.update(toks)
            docs.update(set(toks))
            cls = df.at[idx, label_col]
            if cls in per_class_counts:
                per_class_counts[cls].update(toks)
                per_class_docs[cls].update(set(toks))

        normalized[q] = norm_texts

        for word, cnt in total.most_common():
            counts_rows.append({
                "question": question,
                "word": word,
                "count": cnt,
                "share_in_question": cnt / n_tokens if n_tokens else 0.0,
                "n_rows_with_word": docs[word],
                "share_of_rows_with_word": docs[word] / len(df),
                **{f"count_{c}": per_class_counts[c][word] for c in CLASSES},
                **{f"n_rows_with_word_{c}": per_class_docs[c][word] for c in CLASSES},
            })

        summary_rows.append({
            "question": question,
            "column": col,
            "rows_total": len(df),
            "rows_blank": blank_n,
            "rows_invalid": invalid_n,
            "rows_with_tokens": n_rows_with_tokens,
            "tokens_total": n_tokens,
            "vocabulary_size": len(total),
            "singleton_words": sum(1 for w, c in total.items() if c == 1),
            "tokens_per_row": n_tokens / n_rows_with_tokens if n_rows_with_tokens else 0.0,
            "top_word": total.most_common(1)[0][0] if total else "",
            "top_word_count": total.most_common(1)[0][1] if total else 0,
        })

    return counts_rows, summary_rows, normalized


def print_result(summary_rows, counts_rows, top_n):
    for s in summary_rows:
        print(f"\n{s['question']}")
        print(f"  rows={s['rows_total']}  blank={s['rows_blank']}  invalid={s['rows_invalid']}  "
              f"tokens={s['tokens_total']}  distinct={s['vocabulary_size']}  "
              f"once={s['singleton_words']}  tokens/row={s['tokens_per_row']:.2f}")
        rows = [r for r in counts_rows if r["question"] == s["question"]][:top_n]
        for r in rows:
            print(f"  {r['word']:<20} {r['count']:>5}  {100 * r['share_in_question']:>5.1f}%  "
                  f"rows={r['n_rows_with_word']:<4} "
                  f"Soccer={r['count_Soccer']:<4} Baseball={r['count_Baseball']:<4} "
                  f"Basketball={r['count_Basketball']:<4}")


def main():
    ap = argparse.ArgumentParser(description="word counts for the short-answer questions")
    ap.add_argument("--data", default="data/splits/train.csv")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--min-len", type=int, default=1)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--split-summary", default="data/splits/split_summary.json")
    args = ap.parse_args()

    df = load(args.data)
    split_info = load_split_summary(args.split_summary)
    print(f"{args.data}  rows={len(df)}  min_len={args.min_len}")
    if split_info:
        print("  ".join(f"{n}={split_info['splits'][n]['rows']}"
                        for n in ("train", "val", "test")))

    counts_rows, summary_rows, normalized = count_words(df, args.label_col, args.min_len)
    print_result(summary_rows, counts_rows, args.top)

    out_dir = args.out_dir or os.path.join("analysis", "out", Path(args.data).stem)
    paths = write_text_tables(counts_rows, summary_rows, normalized, df, out_dir, args.label_col)
    print()
    for path in paths.values():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

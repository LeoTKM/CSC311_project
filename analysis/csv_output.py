#!/usr/bin/env python3
"""
CSV output: writes the analysed data as tidy tables.
"""

import os

import pandas as pd


def write_categorical_tables(result, out_dir):
    """Write the three tidy CSV tables of the categorical analysis."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    single_df = pd.DataFrame(result["single_rows"])
    if not single_df.empty:
        paths["single"] = os.path.join(out_dir, "single_choice_stats.csv")
        single_df.to_csv(paths["single"], index=False, encoding="utf-8")

    multi_df = pd.DataFrame(result["multi_rows"])
    if not multi_df.empty:
        paths["multi"] = os.path.join(out_dir, "multi_select_stats.csv")
        multi_df.to_csv(paths["multi"], index=False, encoding="utf-8")

    paths["count"] = os.path.join(out_dir, "selection_count_distribution.csv")
    pd.DataFrame(result["count_rows"]).to_csv(paths["count"], index=False, encoding="utf-8")
    return paths


def write_text_tables(counts_rows, summary_rows, normalized, df, out_dir, label_col="label"):
    """Write the word-count tables and the per-row normalised text."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    paths["counts"] = os.path.join(out_dir, "text_token_counts.csv")
    pd.DataFrame(counts_rows).to_csv(paths["counts"], index=False, encoding="utf-8")

    paths["summary"] = os.path.join(out_dir, "text_token_summary.csv")
    pd.DataFrame(summary_rows).to_csv(paths["summary"], index=False, encoding="utf-8")

    norm_df = pd.DataFrame({q: v for q, v in normalized.items()})
    norm_df.insert(0, label_col, df[label_col].values)
    if "id" in df.columns:
        norm_df.insert(0, "id", df["id"].values)
    paths["normalized"] = os.path.join(out_dir, "text_normalized.csv")
    norm_df.to_csv(paths["normalized"], index=False, encoding="utf-8")
    return paths

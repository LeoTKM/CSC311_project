#!/usr/bin/env python3
"""
Shared definitions and helpers for the survey analysis (no analysis, no report).

This module holds everything that both the analysis modules and the report writer
need to agree on:

  * the question catalogues (which CSV column belongs to which question)
  * the encodings (Likert -> ordinal, multi-select vocabularies)
  * the data-quality rules (what counts as blank / invalid)
  * CSV loading that keeps real answers such as "None" / "N/A" intact
  * small numeric helpers (describe, Cramer's V, eta^2)
"""

import json
import os
import re
import sys
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd


CLASSES = ["Soccer", "Baseball", "Basketball"]

# 5-point Likert -> ordinal score (the real-number map used everywhere below).
LIKERT_SCALE = OrderedDict([
    ("Strongly disagree", 1),
    ("Disagree", 2),
    ("Neutral", 3),
    ("Agree", 4),
    ("Strongly agree", 5),
])

# Multi-select vocabularies exactly as they appear in the CSV.
MULTI_VOCAB = OrderedDict([
    ("Q5 venue", [
        "Indoor facility",
        "Outdoor public space",
        "School or university facility",
        "Community recreation facility",
        "Private sports facility",
        "At home or nearby",
        "Others",
    ]),
    ("Q6 season", ["Winter", "Spring", "Summer", "Fall"]),
    ("Q7 music type", [
        "Pop",
        "Electronic or dance",
        "Latin",
        "Folk or traditional",
        "Classical or jazz",
        "Hip-hop or rap",
        "Country",
        "Rock",
        "No particular type",
        "Other",
    ]),
])
MULTI_VOCAB_BY_Q = {k.split()[0]: v for k, v in MULTI_VOCAB.items()}

SEASON_INDEX = {"Winter": 1, "Spring": 2, "Summer": 3, "Fall": 4}

# (question number, short English name, header prefix of the CSV column)
SINGLE_SPECS = [
    ("Q2", "strategy dependence", "Success in this sport depends heavily"),
    ("Q3", "injury risk", "Players must be comfortable with a high risk of injury"),
    ("Q4", "individual brilliance", "Individual brilliance can outweigh teamwork"),
]
MULTI_SPECS = [
    ("Q5", "venue", "Where might you play this sport?"),
    ("Q6", "season", "During which seasons"),
    ("Q7", "music type", "What types of music do you most associate"),
]
TEXT_SPECS = [
    ("Q8", "food and drinks", "What foods and drinks", "text"),
    ("Q9", "colour", "What colour do you associate", "text"),
    ("Q10", "song", "Which song do you associate", "text"),
    ("Q11", "feeling", "How do you feel when you watch", "text"),
    ("Q12", "willingness to pay", "How much would you be willing to pay", "price"),
]

QUALITY_SPECS = [
    ("Q1", "Ease of starting (0-10 numeric scale)", "How easy do you think it would be", "scale"),
    ("Q2", "Strategy dependence (5-point Likert)", "Success in this sport depends heavily", "likert"),
    ("Q3", "High injury risk (5-point Likert)", "Players must be comfortable with a high risk", "likert"),
    ("Q4", "Individual brilliance over teamwork (5-point Likert)", "Individual brilliance can outweigh teamwork", "likert"),
    ("Q5", "Venue (multi-select)", "Where might you play this sport?", "multi"),
    ("Q6", "Season (multi-select)", "During which seasons", "multi"),
    ("Q7", "Music type (multi-select)", "What types of music do you most associate", "multi"),
    ("Q8", "Food and drinks (short answer)", "What foods and drinks", "text"),
    ("Q9", "Colour (short answer)", "What colour do you associate", "text"),
    ("Q10", "Song (short answer)", "Which song do you associate", "text"),
    ("Q11", "Feeling (short answer)", "How do you feel when you watch", "text"),
    ("Q12", "Willingness to pay (short answer, amount/text mix)", "How much would you be willing to pay", "price"),
]
QUALITY_KIND_LABEL = {"scale": "numeric scale", "likert": "single choice", "multi": "multi-select",
                      "text": "short answer", "price": "short answer (amount)"}

# A free-text reply counts as "no information" if it normalises to one of these.
# The survey is answered in English, so only English forms are listed.
NON_ANSWER = {
    "none", "n/a", "na", "n.a.", "nothing", "not sure", "unsure", "no idea",
    "null", "nil", "nope", "-", "--", "none at all",
}

# Amounts spelled out in words ("Forty five dollars", "couple hundred bucks") and
# replies that clearly express unwillingness (=> willingness 0) are parseable prices.
NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty",
    "seventy", "eighty", "ninety", "hundred", "thousand", "million", "couple", "few",
}
UNWILLING = ("not willing", "not really willing", "unwilling", "would not pay",
             "wouldn't pay", "wouldnt pay", "won't pay", "wont pay", "not pay",
             "no interest", "zero", "none", "nothing", "never pay")


def load(path):
    """Read a survey CSV, keeping every answer as text.

    keep_default_na=False is essential: pandas would otherwise turn real answers
    such as "None" / "N/A" into NaN before any analysis can see them.
    """
    return pd.read_csv(path, encoding="utf-8-sig",
                       keep_default_na=False, na_values=[])


def col_by_prefix(df, prefix):
    """Resolve a question header from a unique prefix of its wording."""
    hits = [c for c in df.columns if c.startswith(prefix)]
    if len(hits) != 1:
        sys.exit(f"prefix {prefix!r} matched {len(hits)} columns: {hits}")
    return hits[0]


def load_split_summary(path):
    """Read the train/val/test manifest produced by make_splits.py (or None)."""
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def describe(values):
    """n, mean, sample variance (ddof=1), sd of the finite values."""
    a = np.asarray(values, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return 0, np.nan, np.nan, np.nan
    mu = a.mean()
    var = a.var(ddof=1) if a.size > 1 else np.nan
    return a.size, mu, var, (np.sqrt(var) if var == var else np.nan)


def cramers_v(table):
    """Cramer's V of a contingency table (rows = categories, cols = classes).

    The chi-square statistic is only an intermediate step here and is not returned:
    this analysis reports effect sizes only.
    """
    t = np.asarray(table, dtype=float)
    if t.shape[0] < 2 or t.shape[1] < 2:
        return np.nan
    total = t.sum()
    if total == 0:
        return np.nan
    expected = np.outer(t.sum(axis=1), t.sum(axis=0)) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.where(expected > 0, (t - expected) ** 2 / expected, 0.0).sum())
    return float(np.sqrt(chi2 / (total * (min(t.shape) - 1))))


def eta_sq(groups):
    """Share of total variance explained by the group means."""
    allv = np.concatenate([np.asarray(g, dtype=float) for g in groups if len(g)])
    if allv.size < 2:
        return np.nan
    grand = allv.mean()
    ss_tot = ((allv - grand) ** 2).sum()
    if ss_tot == 0:
        return np.nan
    ss_b = sum(len(g) * (np.asarray(g, float).mean() - grand) ** 2 for g in groups if len(g))
    return float(ss_b / ss_tot)


def norm_text(text):
    """lowercase, strip surrounding whitespace and trailing punctuation.

    The survey is answered in English, so only ASCII punctuation is handled here;
    any other non-alphanumeric character is treated as a word boundary by the
    tokeniser (see normalize() in text_word_counts.py).
    """
    return re.sub(r"[\s.!?,;:]+$", "", str(text).strip().lower())


def feature_quality(df, prefix, kind, vocab=None):
    """(blank, invalid, valid, invalid_examples) for one feature column.

    blank   : the cell is empty (question not answered)
    invalid : the answer cannot be used under that question type's rules
    valid   : rows - blank - invalid
    """
    col = col_by_prefix(df, prefix)
    raw = df[col].astype(str).str.strip()
    blank = raw.eq("")

    if kind == "scale":                       # Q1: 0-10 numeric rating
        parsed = pd.to_numeric(raw.str.extract(r"^(\d+(?:\.\d+)?)$")[0], errors="coerce")
        invalid = (~blank) & (parsed.isna() | (parsed < 0) | (parsed > 10))
    elif kind == "likert":                    # Q2-Q4
        invalid = (~blank) & (~raw.isin(list(LIKERT_SCALE)))
    elif kind == "multi":                     # Q5-Q7: closed vocabulary
        vocab_set = set(vocab or [])

        def bad(s):
            return bool({t.strip() for t in s.split(",") if t.strip()} - vocab_set)
        invalid = (~blank) & raw.map(bad)
    elif kind == "price":                     # Q12: needs a parseable amount
        low = raw.str.lower()
        has_digit = low.str.contains(r"\d", regex=True)
        has_word = low.map(lambda s: bool(set(re.findall(r"[a-z]+", s)) & NUMBER_WORDS))
        has_unwilling = low.map(lambda s: any(k in s for k in UNWILLING))
        invalid = (~blank) & ~(has_digit | has_word | has_unwilling)
    else:                                     # Q8-Q11 free text: "no information" replies
        def non_answer(s):
            t = norm_text(s)
            return t in NON_ANSWER or norm_text(t.split(",")[0]) in NON_ANSWER
        invalid = (~blank) & raw.map(non_answer)

    blank_mask, invalid_mask = blank, invalid
    examples = (Counter(raw[invalid_mask]).most_common(3)
                if bool(invalid_mask.any()) else [])
    blank_n, invalid_n = int(blank_mask.sum()), int(invalid_mask.sum())
    return blank_n, invalid_n, len(df) - blank_n - invalid_n, examples

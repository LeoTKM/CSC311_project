"""Tokenizer report for v2. Diagnostic only: trains nothing, writes nothing.

Shows which answers each rule touches, what the vocabulary looks like, and which
degenerate tokens the tokenizer still emits before any noise removal.
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'train' / 'random_forest_v2'))
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'data_preprocessing'))

import text as v2text                                     # noqa: E402
from encode_all import TEXT, TOKEN_PATTERN, clean, raw_headers   # noqa: E402

APOSTROPHE = v2text.APOSTROPHE
CJK = re.compile(r'[\u3000-\u9fff\u3040-\u30ff\uff00-\uffef]')
NON_ASCII = re.compile(r'[^\x00-\x7f]')


def current_tokens(value):
    """v1 tokenizer, for side-by-side comparison only."""
    return re.findall(TOKEN_PATTERN, clean(value))


def main():
    parser = argparse.ArgumentParser(description='v2 tokenizer report')
    parser.add_argument('--hyphen', default='space', choices=list(v2text.POLICIES))
    parser.add_argument('--min-len', type=int, default=1)
    parser.add_argument('--require-letter', action='store_true')
    args = parser.parse_args()
    kwargs = dict(hyphen=args.hyphen, min_len=args.min_len,
                  require_letter=args.require_letter)

    train = pd.read_csv(ROOT / 'data/splits/train.csv', dtype=str, keep_default_na=False)
    df = raw_headers(train)
    print(f'rows {len(df)}  respondents {df.id.nunique()}  '
          f'hyphen={args.hyphen} min_len={args.min_len} require_letter={args.require_letter}\n')

    print('=== rows touched by each rule ===')
    for field in TEXT:
        raw = df[TEXT[field]].astype(str)
        print(f'  {field:9s} apostrophe {int(raw.str.contains(APOSTROPHE).sum()):3d}'
              f'   hyphen {int(raw.str.contains("-").sum()):3d}'
              f'   full-width {int(raw.map(lambda s: v2text.unicodedata.normalize("NFKC", s) != s).sum()):3d}'
              f'   CJK {int(raw.str.contains(CJK).sum()):3d}'
              f'   other non-ASCII {int(raw.map(lambda s: bool(NON_ASCII.search(CJK.sub("", v2text.unicodedata.normalize("NFKC", s))))).sum()):3d}')

    print('\n=== vocabulary per field ===')
    for field in TEXT:
        counts, docs = Counter(), Counter()
        for value in df[TEXT[field]]:
            got = v2text.tokens(clean(value), **kwargs)
            counts.update(got)
            docs.update(set(got))
        repeated = sum(1 for c in docs.values() if c >= 2)
        print(f'  {field:9s} tokens {sum(counts.values()):5d}  distinct {len(counts):4d}  '
              f'df>=2 {repeated:4d}  singletons {len(counts) - repeated:4d}')
        print(f'            top 15: {[t for t, _ in counts.most_common(15)]}')

    print('\n=== degenerate tokens still present before noise removal ===')
    for field in TEXT:
        counts = Counter()
        for value in df[TEXT[field]]:
            counts.update(v2text.tokens(clean(value), **kwargs))
        lone = {t: c for t, c in counts.items() if set(t) <= {'-'}}
        digit = {t: c for t, c in counts.items() if not re.search(r'[a-z]', t)}
        single = {t: c for t, c in counts.items() if len(t) == 1}
        print(f'  {field:9s} lone hyphen {lone} | digit-only {len(digit)} {list(digit)[:6]}'
              f' | single char {len(single)} {list(single)[:6]}')

    print('\n=== sanity examples: v1 tokenizer vs v2 ===')
    for value in ["I don't know \u2014 maybe Waka Waka",
                  'Waka Waka (This Time for Africa) - Shakira',
                  'warm-hearted, high-energy',
                  'Green\uff0cbecause it is the field colour\u3002',
                  'popcorn\uff0c\uff43\uff4f\uff4b\uff45',
                  'coca-cola/pepsi',
                  '2000s hip hop / EDM',
                  'hot-dog, hotdog, hot dog',
                  '\u975e\u5e38\u6fc0\u52a8']:
        print(f'  raw : {value!r}')
        print(f'  v1  : {current_tokens(value)}')
        print(f'  v2  : {v2text.tokens(clean(value), **kwargs)}')

    print('\n=== answers that yield no tokens ===')
    for field in TEXT:
        blank = df[TEXT[field]].map(clean).eq('')
        now = df[TEXT[field]].map(current_tokens).map(len).eq(0) & ~blank
        new = pd.Series([not v2text.tokens(clean(v), **kwargs) for v in df[TEXT[field]]],
                        index=df.index) & ~blank
        print(f'  {field:9s} blank {int(blank.sum()):3d}  answered but no tokens: '
              f'v1 {int(now.sum()):3d}  v2 {int(new.sum()):3d}')


if __name__ == '__main__':
    main()

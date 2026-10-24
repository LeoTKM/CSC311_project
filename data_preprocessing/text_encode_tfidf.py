#!/usr/bin/env python3
"""Encode four text questions with TF-IDF and extract/impute payment amounts.

Install: python3 -m pip install pandas scikit-learn joblib
Fit ONLY on the training split:
  python3 text_encode.py fit train.csv --output train_text.csv --model text_model.joblib
Reuse on validation/test (never fit again):
  python3 text_encode.py transform validation.csv --output validation_text.csv --model text_model.joblib

Works with original headers, earlier encoding-script outputs, or the short headers
in the previously supplied split files. Other columns and row order are preserved.
Defaults: at most 20 colour terms and 50 terms per other text question, min_df=2, training-mean payment
imputation. Use --payment-impute median during fit to change the imputation rule.
Currency labels are ignored: all amounts are assumed to share the same currency.
Chinese tokens are excluded explicitly by an A-Z-only token pattern; rows remain.
"""
import argparse
import csv
import math
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

TEXT = {
    'food': 'What foods and drinks do you associate with watching this sport in the place or culture you know best?',
    'colour': 'What colour do you associate with this sport?',
    'song': 'Which song do you associate with this sport?',
    'feeling': 'How do you feel when you watch this sport?',
}
PAYMENT = 'How much would you be willing to pay for a decent seat at a professional competition of this sport?'
MISSING = {'', 'n/a', 'na', 'nan', 'null', '-', '[]'}
# Unicode word boundaries exclude non-ASCII words rather than taking fragments.
# This is an alphabet filter, not English-language detection (e.g. 'bonjour' passes).
TOKEN_PATTERN = r'(?u)\b[a-zA-Z]{2,}\b'
NUMBER = re.compile(r'(?<![\w.])\d+(?:\.\d+)?(?:\s*k)?(?![\d.])')


def clean(value):
    if pd.isna(value):
        return ''
    text = ' '.join(str(value).split()).lower()
    return '' if text in MISSING else text


def parse_payment(value):
    """Return (amount or None, audit status). This is a simple numeric parser.

    Accept one number, or exactly two joined as a range. A single value preceded
    by 'at most'/'under' is treated as the stated amount. Multiple scenarios are
    rejected. Word-only amounts (including 'zero') are missing, per numeric-only
    policy. Explicit numeric zero is preserved. No exchange-rate conversion.
    """
    text = clean(value).replace('–', '-').replace('—', '-')
    if not text:
        return None, 'blank'
    if re.search(r'\bscale\b|order of magnitude|\d+s\b|\b(?:hundreds?|thousands?|millions?|billions?)\b', text):
        return None, 'ambiguous_or_not_a_price'
    # '1,000' / '10 000' are thousands; '200, 500' remains two separate values.
    text = re.sub(r'(?<=\d)[, ](?=\d{3}(?:\D|$))', '', text)
    # Remove currency notation without converting it, as explicitly assumed.
    text = re.sub(r'(?<![a-z])(?:cad|usd|eur|gbp|cny|rmb)(?![a-z])', ' ', text)
    text = re.sub(r'\b(?:canadian|us|dollars?|euros?|bucks|yuan)\b', ' ', text)
    text = re.sub(r'[$€£¥]', ' ', text)
    matches = list(NUMBER.finditer(text))
    if not matches:
        return None, 'no_numeric_amount'
    def number(match):
        token = match.group().strip()
        return float(token[:-1]) * 1000 if token.endswith('k') else float(token)
    if len(matches) == 1:
        # A leading minus sign is a negative price, not a valid amount.
        if text[:matches[0].start()].rstrip().endswith('-'):
            return None, 'negative_amount'
        amount = number(matches[0])
        status = 'single_amount'
    elif len(matches) == 2 and re.fullmatch(r'\s*(?:-|to)\s*', text[matches[0].end():matches[1].start()]):
        if text[:matches[0].start()].rstrip().endswith('-'):
            return None, 'negative_amount'
        amount = (number(matches[0]) + number(matches[1])) / 2
        status = 'range_midpoint'
    else:
        return None, 'multiple_or_ambiguous_amounts'
    if not math.isfinite(amount):
        return None, 'invalid_amount'
    return amount, status


def resolve_columns(df):
    resolved = {}
    for short, full in {**TEXT, 'payment': PAYMENT}.items():
        matches = [c for c in (full, short) if c in df.columns]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one column named {full!r} or {short!r}.')
        resolved[short] = matches[0]
    return resolved


def encode(df, model=None, max_features=50, min_df=2, payment_impute='mean', colour_max_features=20):
    columns = resolve_columns(df)
    fitting = model is None
    if fitting:
        model = {
            'version': 1, 'vectorizers': {}, 'payment_impute': payment_impute,
            'feature_caps': {field: colour_max_features if field == 'colour' else max_features for field in TEXT},
            'min_df': min_df,
        }
    else:
        if model.get('version') != 1:
            raise ValueError('Unrecognized saved encoder.')
    result = df.drop(columns=list(columns.values())).copy()
    blocks = []
    for short in TEXT:
        texts = df[columns[short]].map(clean)
        if fitting:
            vectorizer = TfidfVectorizer(
                lowercase=True, token_pattern=TOKEN_PATTERN, ngram_range=(1, 1),
                max_features=colour_max_features if short == 'colour' else max_features, min_df=min_df,
                sublinear_tf=True, smooth_idf=True, norm='l2', stop_words=None,
            )
            # Deliberately omit an empty vocabulary without relaxing min_df.
            try:
                vectorizer.fit(texts)
            except ValueError as error:
                message = str(error).lower()
                if not any(s in message for s in ['empty vocabulary', 'no terms remain', 'max_df corresponds to < documents than min_df']):
                    raise
                vectorizer = None
            model['vectorizers'][short] = vectorizer
        vectorizer = model['vectorizers'][short]
        if vectorizer is not None:
            matrix = vectorizer.transform(texts)
            names = [f'{short}_tfidf_{term}' for term in vectorizer.get_feature_names_out()]
            blocks.append(pd.DataFrame(matrix.toarray(), columns=names, index=df.index))
        # Chinese-only text is not a missing response; its missing flag stays zero.
        blocks.append(pd.DataFrame({f'{short}_text_missing': texts.eq('').astype(int)}, index=df.index))

    parsed = [parse_payment(value) for value in df[columns['payment']]]
    amounts = pd.Series([amount for amount, _ in parsed], index=df.index, dtype=float)
    if fitting:
        valid = amounts.dropna()
        if valid.empty:
            raise ValueError('No valid training payment amounts; cannot calculate an imputation value.')
        statistic = valid.mean() if payment_impute == 'mean' else valid.median()
        model['payment_fill_value'] = float(statistic)
        model['training_rows'] = len(df)
    payment_block = pd.DataFrame({
        'payment_amount': amounts.fillna(model['payment_fill_value']),
        'payment_missing': amounts.isna().astype(int),
    }, index=df.index)
    blocks.append(payment_block)
    new_names = [name for block in blocks for name in block.columns]
    if set(new_names) & set(result.columns):
        raise ValueError('Encoded output columns already exist in the input.')
    result = pd.concat([result] + blocks, axis=1)
    audit = pd.DataFrame({
        'source_record': range(1, len(df) + 1),
        'original_payment': df[columns['payment']],
        'parsed_amount': amounts,
        'status': [status for _, status in parsed],
        'payment_missing': payment_block.payment_missing,
        'final_amount': payment_block.payment_amount,
    }, index=df.index)
    return result, model, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['fit', 'transform'])
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path,
                        help='Save fitted encoder here, or load it in transform mode')
    parser.add_argument('--max-features', type=int, default=50, help='Fit only: cap for food, song and feeling')
    parser.add_argument('--colour-max-features', type=int, default=20, help='Fit only: cap for colour terms')
    parser.add_argument('--min-df', type=int, default=2, help='Fit only: minimum training document count')
    parser.add_argument('--payment-impute', choices=['mean', 'median'], default='mean', help='Fit only')
    args = parser.parse_args()
    audit_path = args.output.with_name(args.output.stem + '_payment_audit.csv')
    try:
        if args.max_features < 1 or args.colour_max_features < 1 or args.min_df < 1:
            raise ValueError('Feature caps and min-df must be positive integers.')
        targets = [args.output, audit_path] + ([args.model] if args.mode == 'fit' else [])
        # Prevent output collisions, including overwriting the loaded model.
        paths = [args.input.resolve(), args.model.resolve(), args.output.resolve(), audit_path.resolve()]
        if len(paths) != len(set(paths)):
            raise ValueError('Input, model, output and audit paths must be distinct.')
        for target in targets:
            if target.exists():
                raise FileExistsError(f'Will not overwrite existing file: {target}')
            if not target.parent.is_dir():
                raise FileNotFoundError(f'Output directory does not exist: {target.parent}')
        with args.input.open(newline='', encoding='utf-8-sig') as source:
            headers = next(csv.reader(source))
        if len(headers) != len(set(headers)):
            raise ValueError('Input contains duplicate column names.')
        df = pd.read_csv(args.input, encoding='utf-8-sig', keep_default_na=False, dtype=str)
        if df.empty:
            raise ValueError('Input CSV has no rows.')
        # Load only joblib files you trust. Fitted statistics never change in transform mode.
        model = None if args.mode == 'fit' else joblib.load(args.model)
        result, model, audit = encode(df, model, args.max_features, args.min_df, args.payment_impute, args.colour_max_features)
        if args.mode == 'fit':
            with args.model.open('xb') as destination:
                joblib.dump(model, destination)
        result.to_csv(args.output, index=False, mode='x')
        audit.to_csv(audit_path, index=False, mode='x')
    except (OSError, ValueError, StopIteration) as error:
        parser.exit(1, f'Error: {error}\n')
    print(f'Wrote {len(result)} rows to {args.output}')
    print('TF-IDF terms per field:', {k: len(v.vocabulary_) if v is not None else 0 for k,v in model['vectorizers'].items()})
    print(f"Payment imputation: training {model['payment_impute']} = {model['payment_fill_value']:.4f}")
    print(f'Payment audit: {audit_path}')


if __name__ == '__main__':
    main()

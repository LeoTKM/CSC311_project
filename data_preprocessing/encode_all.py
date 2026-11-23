#!/usr/bin/env python3
"""Run all CSC311 encodings in one standalone script.

Full CSV, new respondent-grouped split:
  python data_preprocessing/encode_all.py --input data/class_dataset.csv
Existing raw splits (preferred if your team already has them):
  python data_preprocessing/encode_all.py --train data/splits/train.csv --validation data/splits/val.csv --test data/splits/test.csv
Later prediction data:
  python data_preprocessing/encode_all.py --transform new.csv --model data/encoded/preprocessor.joblib --output data/encoded/new_encoded.csv

Default output directory: data/encoded, relative to your current working directory.
Dependencies: pandas, numpy, scikit-learn, joblib. The other three scripts are not
needed. Fit text vocabularies and payment imputation only on training responses.
"""
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

EASE_COLUMN = (
    'How easy do you think it would be for someone with no prior experience '
    'to start playing this sport recreationally? Rate it on a scale from 0 to 10, '
    'where 0 means extremely difficult and 10 means extremely easy.'
)
AGREEMENT_COLUMNS = (
    'Success in this sport depends heavily on strategy and planning.',
    'Players must be comfortable with a high risk of injury.',
    'Individual brilliance can outweigh teamwork in this sport.',
)
AGREEMENT_MAP = {
    'strongly disagree': '0',
    'disagree': '0.25',
    'neutral': '0.5',
    'agree': '0.75',
    'strongly agree': '1',
}

SCHEMA = {
    'Where might you play this sport?': [
        ('Indoor facility', 'location_indoor'),
        ('Outdoor public space', 'location_outdoor'),
        ('School or university facility', 'location_school'),
        ('Community recreation facility', 'location_community'),
        ('Private sports facility', 'location_private'),
        ('At home or nearby', 'location_home'),
        ('Others', 'location_other'),
    ],
    'During which seasons would you be most likely to play this sport recreationally in Toronto?': [
        ('Winter', 'season_winter'),
        ('Summer', 'season_summer'),
        ('Fall', 'season_fall'),
        ('Spring', 'season_spring'),
    ],
    'What types of music do you most associate with this sport?': [
        ('Pop', 'music_pop'),
        ('Electronic or dance', 'music_electronic_dance'),
        ('Latin', 'music_latin'),
        ('Folk or traditional', 'music_folk_traditional'),
        ('Classical or jazz', 'music_classical_jazz'),
        ('Hip-hop or rap', 'music_hiphop_rap'),
        ('Country', 'music_country'),
        ('Rock', 'music_rock'),
        ('No particular type', 'music_no_particular_type'),
        ('Other', 'music_other'),
    ],
}

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


def encode_text_payment(df, model=None, max_features=50, min_df=2, payment_impute='mean', colour_max_features=20):
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


# Accept both the original CSV headers and the short headers in earlier splits.
FULL_COLUMNS = [EASE_COLUMN, *AGREEMENT_COLUMNS, *SCHEMA, *TEXT.values(), PAYMENT]
SHORT_COLUMNS = ['ease', 'strategy', 'injury', 'individual', 'location', 'season',
                 'music_type', 'food', 'colour', 'song', 'feeling', 'payment']


def read_csv(path):
    """Read all cells as strings so IDs/labels are preserved exactly."""
    with path.open(newline='', encoding='utf-8-sig') as source:
        reader = csv.reader(source)
        headers = next(reader, [])
        if not headers or len(headers) != len(set(headers)):
            raise ValueError(f'{path}: missing or duplicate headers.')
        for record, row in enumerate(reader, start=1):
            if len(row) != len(headers):
                raise ValueError(f'{path}: record {record} has the wrong number of cells.')
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    if df.empty:
        raise ValueError(f'{path}: no rows.')
    return df


def raw_headers(df):
    renamed = {}
    for short, full in zip(SHORT_COLUMNS, FULL_COLUMNS):
        if short in df and full in df:
            raise ValueError(f'Both {short!r} and its original survey column are present.')
        if full not in df:
            if short not in df:
                raise ValueError(f'Missing raw question column: {full!r}. Start with a raw CSV, not an encoded one.')
            renamed[short] = full
    return df.rename(columns=renamed).copy()


def encode_structured(df):
    """Q1-Q4 ordinal values and Q5-Q7 multi-hot values; missing stays blank."""
    df = raw_headers(df)
    values = []
    for record, answer in enumerate(df[EASE_COLUMN], start=1):
        answer = clean(answer)
        if not answer:
            values.append(np.nan)
            continue
        try:
            value = float(answer)
        except ValueError:
            raise ValueError(f'Record {record}: invalid ease value {answer!r}.') from None
        if not math.isfinite(value) or not 0 <= value <= 10:
            raise ValueError(f'Record {record}: ease must be between 0 and 10.')
        values.append(value / 10)
    df[EASE_COLUMN] = values
    for column in AGREEMENT_COLUMNS:
        values = []
        for record, answer in enumerate(df[column], start=1):
            answer = clean(answer)
            if not answer:
                values.append(np.nan)
            elif answer in AGREEMENT_MAP:
                values.append(float(AGREEMENT_MAP[answer]))
            else:
                raise ValueError(f'Record {record}: unknown agreement answer {answer!r}.')
        df[column] = values
    for column, choices in SCHEMA.items():
        names = [name for _, name in choices]
        if set(names) & set(df.columns):
            raise ValueError('Multi-hot columns already exist. Use the raw CSV.')
        allowed = {clean(option) for option, _ in choices}
        encoded = []
        for record, answer in enumerate(df[column], start=1):
            answer = clean(answer)
            if not answer:
                encoded.append([np.nan] * len(choices))
                continue
            selected = {part.strip() for part in answer.split(',') if part.strip()}
            if not selected or selected - allowed:
                raise ValueError(f'Record {record}: unknown multi-select option(s) {selected - allowed} in {column!r}.')
            encoded.append([int(clean(option) in selected) for option, _ in choices])
        position = df.columns.get_loc(column)
        block = pd.DataFrame(encoded, columns=names, index=df.index)
        df = pd.concat([df.iloc[:, :position], block, df.iloc[:, position+1:]], axis=1)
    return df


def encode_all(df, artifact=None):
    structured = encode_structured(df)
    if artifact is not None and artifact.get('artifact_kind') != 'csc311_all_encoders_v1':
        raise ValueError('Use a preprocessor.joblib created by this encode_all.py script.')
    text_model = None if artifact is None else artifact['text_encoder']
    encoded, text_model, audit = encode_text_payment(structured, model=text_model)
    feature_columns = [c for c in encoded if c not in ('id', 'label')]
    # No string survey responses should remain among the feature columns.
    encoded[feature_columns] = encoded[feature_columns].apply(pd.to_numeric, errors='raise')
    if artifact is None:
        artifact = {'artifact_kind': 'csc311_all_encoders_v1',
                    'text_encoder': text_model, 'feature_columns': feature_columns,
                    'training_rows': len(df), 'sklearn_version': sklearn.__version__}
    else:
        if set(feature_columns) != set(artifact['feature_columns']):
            raise ValueError('Prediction feature columns do not match the saved training schema.')
        # Preserve metadata if present, and guarantee training feature order.
        metadata = [c for c in encoded if c in ('id', 'label')]
        encoded = encoded[metadata + artifact['feature_columns']]
    metadata = [c for c in encoded if c in ('id', 'label')]
    encoded = encoded[metadata + artifact['feature_columns']]
    return encoded, artifact, audit


def check_groups(parts):
    """Refuse row-wise splits that put the same respondent in multiple sets."""
    seen = set()
    for name, df in parts.items():
        if 'id' not in df or df.id.map(clean).eq('').any():
            raise ValueError(f'{name}: a nonmissing id is required for respondent-level splitting.')
        if 'label' not in df or df.label.map(clean).eq('').any():
            raise ValueError(f'{name}: a nonmissing label is required.')
        groups = set(df.id)
        if groups & seen:
            raise ValueError(f'{name}: respondent IDs overlap with another split. Keep each person in one split.')
        seen |= groups
    if any(set(df.label) - set(parts['train'].label) for df in parts.values()):
        raise ValueError('A held-out split contains a label absent from training.')


def grouped_split(df, seed):
    if 'id' not in df or df.id.map(clean).eq('').any():
        raise ValueError('The input needs a nonmissing respondent id column.')
    ids = df.id.unique().tolist()
    # Match analysis/make_splits.py exactly: lexical order, then test/val/train.
    # A seed alone is insufficient when ordering and allocation differ.
    ids = sorted(ids, key=str)
    if len(ids) < 3:
        raise ValueError('At least three respondents are needed for three splits.')
    ids = np.random.default_rng(seed).permutation(ids)
    held_out = max(1, int(round(.15 * len(ids))))
    n_train = len(ids) - 2 * held_out
    if n_train < 1:
        raise ValueError('Not enough respondents for this split.')
    groups = {'test': ids[:held_out], 'validation': ids[held_out:2*held_out],
              'train': ids[2*held_out:]}
    return {name: df[df.id.isin(group)].copy() for name, group in groups.items()}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv_exclusive(df, path):
    with path.open('x', newline='', encoding='utf-8') as destination:
        df.to_csv(destination, index=False, na_rep='')


def process_splits(parts, output_dir, provenance):
    check_groups(parts)
    names = ['preprocessor.joblib', 'split_manifest.csv', 'encoding_summary.json']
    for name in parts:
        names += [name + '_encoded.csv', name + '_payment_audit.csv']
    for name in names:
        if (output_dir/name).exists():
            raise FileExistsError(f'Will not overwrite {output_dir/name}; choose a new --output-dir.')
    # Validate and encode everything before creating output files.
    encoded, audits = {}, {}
    encoded['train'], artifact, audits['train'] = encode_all(parts['train'])
    for name, df in parts.items():
        if name != 'train':
            encoded[name], _, audits[name] = encode_all(df, artifact)
    model = artifact['text_encoder']
    feature_cols = artifact['feature_columns']
    manifest = pd.concat([pd.DataFrame({'id': df.id.drop_duplicates(), 'split': name})
                          for name, df in parts.items()], ignore_index=True)
    summary = {
        'inputs': provenance, 'scikit_learn': sklearn.__version__,
        'feature_count': len(feature_cols), 'feature_columns': feature_cols,
        'tfidf_terms': {k: len(v.vocabulary_) if v is not None else 0 for k,v in model['vectorizers'].items()},
        'payment_training_mean': model['payment_fill_value'],
        'test_model_performance_evaluated': False,
        'splits': {name: {'rows': len(df), 'respondents': df.id.nunique(),
                          'label_counts': df.label.value_counts().to_dict(),
                          'remaining_missing_feature_cells': int(encoded[name][feature_cols].isna().sum().sum())}
                   for name, df in parts.items()},
        'missingness_policy': 'Single-choice/multi-select missing cells remain blank. Text uses zero vectors plus flags. Payment uses training mean plus flag.',
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in parts:
        write_csv_exclusive(encoded[name], output_dir/f'{name}_encoded.csv')
        write_csv_exclusive(audits[name], output_dir/f'{name}_payment_audit.csv')
    write_csv_exclusive(manifest, output_dir/'split_manifest.csv')
    with (output_dir/'preprocessor.joblib').open('xb') as destination:
        joblib.dump(artifact, destination)
    with (output_dir/'encoding_summary.json').open('x', encoding='utf-8') as destination:
        json.dump(summary, destination, indent=2)
    print(f'Saved encoded data in {output_dir.resolve()}')
    print('Rows:', {name: len(df) for name, df in parts.items()})
    print(f'Numeric feature columns: {len(feature_cols)} (id and label are metadata, not features).')
    print('TF-IDF terms:', summary['tfidf_terms'])
    print(f"Payment imputation mean, learned from training only: {model['payment_fill_value']:.4f}")
    print('Missing single-choice and multi-select cells remain blank, as in the earlier scripts.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--input', type=Path, help='Full raw CSV: create a NEW grouped split, then encode')
    mode.add_argument('--train', type=Path, help='Existing raw training split: preserve its membership')
    mode.add_argument('--transform', type=Path, help='Encode one raw CSV with a previously fitted encoder')
    parser.add_argument('--validation', type=Path, help='Existing raw validation split, with --train')
    parser.add_argument('--test', type=Path, help='Existing raw test split, with --train')
    parser.add_argument('--output-dir', type=Path, default=Path('data/encoded'))
    parser.add_argument('--seed', type=int, default=311, help='Only used with --input')
    parser.add_argument('--model', type=Path, help='Saved encoder, required for --transform')
    parser.add_argument('--output', type=Path, help='Output CSV, required for --transform')
    args = parser.parse_args()
    try:
        if args.transform:
            if not args.model or not args.output or args.validation or args.test:
                raise ValueError('--transform requires --model and --output, without split arguments.')
            audit_path = args.output.with_name(args.output.stem + '_payment_audit.csv')
            paths = [args.transform.resolve(), args.model.resolve(), args.output.resolve(), audit_path.resolve()]
            if len(paths) != len(set(paths)):
                raise ValueError('Input, model, output and audit paths must be different.')
            for path in (args.output, audit_path):
                if path.exists():
                    raise FileExistsError(f'Will not overwrite {path}.')
            # Only load saved joblib files that you trust.
            artifact = joblib.load(args.model)
            df, _, audit = encode_all(read_csv(args.transform), artifact)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_csv_exclusive(df, args.output)
            write_csv_exclusive(audit, audit_path)
            print(f'Saved {len(df)} encoded rows to {args.output.resolve()}')
            return
        if args.model or args.output:
            raise ValueError('--model and --output are only for --transform; use --output-dir when fitting.')
        if args.input:
            if args.validation or args.test:
                raise ValueError('Use --train, not --input, when supplying existing split files.')
            raw = read_csv(args.input)
            parts = grouped_split(raw, args.seed)
            provenance = {'mode': 'new_grouped_split', 'seed': args.seed,
                          'source': str(args.input), 'sha256': sha256(args.input)}
        else:
            if not args.validation or not args.test:
                raise ValueError('--train requires --validation and --test.')
            paths = {'train': args.train, 'validation': args.validation, 'test': args.test}
            parts = {name: read_csv(path) for name, path in paths.items()}
            provenance = {'mode': 'existing_splits', 'files': {name: {'path': str(path), 'sha256': sha256(path)}
                                                               for name, path in paths.items()}}
        process_splits(parts, args.output_dir, provenance)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()

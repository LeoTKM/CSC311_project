"""v2's own copy of the survey parsing and structured encoding helpers.

Copied from `data_preprocessing/encode_all.py` (sha256
c374e9502ec92808eeede80c8f17f6ad36024a4acd046da34eac0304c291c601) so that editing v2
cannot silently change v1 and editing `encode_all.py` cannot silently change v2.
`clean`, `parse_payment`, `sanitize`, `raw_headers` and `encode_structured` keep the v1
behaviour exactly.

The v2 additions are at the bottom: the refusal set, the colour lexicon and the column
lists that `features.py` consumes. None of them is imported from v1.
"""
import math
import re

import numpy as np
import pandas as pd

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
NUMBER = re.compile(r'(?<![\w.])\d+(?:\.\d+)?(?:\s*k)?(?![\d.])')


def clean(value):
    if pd.isna(value):
        return ''
    text = ' '.join(str(value).split()).lower()
    return '' if text in MISSING else text


def parse_payment(value):
    """Return (amount or None, audit status). Simple numeric parser, v1 behaviour.

    Accept one number, or exactly two joined as a range. A single value preceded by
    'at most'/'under' is treated as the stated amount. Multiple scenarios are rejected.
    Word-only amounts (including 'zero') are missing, per numeric-only policy. Explicit
    numeric zero is preserved. No exchange-rate conversion.
    """
    text = clean(value).replace('\u2013', '-').replace('\u2014', '-')
    if not text:
        return None, 'blank'
    if re.search(r'\bscale\b|order of magnitude|\d+s\b|\b(?:hundreds?|thousands?|millions?|billions?)\b', text):
        return None, 'ambiguous_or_not_a_price'
    # '1,000' / '10 000' are thousands; '200, 500' remains two separate values.
    text = re.sub(r'(?<=\d)[, ](?=\d{3}(?:\D|$))', '', text)
    # Remove currency notation without converting it, as explicitly assumed.
    text = re.sub(r'(?<![a-z])(?:cad|usd|eur|gbp|cny|rmb)(?![a-z])', ' ', text)
    text = re.sub(r'\b(?:canadian|us|dollars?|euros?|bucks|yuan)\b', ' ', text)
    text = re.sub(r'[$\u20ac\u00a3\u00a5]', ' ', text)
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


# Accept both the original CSV headers and the short headers in earlier splits.
FULL_COLUMNS = [EASE_COLUMN, *AGREEMENT_COLUMNS, *SCHEMA, *TEXT.values(), PAYMENT]
SHORT_COLUMNS = ['ease', 'strategy', 'injury', 'individual', 'location', 'season',
                 'music_type', 'food', 'colour', 'song', 'feeling', 'payment']


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


def sanitize(df):
    """Unknown structured answers become missing; raw text and payment are untouched."""
    df = raw_headers(df).reset_index(drop=True)
    vals = pd.to_numeric(df[EASE_COLUMN].map(clean), errors='coerce')
    df.loc[~(vals.between(0, 10) & np.isfinite(vals)), EASE_COLUMN] = ''
    for col in AGREEMENT_COLUMNS:
        df.loc[~df[col].map(clean).isin(AGREEMENT_MAP), col] = ''
    for col, choices in SCHEMA.items():
        allowed = {clean(option) for option, _ in choices}

        def valid(value):
            selected = {p.strip() for p in clean(value).split(',') if p.strip()}
            return bool(selected) and not (selected - allowed)

        df.loc[~df[col].map(valid), col] = ''
    return df


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
        df = pd.concat([df.iloc[:, :position], block, df.iloc[:, position + 1:]], axis=1)
    return df


# ------------------------------------------------------------------ v2 additions

# Replies that decline to answer. PLAN.md section 5: a reply in this set is a
# non-answer, which is different from a reply whose words are all outside the
# vocabulary. The words themselves are still tokenised and still reach the vocabulary.
NON_ANSWER = {
    'none', 'n/a', 'na', 'n.a.', 'nothing', 'not sure', 'unsure', 'no idea',
    'null', 'nil', 'nope', '-', '--', 'none at all',
}

# PLAN.md section 4. Fixed, never fitted, one column per colour.
COLOUR_LEXICON = {
    'blue': ('blue', 'blues'),
    'red': ('red',),
    'orange': ('orange',),
    'green': ('green',),
    'white': ('white',),
    'black': ('black', 'balck'),
    'brown': ('brown',),
    'yellow': ('yellow', 'yellowish'),
    'gold': ('gold', 'golden'),
    'purple': ('purple',),
    'navy': ('navy',),
    'gray': ('gray', 'grey'),
    'beige': ('beige',),
    'maroon': ('maroon',),
    'burgundy': ('burgundy',),
    'cream': ('cream',),
    'tan': ('tan',),
}
COLOUR_WORDS = {word for words in COLOUR_LEXICON.values() for word in words}

# Column names of the structured block, in output order.
MULTI_COLUMNS = [name for choices in SCHEMA.values() for _, name in choices]
ORDINAL_COLUMNS = [EASE_COLUMN, *AGREEMENT_COLUMNS]
PAYMENT_COLUMNS = ['payment_amount', 'payment_missing']
STRUCTURED_COLUMNS = [*ORDINAL_COLUMNS, *MULTI_COLUMNS, *PAYMENT_COLUMNS]
# q1..q7: the ease answer, the three agreement answers, the three multi-select answers.
FLAG_COLUMNS = [f'q{index}_missing' for index in range(1, 8)]

#!/usr/bin/env python3
"""Encode the first four CSC311 survey questions on a 0-1 scale.

Uses only Python's standard library (no packages to install).

Example:
    python3 encode_first_four.py survey.csv
    python3 encode_first_four.py survey.csv --output survey_encoded.csv

Q1 measures EASE, not difficulty: 0 = extremely difficult, 10 = extremely easy.
Q2-Q4 measure agreement. Missing answers remain blank. Other columns and column
names are preserved. The script refuses to overwrite an existing file.
"""
import argparse
import csv
import math
from pathlib import Path

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


def encode_csv(input_path, output_path):
    """Validate and convert all rows before writing a new output file."""
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError('Choose a different output file to preserve the original.')
    if output_path.exists():
        raise FileExistsError(f'Output already exists: {output_path}')

    with input_path.open(newline='', encoding='utf-8-sig') as source:
        reader = csv.DictReader(source)
        headers = reader.fieldnames or []
        required = (EASE_COLUMN,) + AGREEMENT_COLUMNS
        missing = [column for column in required if column not in headers]
        if missing:
            raise ValueError('Required columns not found: ' + '; '.join(missing))
        rows = list(reader)

    for row_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f'CSV row {row_number}: incorrect number of fields.')
        # The fixed survey endpoints are 0 and 10. Do not use observed min/max.
        answer = row[EASE_COLUMN].strip()
        if answer:
            try:
                value = float(answer)
            except ValueError:
                raise ValueError(f'CSV row {row_number}: invalid ease rating {answer!r}.') from None
            if not math.isfinite(value) or not 0 <= value <= 10:
                raise ValueError(f'CSV row {row_number}: ease rating must be between 0 and 10.')
            row[EASE_COLUMN] = format(value / 10, '.12g')
        else:
            row[EASE_COLUMN] = ''

        for column in AGREEMENT_COLUMNS:
            answer = ' '.join(row[column].split()).lower()
            if not answer:
                row[column] = ''
            elif answer in AGREEMENT_MAP:
                row[column] = AGREEMENT_MAP[answer]
            else:
                raise ValueError(
                    f'CSV row {row_number}: unknown agreement answer {row[column]!r} '
                    f'in {column!r}.'
                )

    # Exclusive creation prevents overwriting a file, including the original.
    with output_path.open('x', newline='', encoding='utf-8') as destination:
        writer = csv.DictWriter(destination, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Original survey CSV')
    parser.add_argument('--output', type=Path, help='New CSV path; default: INPUT_encoded.csv')
    args = parser.parse_args()
    output = args.output or args.input.with_name(args.input.stem + '_encoded.csv')
    try:
        count = encode_csv(args.input, output)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f'Error: {error}\n')
    print(f'Encoded the first four question columns for {count} rows.')
    print(f'Saved: {output.resolve()}')
    print('Missing answers remain blank; all other columns are unchanged.')


if __name__ == '__main__':
    main()

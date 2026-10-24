#!/usr/bin/env python3
"""Multi-hot encode the three multiple-selection CSC311 survey questions.

Usage:
    python3 multiple_choice_encode.py survey.csv
    python3 multiple_choice_encode.py survey.csv --output survey_multihot.csv

Replaces location, season and music-type columns with 7, 4 and 10 binary columns.
For answered questions: 1 = selected, 0 = not selected.
For missing questions: all corresponding cells remain blank (unknown).
Other columns and row order are preserved. No extra Python packages are needed.
Works on either the original CSV or the output of encode_first_four.py.
"""
import argparse
import csv
from pathlib import Path

# Fixed survey choices, not vocabularies learned from validation or test data.
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


def normalize(value):
    return ' '.join(value.split()).lower()


def encode_csv(input_path, output_path):
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError('Choose a different output file to preserve the original.')
    if output_path.exists():
        raise FileExistsError(f'Output already exists: {output_path}')

    with input_path.open(newline='', encoding='utf-8-sig') as source:
        reader = csv.DictReader(source)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)):
            raise ValueError('The input contains duplicate column names.')
        missing = [column for column in SCHEMA if column not in headers]
        if missing:
            raise ValueError('Required survey columns not found: ' + '; '.join(missing))
        new_columns = [name for choices in SCHEMA.values() for _, name in choices]
        if set(new_columns) & set(headers):
            raise ValueError('Some encoded columns already exist in the input.')
        rows = list(reader)

    output_headers = []
    for column in headers:
        output_headers.extend([name for _, name in SCHEMA[column]] if column in SCHEMA else [column])

    # Validate every answer before creating the output file.
    encoded_rows = []
    for row_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f'CSV row {row_number}: incorrect number of fields.')
        encoded = {column: value for column, value in row.items() if column not in SCHEMA}
        for column, choices in SCHEMA.items():
            raw = row[column]
            # Also recognize explicit missing markers, but retain 'No particular type'.
            if normalize(raw) in {'', 'n/a', 'na', 'nan', 'null', '-', '[]'}:
                encoded.update({name: '' for _, name in choices})
                continue
            selected = {normalize(part) for part in raw.split(',') if normalize(part)}
            allowed = {normalize(option) for option, _ in choices}
            if not selected or selected - allowed:
                raise ValueError(f'CSV row {row_number}: unknown or empty selection {raw!r} in {column!r}.')
            encoded.update({name: int(normalize(option) in selected) for option, name in choices})
        encoded_rows.append(encoded)

    with output_path.open('x', newline='', encoding='utf-8') as destination:
        writer = csv.DictWriter(destination, fieldnames=output_headers)
        writer.writeheader()
        writer.writerows(encoded_rows)
    return len(encoded_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Original CSV or first-four-encoded CSV')
    parser.add_argument('--output', type=Path, help='New CSV path; default: INPUT_multihot.csv')
    args = parser.parse_args()
    output = args.output or args.input.with_name(args.input.stem + '_multihot.csv')
    try:
        count = encode_csv(args.input, output)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f'Error: {error}\n')
    print(f'Encoded {count} rows: 7 location + 4 season + 10 music columns.')
    print(f'Saved: {output.resolve()}')
    print('Missing answers remain blank; all other columns are unchanged.')


if __name__ == '__main__':
    main()

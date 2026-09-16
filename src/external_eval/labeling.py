"""Spreadsheet-friendly export and strict validation for human labels."""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Sequence

from src.config import LABELS, PROJECT_ROOT

logger = logging.getLogger(__name__)
DEFAULT_DATA_ROOT = PROJECT_ROOT / 'data' / 'external_eval'
LABELING_COLUMNS = (
    'sample_id', 'source', 'post_id', 'created_at', 'query', 'text',
    'label', 'review_flag', 'notes',
)


def load_prepared_rows(path: Path | str) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f'Invalid prepared JSON on line {line_number}') from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f'Prepared line {line_number} must be a JSON object')
            rows.append(row)
    return rows


def export_labeling_csv(
    prepared_path: Path | str,
    output_path: Path | str,
) -> Path:
    """Export privacy-minimized rows with intentionally blank human labels."""

    rows = load_prepared_rows(prepared_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=LABELING_COLUMNS)
        writer.writeheader()
        for row in rows:
            exported = {
                'sample_id': row.get('sample_id', ''),
                'source': row.get('source', ''),
                'post_id': row.get('post_id', ''),
                'created_at': row.get('created_at', ''),
                'query': row.get('query', ''),
                'text': row.get('text', ''),
                'label': '',
                'review_flag': row.get('review_flag', ''),
                'notes': '',
            }
            writer.writerow(exported)
    return destination


def read_labeling_csv(path: Path | str) -> list[dict[str, str]]:
    with Path(path).open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = set(LABELING_COLUMNS) - fieldnames
        if missing:
            raise ValueError(f'Labeling file is missing columns: {sorted(missing)}')
        return [dict(row) for row in reader]


def validate_labeled_file(
    path: Path | str,
    *,
    expected_min: int | None = None,
    expected_max: int | None = None,
) -> dict:
    """Require complete human labels and return a text-free validation summary."""

    rows = read_labeling_csv(path)
    if not rows:
        raise ValueError('Labeling file contains no rows')
    if expected_min is not None and len(rows) < expected_min:
        raise ValueError(
            f'Expected at least {expected_min} labeled rows; found {len(rows)}')
    if expected_max is not None and len(rows) > expected_max:
        raise ValueError(
            f'Expected at most {expected_max} labeled rows; found {len(rows)}')
    seen_samples: set[str] = set()
    seen_stable_ids: set[tuple[str, str]] = set()
    label_counts = {label: 0 for label in LABELS}
    for line_number, row in enumerate(rows, start=2):
        for field in ('sample_id', 'source', 'post_id', 'query', 'text'):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f'Row {line_number} has no {field}')
        sample_id = row['sample_id'].strip()
        if sample_id in seen_samples:
            raise ValueError(f'Duplicate sample_id: {sample_id}')
        seen_samples.add(sample_id)
        stable_id = (row['source'].strip(), row['post_id'].strip())
        if stable_id in seen_stable_ids:
            raise ValueError(
                f'Duplicate source/post_id: {stable_id[0]}/{stable_id[1]}')
        seen_stable_ids.add(stable_id)
        review_flag = (row.get('review_flag') or '').strip()
        if 'language_review' in {item.strip() for item in review_flag.split(';')}:
            raise ValueError(
                f'Row {line_number} has unresolved language_review flag')
        label = (row.get('label') or '').strip()
        if not label:
            raise ValueError(
                f'Row {line_number} has no human label; evaluation is blocked')
        if label not in LABELS:
            raise ValueError(
                f'Row {line_number} label must be one of {LABELS}')
        label_counts[label] += 1
    return {'row_count': len(rows), 'label_counts': label_counts,
            'unique_sample_ids': len(seen_samples)}


def latest_prepared_file(root: Path = DEFAULT_DATA_ROOT) -> Path:
    paths = sorted((root / 'prepared').glob('external_prepared_*.jsonl'))
    if not paths:
        raise FileNotFoundError('No prepared dataset found; run prepare first')
    return paths[-1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Export or validate human external-evaluation labels.')
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export')
    export.add_argument('--prepared', type=Path)
    export.add_argument(
        '--output', type=Path,
        default=DEFAULT_DATA_ROOT / 'labels' / 'external_eval_labeling.csv')
    validate = commands.add_parser('validate')
    validate.add_argument('--file', type=Path, required=True)
    validate.add_argument('--expected-min', type=int)
    validate.add_argument('--expected-max', type=int)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        if args.command == 'export':
            source = args.prepared or latest_prepared_file()
            path = export_labeling_csv(source, args.output)
            print(f'Blank labeling CSV: {path}')
            print('STOP: human labels are required before validation or evaluation.')
        else:
            result = validate_labeled_file(
                args.file, expected_min=args.expected_min,
                expected_max=args.expected_max)
            print(json.dumps(result, indent=2, sort_keys=True))
    except (FileExistsError, FileNotFoundError, TypeError, ValueError) as exc:
        logger.error('%s', exc)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

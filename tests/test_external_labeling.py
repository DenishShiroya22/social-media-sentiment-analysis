"""Tests for blank labeling export and strict human-label validation."""
import csv
import json

import pytest

from src.external_eval.labeling import (
    LABELING_COLUMNS, export_labeling_csv, read_labeling_csv,
    validate_labeled_file,
)


def prepared_file(tmp_path):
    path = tmp_path/'prepared.jsonl'
    rows = [
        {
            'sample_id': 'sample-one', 'source': 'reddit', 'post_id': 'r1',
            'created_at': '2026-09-16T10:00:00Z', 'query': 'phone',
            'text': 'Entirely synthetic positive comment',
            'review_flag': '', 'author_id_or_name': None,
        },
        {
            'sample_id': 'sample-two', 'source': 'youtube', 'post_id': 'y1',
            'created_at': '2026-09-16T11:00:00Z', 'query': 'service',
            'text': 'Entirely synthetic neutral comment',
            'review_flag': 'language_review', 'author_id_or_name': None,
        },
    ]
    path.write_text('\n'.join(json.dumps(row) for row in rows)+'\n',
                    encoding='utf-8')
    return path


def write_rows(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=LABELING_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_export_is_blank_private_and_spreadsheet_friendly(tmp_path):
    destination = tmp_path/'labels'/'external_eval_labeling.csv'
    export_labeling_csv(prepared_file(tmp_path), destination)
    rows = read_labeling_csv(destination)
    assert [row['label'] for row in rows] == ['', '']
    assert rows[1]['review_flag'] == 'language_review'
    assert set(rows[0]) == set(LABELING_COLUMNS)
    text = destination.read_text(encoding='utf-8-sig')
    assert 'author' not in text.casefold()
    with pytest.raises(FileExistsError):
        export_labeling_csv(prepared_file(tmp_path), destination)


def test_blank_labels_block_evaluation_validation(tmp_path):
    destination = tmp_path/'labels.csv'
    export_labeling_csv(prepared_file(tmp_path), destination)
    with pytest.raises(ValueError, match='no human label'):
        validate_labeled_file(destination)


def test_valid_completed_human_labels_return_aggregate_counts(tmp_path):
    destination = tmp_path/'labels.csv'
    export_labeling_csv(prepared_file(tmp_path), destination)
    rows = read_labeling_csv(destination)
    rows[0]['label'] = 'positive'
    rows[1]['label'] = 'neutral'
    rows[1]['review_flag'] = ''
    write_rows(destination, rows)
    result = validate_labeled_file(
        destination, expected_min=2, expected_max=2)
    assert result == {
        'row_count': 2,
        'label_counts': {'negative': 0, 'neutral': 1, 'positive': 1},
        'unique_sample_ids': 2,
    }


@pytest.mark.parametrize('change,match', [
    (lambda rows: rows[0].update(label='mixed'), 'label must be'),
    (lambda rows: rows[1].update(sample_id='sample-one'), 'Duplicate sample_id'),
    (lambda rows: rows[1].update(source='reddit', post_id='r1'),
     'Duplicate source/post_id'),
    (lambda rows: rows[0].update(text=''), 'has no text'),
    (lambda rows: rows[0].update(review_flag='language_review'),
     'unresolved language_review'),
])
def test_invalid_labeled_schema_is_rejected(tmp_path, change, match):
    destination = tmp_path/'labels.csv'
    export_labeling_csv(prepared_file(tmp_path), destination)
    rows = read_labeling_csv(destination)
    for row in rows:
        row['label'] = 'neutral'
    change(rows)
    write_rows(destination, rows)
    with pytest.raises(ValueError, match=match):
        validate_labeled_file(destination)

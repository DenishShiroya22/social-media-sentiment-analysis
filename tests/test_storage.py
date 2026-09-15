"""Tests for local JSONL records and run manifests."""
from datetime import datetime, timezone
import json

import pytest

from src.social.schemas import SocialPost
from src.storage.repository import (
    JsonlRepository, RunManifest, create_run_id, safe_segment,
)


def test_jsonl_utf8_round_trip_and_directory_creation(tmp_path):
    repository = JsonlRepository(tmp_path/'data')
    run_id = create_run_id(datetime(2026, 9, 15, tzinfo=timezone.utc))
    post = SocialPost(
        'reddit', 'one', datetime(2026, 9, 15, tzinfo=timezone.utc),
        'I love café batteries 🔋', 'Galaxy S26')
    path = repository.write_records(
        'collected', 'reddit', 'Galaxy S26', run_id, [post])
    assert path.parent == tmp_path/'data'/'collected'
    assert 'café' in path.read_text(encoding='utf-8')
    values = repository.read_records(path)
    assert values[0]['text'] == 'I love café batteries 🔋'
    assert list(values[0]) == sorted(values[0])
    assert repository.relative_path(path).startswith('data/collected/')


def test_empty_jsonl_is_valid_and_existing_file_is_not_overwritten(tmp_path):
    repository = JsonlRepository(tmp_path/'data')
    run_id = create_run_id()
    path = repository.write_records('analyzed', 'reddit', 'phone', run_id, [])
    assert path.read_bytes() == b''
    with pytest.raises(FileExistsError):
        repository.write_records(
            'analyzed', 'reddit', 'phone', run_id, [{'text': 'new'}])


def test_manifest_write_round_trip_and_no_overwrite(tmp_path):
    repository = JsonlRepository(tmp_path/'data')
    run_id = create_run_id(datetime(2026, 9, 15, tzinfo=timezone.utc))
    manifest = RunManifest(
        run_id=run_id, source='reddit', query='phone',
        retrieved_count=3, analyzed_count=3,
        created_at='2026-09-15T00:00:00Z',
        model_sha256='a'*64, model_version='final-lr',
        connector_version='praw-8.0.3',
        collected_file=f'data/collected/{run_id}.jsonl',
        output_file=f'data/analyzed/{run_id}.jsonl')
    path = repository.write_manifest(manifest)
    assert json.loads(path.read_text(encoding='utf-8'))['run_id'] == run_id
    with pytest.raises(FileExistsError):
        repository.write_manifest(manifest)


@pytest.mark.parametrize('value,expected', [
    ('Samsung Galaxy S26', 'samsung-galaxy-s26'),
    ('  Café Phone  ', 'caf-phone'),
])
def test_safe_filename_segments(value, expected):
    assert safe_segment(value) == expected


def test_storage_rejects_bad_categories_and_manifest_counts(tmp_path):
    repository = JsonlRepository(tmp_path/'data')
    with pytest.raises(ValueError, match='category'):
        repository.write_records(
            'private', 'reddit', 'query', create_run_id(), [])
    with pytest.raises(ValueError, match='exceed'):
        RunManifest(
            run_id=create_run_id(), source='reddit', query='q',
            retrieved_count=1, analyzed_count=2,
            created_at='2026-09-15T00:00:00Z', model_sha256='a'*64,
            model_version='v', connector_version='v',
            collected_file='a', output_file='b')

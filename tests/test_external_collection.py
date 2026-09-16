"""Offline collection-config and API orchestration tests."""
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from src.external_eval.collect import collect_external_posts
from src.external_eval.config import ExternalEvalConfig, load_config
from src.external_eval.prepare import prepare_external_dataset
from src.social.schemas import SocialPost


def config_mapping():
    return {
        'queries': ['synthetic phone', 'synthetic service'],
        'target_total': 4,
        'per_query_target': 2,
        'sampling_seed': 42,
        'max_per_context': 2,
        'sources': {
            'reddit': {
                'enabled': True, 'per_query_limit': 1,
                'sort': 'relevance', 'time_filter': 'year',
                'include_comments': True,
                'max_comments_per_submission': 1,
            },
            'youtube': {
                'enabled': True, 'max_videos_per_query': 1,
                'max_comments_per_video': 1,
            },
        },
    }


class FakeConnector:
    def __init__(self, source):
        self.source = source
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return [SocialPost(
            self.source, f'{self.source}-{len(self.calls)}',
            datetime(2026, 9, 16, tzinfo=timezone.utc),
            f'Entirely synthetic {self.source} text for {query}', query,
            author_id_or_name='must-not-be-stored',
            metadata={'video_id' if self.source == 'youtube' else 'submission_id':
                      f'context-{len(self.calls)}'})]


def test_config_load_and_canonical_hash(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config_mapping()), encoding='utf-8')
    one = load_config(path)
    two = ExternalEvalConfig.from_mapping(config_mapping())
    assert one == two
    assert one.sha256 == two.sha256
    assert len(one.sha256) == 64
    assert one.sampling_seed == 42


@pytest.mark.parametrize('mutator', [
    lambda value: value.update(queries=[]),
    lambda value: value.update(target_total=501),
    lambda value: value.update(sampling_seed=-1),
    lambda value: value['sources']['reddit'].update(sort='hot'),
    lambda value: value['sources']['youtube'].update(max_videos_per_query=0),
])
def test_invalid_collection_config_is_rejected(mutator):
    value = config_mapping()
    mutator(value)
    with pytest.raises((TypeError, ValueError)):
        ExternalEvalConfig.from_mapping(value)


def test_collection_writes_privacy_minimized_jsonl_and_manifest(tmp_path):
    config = ExternalEvalConfig.from_mapping(config_mapping())
    reddit = FakeConnector('reddit')
    youtube = FakeConnector('youtube')
    result = collect_external_posts(
        config, connectors={'reddit': reddit, 'youtube': youtube},
        output_root=tmp_path/'data'/'external_eval',
        now=datetime(2026, 9, 17, 8, tzinfo=timezone.utc))
    assert result['manifest']['retrieved_total'] == 4
    assert result['manifest']['sampling_seed'] == 42
    assert result['manifest']['privacy']['author_identity_collected'] is False
    raw_rows = [json.loads(line) for line in
                result['raw_path'].read_text(encoding='utf-8').splitlines()]
    assert len(raw_rows) == 4
    assert all(row['author_id_or_name'] is None for row in raw_rows)
    assert 'must-not-be-stored' not in result['raw_path'].read_text(encoding='utf-8')
    manifest_text = result['manifest_path'].read_text(encoding='utf-8')
    assert 'synthetic-key' not in manifest_text
    assert reddit.calls[0][1]['max_comments_per_submission'] == 1
    assert youtube.calls[0][1]['max_videos_per_query'] == 1
    prepared = prepare_external_dataset(
        result['raw_path'], config,
        output_root=tmp_path/'data'/'external_eval',
        now=datetime(2026, 9, 17, 9, tzinfo=timezone.utc))
    updated = json.loads(result['manifest_path'].read_text(encoding='utf-8'))
    assert updated['status'] == 'prepared_waiting_for_human_labeling'
    assert updated['prepared_count'] == 4
    assert updated['deduplicated_count'] == 4
    assert not Path(updated['output_paths']['prepared_jsonl']).is_absolute()
    assert prepared['manifest']['source_collection_run_id'] == (
        result['manifest']['run_id'])


def test_real_external_data_paths_are_git_ignored():
    ignore = Path('.gitignore').read_text(encoding='utf-8')
    assert 'data/external_eval/raw/' in ignore
    assert 'data/external_eval/prepared/' in ignore
    assert 'data/external_eval/labels/' in ignore
    assert 'data/external_eval/manifests/' not in ignore

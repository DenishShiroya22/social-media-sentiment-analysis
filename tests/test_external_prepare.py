"""Tests for filtering, deduplication, and reproducible sampling."""
from collections import Counter
from datetime import datetime, timezone
import json

from src.external_eval.config import ExternalEvalConfig
from src.external_eval.prepare import (
    deduplicate_posts, filter_content, language_review_flag,
    prepare_external_dataset,
)
from src.external_eval.sampling import stratified_sample
from src.social.schemas import SocialPost


def post(source, identifier, query, text, context):
    metadata_key = 'video_id' if source == 'youtube' else 'submission_id'
    return SocialPost(
        source, identifier, datetime(2026, 9, 16, tzinfo=timezone.utc),
        text, query, author_id_or_name='synthetic-author',
        metadata={metadata_key: context})


def small_config():
    return ExternalEvalConfig.from_mapping({
        'queries': ['phone', 'service'],
        'target_total': 4,
        'per_query_target': 2,
        'sampling_seed': 42,
        'max_per_context': 1,
        'sources': {
            'reddit': {'enabled': True},
            'youtube': {'enabled': False},
        },
    })


def test_content_filters_are_narrow_and_keep_short_sentiment():
    values = [
        post('reddit', '1', 'phone', '[deleted]', 'a'),
        post('reddit', '2', 'phone', 'https://example.test/a', 'b'),
        post('reddit', '3', 'phone', 'Bad', 'c'),
    ]
    accepted, counts = filter_content(values)
    assert [item.post_id for item in accepted] == ['3']
    assert counts == {'placeholder': 1, 'url_only': 1}


def test_deduplication_reports_id_and_exact_text_counts():
    values = [
        post('reddit', '1', 'phone', 'Same artificial text', 'a'),
        post('reddit', '1', 'phone', 'Changed duplicate ID', 'a'),
        post('youtube', '2', 'phone', ' same  ARTIFICIAL text ', 'b'),
        post('youtube', '3', 'phone', 'Unique artificial text', 'c'),
    ]
    unique, stats = deduplicate_posts(values)
    assert [item.post_id for item in unique] == ['1', '3']
    assert stats == {
        'id_duplicate_count': 1,
        'exact_text_duplicate_count': 1,
        'removed_duplicate_count': 2,
    }


def test_sampling_is_deterministic_balanced_and_caps_contexts():
    values = []
    for source in ('reddit', 'youtube'):
        for query in ('phone', 'service'):
            for index in range(4):
                values.append(post(
                    source, f'{source}-{query}-{index}', query,
                    f'Synthetic {source} {query} text {index}',
                    f'dominant-{query}' if index < 3 else f'unique-{source}-{query}'))
    one = stratified_sample(
        values, target_total=8, per_query_target=4,
        max_per_context=1, seed=42)
    two = stratified_sample(
        reversed(values), target_total=8, per_query_target=4,
        max_per_context=1, seed=42)
    assert [item.post_id for item in one] == [item.post_id for item in two]
    assert Counter(item.query for item in one) == {'phone': 4, 'service': 4}
    contexts = Counter((item.source, item.metadata.get(
        'video_id') or item.metadata.get('submission_id')) for item in one)
    assert max(contexts.values()) == 1


def test_language_policy_flags_for_review_instead_of_dropping():
    assert language_review_flag('This is an artificial English comment') == ''
    assert language_review_flag('यह एक कृत्रिम टिप्पणी है') == 'language_review'
    assert language_review_flag('Bad') == ''


def test_prepare_writes_unlabeled_privacy_minimized_sample(tmp_path):
    values = [
        post('reddit', 'r1', 'phone', 'Artificial phone praise', 'one'),
        post('youtube', 'y1', 'phone', 'Artificial phone complaint', 'two'),
        post('reddit', 'r2', 'service', 'Artificial service fact', 'three'),
        post('youtube', 'y2', 'service', 'Artificial service question', 'four'),
        post('reddit', 'r2', 'service', 'Duplicate ID', 'three'),
    ]
    raw = tmp_path/'raw.jsonl'
    raw.write_text('\n'.join(item.to_json() for item in values)+'\n', encoding='utf-8')
    result = prepare_external_dataset(
        raw, small_config(), output_root=tmp_path/'data'/'external_eval',
        now=datetime(2026, 9, 17, tzinfo=timezone.utc))
    assert result['manifest']['initial_count'] == 5
    assert result['manifest']['id_duplicate_count'] == 1
    assert result['manifest']['prepared_count'] == 4
    assert result['manifest']['human_labels_present'] is False
    rows = [json.loads(line) for line in
            result['prepared_path'].read_text(encoding='utf-8').splitlines()]
    assert all(row['author_id_or_name'] is None for row in rows)
    assert all(len(row['sample_id']) == 20 for row in rows)
    assert all('label' not in row for row in rows)

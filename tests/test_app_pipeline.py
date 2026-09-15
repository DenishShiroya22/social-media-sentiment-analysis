"""Offline application orchestration and aggregation tests."""
from datetime import datetime, timezone

import pytest

from src.app_pipeline import (
    analyze_posts, print_summary, run_application, summarize_posts,
)
from src.inference import SentimentPrediction
from src.social.base import SocialConnector
from src.social.schemas import SocialPost
from src.storage.repository import JsonlRepository


class FakeConnector(SocialConnector):
    source = 'fixture'

    def __init__(self, posts):
        self.posts = posts
        self.calls = []

    def search(self, query, limit=50, **kwargs):
        self.calls.append((query, limit, kwargs))
        return self.posts[:limit]


class FakePredictor:
    artifact_sha256 = 'b'*64
    model_version = 'final-lr-fixture'

    def __init__(self):
        self.calls = []

    def predict_batch(self, texts):
        self.calls.append(list(texts))
        labels = ['positive', 'neutral', 'negative']
        results = []
        for index, text in enumerate(texts):
            label = labels[index % 3]
            probabilities = {
                'negative': .8 if label == 'negative' else .1,
                'neutral': .8 if label == 'neutral' else .1,
                'positive': .8 if label == 'positive' else .1}
            results.append(SentimentPrediction(
                text=text, clean_text=text.lower(), sentiment=label,
                confidence=.8, probabilities=probabilities,
                model_version=self.model_version,
                model_artifact_sha256=self.artifact_sha256,
                inference_timestamp='2026-09-15T00:00:00Z'))
        return results


def posts():
    return [
        SocialPost('fixture', f'id-{index}',
                   datetime(2026, 9, 15, tzinfo=timezone.utc),
                   text, 'phone')
        for index, text in enumerate(
            ['Love this phone', 'Phone update today', 'Terrible phone'])
    ]


def test_analyze_posts_batches_once_and_preserves_order():
    predictor = FakePredictor()
    source = posts()
    result = analyze_posts(source, predictor)
    assert predictor.calls == [[post.text for post in source]]
    assert [item.post_id for item in result] == [post.post_id for post in source]
    assert [item.sentiment for item in result] == [
        'positive', 'neutral', 'negative']


def test_summary_counts_percentages_and_confidence():
    result = analyze_posts(posts(), FakePredictor())
    summary = summarize_posts(result)
    assert summary['analyzed_count'] == 3
    assert summary['sentiment_counts'] == {
        'negative': 1, 'neutral': 1, 'positive': 1}
    assert summary['sentiment_percentages']['negative'] == pytest.approx(100/3)
    assert summary['average_confidence'] == pytest.approx(.8)
    assert summarize_posts([])['average_confidence'] == 0


def test_full_application_run_writes_collected_analyzed_and_manifest(tmp_path):
    connector = FakeConnector(posts())
    predictor = FakePredictor()
    repository = JsonlRepository(tmp_path/'data')
    result = run_application(
        connector, query='phone', limit=3, sort='new', time_filter='week',
        predictor=predictor, storage=repository,
        now=datetime(2026, 9, 15, 12, tzinfo=timezone.utc))
    assert connector.calls == [
        ('phone', 3, {'sort': 'new', 'time_filter': 'week'})]
    assert result['summary']['analyzed_count'] == 3
    assert result['manifest']['retrieved_count'] == 3
    assert result['manifest']['model_sha256'] == 'b'*64
    assert result['manifest']['connector_version'].endswith('FakeConnector')
    collected = repository.read_records(
        tmp_path/result['manifest']['collected_file'])
    analyzed = repository.read_records(
        tmp_path/result['manifest']['output_file'])
    assert len(collected) == len(analyzed) == 3
    assert 'sentiment' not in collected[0]
    assert analyzed[0]['sentiment'] == 'positive'
    assert (tmp_path/result['manifest_file']).is_file()


def test_zero_result_run_still_has_auditable_files(tmp_path):
    result = run_application(
        FakeConnector([]), query='nothing', predictor=FakePredictor(),
        storage=JsonlRepository(tmp_path/'data'))
    assert result['summary']['analyzed_count'] == 0
    assert result['manifest']['retrieved_count'] == 0

def test_model_failure_prevents_collection(tmp_path):
    class BrokenPredictor:
        @property
        def artifact_sha256(self):
            raise FileNotFoundError('missing model')

    connector = FakeConnector(posts())
    with pytest.raises(FileNotFoundError, match='missing model'):
        run_application(
            connector, query='phone', predictor=BrokenPredictor(),
            storage=JsonlRepository(tmp_path/'data'))
    assert connector.calls == []
    assert not (tmp_path/'data').exists()


def test_console_summary_is_compact(capsys, tmp_path):
    result = run_application(
        FakeConnector(posts()), query='phone', predictor=FakePredictor(),
        storage=JsonlRepository(tmp_path/'data'))
    print_summary('phone', result)
    output = capsys.readouterr().out
    assert 'Posts analyzed: 3' in output
    assert 'Negative: 1 (33.3%)' in output
    assert 'Average confidence: 0.800' in output
    assert 'data/analyzed/' in output

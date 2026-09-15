"""Tests for platform-neutral social record schemas."""
from datetime import datetime, timezone
import json

import pytest

from src.inference import SentimentPrediction
from src.social.schemas import AnalyzedPost, SocialPost


@pytest.fixture
def post():
    return SocialPost(
        source='reddit', post_id='abc123',
        created_at=datetime(2026, 9, 15, 10, 30, tzinfo=timezone.utc),
        text='A synthetic product comment', query='sample phone',
        url='https://www.reddit.com/r/example/comments/abc123',
        engagement={'score': 4, 'comment_count': 2},
        metadata={'subreddit': 'example'})


def test_social_post_round_trip_and_timestamp(post):
    value = post.to_dict()
    assert value['created_at'] == '2026-09-15T10:30:00Z'
    assert value['author_id_or_name'] is None
    assert SocialPost.from_dict(json.loads(post.to_json())) == post


def test_author_is_optional_and_explicit():
    no_author = SocialPost(
        'reddit', 'one', datetime.now(timezone.utc), 'text', 'query')
    with_author = SocialPost(
        'reddit', 'two', datetime.now(timezone.utc), 'text', 'query',
        author_id_or_name='synthetic-user')
    assert no_author.author_id_or_name is None
    assert with_author.author_id_or_name == 'synthetic-user'


@pytest.mark.parametrize('kwargs', [
    {'source': 'Reddit'},
    {'source': ''},
    {'post_id': ''},
    {'text': '  '},
    {'query': ''},
    {'created_at': datetime(2026, 1, 1)},
    {'engagement': {'score': 'high'}},
    {'engagement': []},
    {'metadata': []},
    {'author_id_or_name': 42},
])
def test_invalid_social_post_values(post, kwargs):
    values = {
        'source': post.source, 'post_id': post.post_id,
        'created_at': post.created_at, 'text': post.text,
        'query': post.query, 'engagement': post.engagement,
        'metadata': post.metadata,
        'author_id_or_name': post.author_id_or_name,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        SocialPost(**values)


def test_analyzed_post_from_prediction_round_trip(post):
    prediction = SentimentPrediction(
        text=post.text, clean_text='a synthetic product comment',
        sentiment='neutral', confidence=.6,
        probabilities={'negative': .1, 'neutral': .6, 'positive': .3},
        model_version='final-lr-test',
        model_artifact_sha256='a'*64,
        inference_timestamp='2026-09-15T10:31:00Z')
    analyzed = AnalyzedPost.from_prediction(post, prediction)
    assert analyzed.sentiment == 'neutral'
    assert analyzed.prob_neutral == .6
    assert AnalyzedPost.from_dict(json.loads(analyzed.to_json())) == analyzed


def test_analyzed_schema_rejects_invalid_probabilities(post):
    values = {
        **post.to_dict(), 'created_at': post.created_at,
        'clean_text': 'clean', 'sentiment': 'positive',
        'confidence': .8, 'prob_negative': .2,
        'prob_neutral': .2, 'prob_positive': .2}
    with pytest.raises(ValueError, match='sum'):
        AnalyzedPost(**values)
    values.update(
        prob_negative=.1, prob_neutral=.1, prob_positive=.8,
        confidence=.7)
    with pytest.raises(ValueError, match='maximum'):
        AnalyzedPost(**values)

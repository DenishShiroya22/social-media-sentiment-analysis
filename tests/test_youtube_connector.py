"""Offline tests for the bounded YouTube Data API connector."""
import json

import pytest

from src.social.base import (
    ConnectorConfigurationError, ConnectorContentUnavailableError,
    ConnectorRateLimitError, ConnectorRequestError,
)
from src.social.youtube import (
    MAX_COMMENTS, MAX_VIDEOS, YouTubeConnector, YouTubeCredentials,
)


def fixture_payload():
    return json.loads(open(
        'tests/fixtures/youtube_api_responses.json', encoding='utf-8').read())


def test_missing_youtube_key_is_clear():
    with pytest.raises(ConnectorConfigurationError, match='YOUTUBE_API_KEY'):
        YouTubeCredentials.from_environment({})


def test_youtube_normalizes_comments_without_author_identity():
    fixture = fixture_payload()
    calls = []

    def request(endpoint, params):
        calls.append((endpoint, dict(params)))
        if endpoint == 'search':
            return fixture['search']
        return fixture['comments'][params['videoId']]

    connector = YouTubeConnector(
        YouTubeCredentials('synthetic-key'), request_json=request)
    posts = connector.search(
        'sample phone', max_videos_per_query=2, max_comments_per_video=2)
    assert [post.post_id for post in posts] == ['comment-one', 'comment-two']
    assert all(post.source == 'youtube' for post in posts)
    assert all(post.query == 'sample phone' for post in posts)
    assert all(post.author_id_or_name is None for post in posts)
    assert posts[0].metadata == {
        'video_id': 'video-one', 'record_type': 'comment'}
    assert posts[0].parent_id == 'video-one'
    assert posts[0].engagement == {'like_count': 7}
    assert 'unused-synthetic-author' not in posts[0].to_json()
    assert calls[0][0] == 'search'
    assert calls[1][1]['maxResults'] == 2


def test_disabled_comments_are_skipped_without_fake_records():
    fixture = fixture_payload()

    def request(endpoint, params):
        if endpoint == 'search':
            return fixture['search']
        if params['videoId'] == 'video-one':
            raise ConnectorContentUnavailableError('disabled')
        return {'items': []}

    connector = YouTubeConnector(
        YouTubeCredentials('synthetic-key'), request_json=request)
    assert connector.search('query', max_videos_per_query=2) == []


def test_malformed_comments_are_logged_and_skipped(caplog):
    def request(endpoint, params):
        if endpoint == 'search':
            return {'items': [{'id': {'videoId': 'video-one'}}]}
        return {'items': [{'snippet': {}}]}

    connector = YouTubeConnector(
        YouTubeCredentials('synthetic-key'), request_json=request)
    assert connector.search('query') == []
    assert 'malformed YouTube comment' in caplog.text


@pytest.mark.parametrize('kwargs,error', [
    ({'query': ''}, ValueError),
    ({'query': 'q', 'max_videos_per_query': 0}, ValueError),
    ({'query': 'q', 'max_videos_per_query': MAX_VIDEOS + 1}, ValueError),
    ({'query': 'q', 'max_comments_per_video': MAX_COMMENTS + 1}, ValueError),
    ({'query': 'q', 'max_comments_per_video': 1.5}, TypeError),
])
def test_youtube_bounds_are_validated(kwargs, error):
    connector = YouTubeConnector(
        YouTubeCredentials('synthetic-key'), request_json=lambda *_: {})
    with pytest.raises(error):
        connector.search(**kwargs)


@pytest.mark.parametrize('error', [ConnectorRateLimitError, ConnectorRequestError])
def test_youtube_connector_propagates_translated_api_errors(error):
    def request(endpoint, params):
        raise error('synthetic remote failure')

    connector = YouTubeConnector(
        YouTubeCredentials('synthetic-key'), request_json=request)
    with pytest.raises(error):
        connector.search('query')


def test_request_hook_must_return_mapping():
    connector = YouTubeConnector(
        YouTubeCredentials('synthetic-key'), request_json=lambda *_: [])
    with pytest.raises(ConnectorRequestError, match='malformed'):
        connector.search('query')

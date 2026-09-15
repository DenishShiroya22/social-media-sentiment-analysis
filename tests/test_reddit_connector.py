"""Mocked, offline tests for the bounded Reddit connector."""
from datetime import timezone
import json
from types import SimpleNamespace

import pytest

import src.social.reddit as reddit_module
from src.social.base import (
    ConnectorAuthenticationError,
    ConnectorConfigurationError,
    ConnectorRateLimitError,
    ConnectorRequestError,
)
from src.social.reddit import (
    MAX_LIMIT, RedditConnector, RedditCredentials,
)


class FakeSubreddit:
    def __init__(self, values=None, error=None):
        self.values = [] if values is None else values
        self.error = error
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if self.error:
            raise self.error
        return iter(self.values)


class FakeReddit:
    def __init__(self, subreddit):
        self._subreddit = subreddit
        self.names = []

    def subreddit(self, name):
        self.names.append(name)
        return self._subreddit


def submissions():
    raw = json.loads(open(
        'tests/fixtures/social_posts.json', encoding='utf-8').read())
    return [SimpleNamespace(**item, author='synthetic-user') for item in raw]


def test_missing_credentials_are_named():
    with pytest.raises(ConnectorConfigurationError) as error:
        RedditCredentials.from_environment({'REDDIT_CLIENT_ID': None})
    assert 'REDDIT_CLIENT_ID' in str(error.value)
    assert 'REDDIT_CLIENT_SECRET' in str(error.value)
    assert 'REDDIT_USER_AGENT' in str(error.value)


def test_credentials_construct_client_without_logging_values(monkeypatch):
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(reddit_module.praw, 'Reddit', factory)
    credentials = RedditCredentials('id-value', 'secret-value', 'agent-value')
    RedditConnector(credentials)
    assert captured['client_id'] == 'id-value'
    assert captured['client_secret'] == 'secret-value'
    assert captured['user_agent'] == 'agent-value'


def test_search_normalizes_fixture_and_omits_author_by_default():
    source = FakeSubreddit(submissions())
    connector = RedditConnector(reddit=FakeReddit(source))
    posts = connector.search('sample phone', limit=3, sort='new', time_filter='week')
    assert len(posts) == 3
    assert source.calls == [('sample phone', {
        'sort': 'new', 'time_filter': 'week', 'limit': 3})]
    assert posts[0].source == 'reddit'
    assert posts[0].query == 'sample phone'
    assert posts[0].text == 'Battery feedback\n\nI love the battery life'
    assert posts[0].url.startswith('https://www.reddit.com/')
    assert posts[0].created_at.tzinfo == timezone.utc
    assert posts[0].author_id_or_name is None
    assert posts[0].engagement == {'score': 12, 'comment_count': 3}
    assert posts[0].metadata['subreddit'] == 'example'


def test_author_collection_is_explicit_opt_in():
    connector = RedditConnector(
        reddit=FakeReddit(FakeSubreddit(submissions()[:1])),
        collect_author=True)
    assert connector.search('query', 1)[0].author_id_or_name == 'synthetic-user'


def test_removed_body_and_malformed_records_are_handled(caplog):
    valid = submissions()[0]
    valid.selftext = '[removed]'
    malformed = SimpleNamespace(title='', selftext='', id='', created_utc='bad')
    connector = RedditConnector(
        reddit=FakeReddit(FakeSubreddit([malformed, valid])))
    posts = connector.search('query', 2)
    assert len(posts) == 1
    assert posts[0].text == 'Battery feedback'
    assert 'Skipping malformed' in caplog.text


def test_empty_results_are_valid():
    connector = RedditConnector(reddit=FakeReddit(FakeSubreddit()))
    assert connector.search('nothing', 10) == []


@pytest.mark.parametrize('kwargs,error', [
    ({'query': ''}, ValueError),
    ({'query': 'q', 'limit': 0}, ValueError),
    ({'query': 'q', 'limit': MAX_LIMIT + 1}, ValueError),
    ({'query': 'q', 'limit': 1.5}, TypeError),
    ({'query': 'q', 'sort': 'hot'}, ValueError),
    ({'query': 'q', 'time_filter': 'hour'}, ValueError),
])
def test_search_parameter_validation(kwargs, error):
    connector = RedditConnector(reddit=FakeReddit(FakeSubreddit()))
    with pytest.raises(error):
        connector.search(**kwargs)


@pytest.mark.parametrize('exception_name,expected', [
    ('TooManyRequests', ConnectorRateLimitError),
    ('OAuthException', ConnectorAuthenticationError),
    ('RequestException', ConnectorRequestError),
])
def test_remote_failures_are_translated(monkeypatch, exception_name, expected):
    class FakeRemoteError(Exception):
        pass

    monkeypatch.setattr(reddit_module, exception_name, FakeRemoteError)
    connector = RedditConnector(
        reddit=FakeReddit(FakeSubreddit(error=FakeRemoteError())))
    with pytest.raises(expected):
        connector.search('query', 1)

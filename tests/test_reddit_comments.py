"""Offline tests for bounded Reddit top-level comment collection."""
import json
from pathlib import Path
from types import SimpleNamespace

from src.social.reddit import RedditConnector


class FakeComments(list):
    def __init__(self, values):
        super().__init__(values)
        self.replace_calls = []

    def replace_more(self, **kwargs):
        self.replace_calls.append(kwargs)


class FakeReddit:
    def __init__(self, submission):
        self.submission = submission

    def subreddit(self, name):
        return SimpleNamespace(search=lambda *args, **kwargs: [self.submission])


def test_reddit_comment_collection_is_bounded_and_normalized():
    fixture = json.loads(Path(
        'tests/fixtures/reddit_api_response.json').read_text(encoding='utf-8'))
    comments = FakeComments([
        SimpleNamespace(**item, author='unused')
        for item in fixture['comments']])
    submission = SimpleNamespace(
        **fixture['submission'], comments=comments, author='unused')
    posts = RedditConnector(reddit=FakeReddit(submission)).search(
        'synthetic phone', limit=1, include_comments=True,
        max_comments_per_submission=1, time_filter='year')
    assert len(posts) == 2
    assert posts[1].metadata['record_type'] == 'comment'
    assert posts[1].metadata['submission_id'] == 'synthetic-submission'
    assert posts[1].author_id_or_name is None
    assert comments.replace_calls == [{'limit': 0}]

"""Credential-gated, bounded Reddit submission search through PRAW."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
from typing import Mapping

import praw
from prawcore.exceptions import (
    OAuthException, RequestException, ResponseException, ServerError,
    TooManyRequests,
)

from .base import (
    ConnectorAuthenticationError,
    ConnectorConfigurationError,
    ConnectorRateLimitError,
    ConnectorRequestError,
    SocialConnector,
)
from .schemas import SocialPost

logger = logging.getLogger(__name__)
DEFAULT_LIMIT = 50
MAX_LIMIT = 500
SORT_VALUES = frozenset({'relevance', 'new', 'top'})
TIME_FILTER_VALUES = frozenset({'day', 'week', 'month', 'year', 'all'})


@dataclass(frozen=True)
class RedditCredentials:
    client_id: str
    client_secret: str
    user_agent: str

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> RedditCredentials:
        env = os.environ if environment is None else environment
        names = ('REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'REDDIT_USER_AGENT')
        values = {
            name: value.strip() if isinstance(value := env.get(name), str) else ''
            for name in names}
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ConnectorConfigurationError(
                f"Missing Reddit environment variables: {', '.join(missing)}")
        return cls(values[names[0]], values[names[1]], values[names[2]])


class RedditConnector(SocialConnector):
    """Search public Reddit submissions and return generic SocialPost objects."""

    source = 'reddit'

    def __init__(
        self,
        credentials: RedditCredentials | None = None,
        *,
        reddit=None,
        collect_author: bool = False,
    ) -> None:
        self.collect_author = collect_author
        if reddit is not None:
            self._reddit = reddit
        else:
            credentials = credentials or RedditCredentials.from_environment()
            self._reddit = praw.Reddit(
                client_id=credentials.client_id,
                client_secret=credentials.client_secret,
                user_agent=credentials.user_agent,
                check_for_async=False,
            )

    @staticmethod
    def _validate(query: str, limit: int, sort: str, time_filter: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError('query must be a non-empty string')
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError('limit must be an integer')
        if not 1 <= limit <= MAX_LIMIT:
            raise ValueError(f'limit must be between 1 and {MAX_LIMIT}')
        if sort not in SORT_VALUES:
            raise ValueError(f'sort must be one of {sorted(SORT_VALUES)}')
        if time_filter not in TIME_FILTER_VALUES:
            raise ValueError(
                f'time_filter must be one of {sorted(TIME_FILTER_VALUES)}')
        return query.strip()

    def _normalize(self, submission, query: str) -> SocialPost:
        title = str(getattr(submission, 'title', '') or '').strip()
        body = str(getattr(submission, 'selftext', '') or '').strip()
        if body in {'[removed]', '[deleted]'}:
            body = ''
        text = '\n\n'.join(part for part in (title, body) if part)
        post_id = str(getattr(submission, 'id', '') or '').strip()
        created = float(getattr(submission, 'created_utc'))
        permalink = str(getattr(submission, 'permalink', '') or '').strip()
        subreddit = str(getattr(submission, 'subreddit', '') or '').strip()
        author = getattr(submission, 'author', None)
        author_value = str(author) if self.collect_author and author else None
        return SocialPost(
            source=self.source,
            post_id=post_id,
            created_at=datetime.fromtimestamp(created, tz=timezone.utc),
            text=text,
            query=query,
            url=f'https://www.reddit.com{permalink}' if permalink else None,
            author_id_or_name=author_value,
            engagement={
                'score': int(getattr(submission, 'score', 0) or 0),
                'comment_count': int(getattr(submission, 'num_comments', 0) or 0),
            },
            metadata={'subreddit': subreddit, 'record_type': 'submission'},
        )

    def search(
        self,
        query: str,
        limit: int = DEFAULT_LIMIT,
        *,
        sort: str = 'relevance',
        time_filter: str = 'all',
    ) -> list[SocialPost]:
        query = self._validate(query, limit, sort, time_filter)
        try:
            listing = self._reddit.subreddit('all').search(
                query, sort=sort, time_filter=time_filter, limit=limit)
            posts = []
            for submission in listing:
                try:
                    posts.append(self._normalize(submission, query))
                except (AttributeError, OverflowError, TypeError, ValueError) as exc:
                    logger.warning('Skipping malformed Reddit submission: %s', exc)
            return posts
        except TooManyRequests as exc:
            raise ConnectorRateLimitError('Reddit rate limit reached; retry later') from exc
        except OAuthException as exc:
            raise ConnectorAuthenticationError('Reddit OAuth authentication failed') from exc
        except ResponseException as exc:
            status = getattr(exc, 'response', None)
            status_code = getattr(status, 'status_code', None)
            if status_code in {401, 403}:
                raise ConnectorAuthenticationError(
                    'Reddit rejected the application credentials') from exc
            if status_code == 429:
                raise ConnectorRateLimitError(
                    'Reddit rate limit reached; retry later') from exc
            raise ConnectorRequestError(
                f'Reddit returned HTTP {status_code or "error"}') from exc
        except (RequestException, ServerError) as exc:
            raise ConnectorRequestError('Reddit request failed') from exc

"""Privacy-minimized YouTube Data API v3 public comment connector."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import unescape
import json
import logging
import os
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .base import (
    ConnectorAuthenticationError,
    ConnectorConfigurationError,
    ConnectorContentUnavailableError,
    ConnectorRateLimitError,
    ConnectorRequestError,
    SocialConnector,
)
from .schemas import SocialPost

logger = logging.getLogger(__name__)
API_ROOT = 'https://www.googleapis.com/youtube/v3'
DEFAULT_MAX_VIDEOS = 5
DEFAULT_MAX_COMMENTS = 20
MAX_VIDEOS = 50
MAX_COMMENTS = 100
RequestJson = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class YouTubeCredentials:
    """YouTube API credentials sourced without logging the key."""

    api_key: str

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> YouTubeCredentials:
        env = os.environ if environment is None else environment
        value = env.get('YOUTUBE_API_KEY')
        key = value.strip() if isinstance(value, str) else ''
        if not key:
            raise ConnectorConfigurationError(
                'Missing YouTube environment variable: YOUTUBE_API_KEY')
        return cls(key)


class YouTubeConnector(SocialConnector):
    """Search public videos and return bounded top-level public comments."""

    source = 'youtube'

    def __init__(
        self,
        credentials: YouTubeCredentials | None = None,
        *,
        request_json: RequestJson | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        credentials = credentials or YouTubeCredentials.from_environment()
        self._api_key = credentials.api_key
        self._request_json = request_json
        if timeout_seconds <= 0:
            raise ValueError('timeout_seconds must be positive')
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _validate(
        query: str, max_videos_per_query: int, max_comments_per_video: int
    ) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError('query must be a non-empty string')
        for name, value, maximum in (
            ('max_videos_per_query', max_videos_per_query, MAX_VIDEOS),
            ('max_comments_per_video', max_comments_per_video, MAX_COMMENTS),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f'{name} must be an integer')
            if not 1 <= value <= maximum:
                raise ValueError(f'{name} must be between 1 and {maximum}')
        return query.strip()

    @staticmethod
    def _error_reason(payload: Mapping[str, Any]) -> str:
        try:
            errors = payload['error']['errors']
            return str(errors[0].get('reason', ''))
        except (KeyError, IndexError, TypeError, AttributeError):
            return ''

    def _request(self, endpoint: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._request_json is not None:
            result = self._request_json(endpoint, params)
            if not isinstance(result, Mapping):
                raise ConnectorRequestError('YouTube returned a malformed response')
            return result
        safe_params = {**params, 'key': self._api_key}
        url = f'{API_ROOT}/{endpoint}?{urlencode(safe_params)}'
        request = Request(url, headers={'Accept': 'application/json'})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode('utf-8'))
        except HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = {}
            reason = self._error_reason(payload)
            if exc.code in {401} or reason in {'keyInvalid', 'accessNotConfigured'}:
                raise ConnectorAuthenticationError(
                    'YouTube rejected the API credentials') from exc
            if exc.code == 429 or reason in {'quotaExceeded', 'dailyLimitExceeded'}:
                raise ConnectorRateLimitError(
                    'YouTube API quota or rate limit reached') from exc
            if reason in {'commentsDisabled', 'videoNotFound', 'commentThreadNotFound'}:
                raise ConnectorContentUnavailableError(
                    'YouTube comments are disabled or the public content is unavailable') from exc
            raise ConnectorRequestError(
                f'YouTube returned HTTP {exc.code}') from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ConnectorRequestError('YouTube request failed') from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ConnectorRequestError('YouTube returned invalid JSON') from exc
        if not isinstance(payload, Mapping):
            raise ConnectorRequestError('YouTube returned a malformed response')
        return payload

    @staticmethod
    def _video_ids(payload: Mapping[str, Any]) -> list[str]:
        result: list[str] = []
        for item in payload.get('items', []):
            try:
                video_id = str(item['id']['videoId']).strip()
            except (KeyError, TypeError, AttributeError):
                continue
            if video_id:
                result.append(video_id)
        return result

    def _normalize_comment(
        self, item: Mapping[str, Any], query: str, video_id: str
    ) -> SocialPost:
        top_level = item['snippet']['topLevelComment']
        comment_id = str(top_level['id']).strip()
        snippet = top_level['snippet']
        text = unescape(str(
            snippet.get('textOriginal') or snippet.get('textDisplay') or '')).strip()
        published = str(snippet['publishedAt'])
        created_at = datetime.fromisoformat(published.replace('Z', '+00:00'))
        return SocialPost(
            source=self.source,
            post_id=comment_id,
            created_at=created_at,
            text=text,
            query=query,
            url=f'https://www.youtube.com/watch?v={video_id}&lc={comment_id}',
            author_id_or_name=None,
            parent_id=video_id,
            engagement={'like_count': int(snippet.get('likeCount', 0) or 0)},
            metadata={'video_id': video_id, 'record_type': 'comment'},
        )

    def search(
        self,
        query: str,
        limit: int = DEFAULT_MAX_COMMENTS,
        *,
        max_videos_per_query: int = DEFAULT_MAX_VIDEOS,
        max_comments_per_video: int | None = None,
    ) -> list[SocialPost]:
        """Return bounded top-level comments from bounded matching videos."""

        comments_per_video = limit if max_comments_per_video is None else max_comments_per_video
        query = self._validate(query, max_videos_per_query, comments_per_video)
        search_payload = self._request('search', {
            'part': 'snippet', 'type': 'video', 'q': query,
            'maxResults': max_videos_per_query,
        })
        posts: list[SocialPost] = []
        for video_id in self._video_ids(search_payload)[:max_videos_per_query]:
            try:
                payload = self._request('commentThreads', {
                    'part': 'snippet', 'videoId': video_id,
                    'maxResults': comments_per_video,
                    'order': 'relevance', 'textFormat': 'plainText',
                })
            except ConnectorContentUnavailableError:
                logger.info('Skipping video with unavailable comments')
                continue
            for item in payload.get('items', [])[:comments_per_video]:
                try:
                    posts.append(self._normalize_comment(item, query, video_id))
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    logger.warning('Skipping malformed YouTube comment: %s', exc)
        return posts

"""Validated, human-editable configuration for external-domain collection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.config import PROJECT_ROOT

DEFAULT_CONFIG_PATH = PROJECT_ROOT / 'config' / 'external_eval_queries.json'


def _positive_int(name: str, value: Any, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'{name} must be an integer')
    if not 1 <= value <= maximum:
        raise ValueError(f'{name} must be between 1 and {maximum}')
    return value


@dataclass(frozen=True)
class RedditCollectionConfig:
    enabled: bool = True
    per_query_limit: int = 20
    sort: str = 'relevance'
    time_filter: str = 'year'
    include_comments: bool = True
    max_comments_per_submission: int = 10

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RedditCollectionConfig:
        config = cls(**dict(value))
        _positive_int('reddit.per_query_limit', config.per_query_limit, 500)
        if config.sort not in {'relevance', 'new', 'top'}:
            raise ValueError('reddit.sort must be relevance, new, or top')
        if config.time_filter not in {'day', 'week', 'month', 'year', 'all'}:
            raise ValueError('reddit.time_filter is invalid')
        if not isinstance(config.enabled, bool) or not isinstance(config.include_comments, bool):
            raise TypeError('reddit enabled/include_comments values must be booleans')
        if (isinstance(config.max_comments_per_submission, bool) or
                not isinstance(config.max_comments_per_submission, int) or
                not 0 <= config.max_comments_per_submission <= 100):
            raise ValueError(
                'reddit.max_comments_per_submission must be between 0 and 100')
        return config


@dataclass(frozen=True)
class YouTubeCollectionConfig:
    enabled: bool = True
    max_videos_per_query: int = 5
    max_comments_per_video: int = 20

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> YouTubeCollectionConfig:
        config = cls(**dict(value))
        if not isinstance(config.enabled, bool):
            raise TypeError('youtube.enabled must be a boolean')
        _positive_int('youtube.max_videos_per_query', config.max_videos_per_query, 50)
        _positive_int('youtube.max_comments_per_video', config.max_comments_per_video, 100)
        return config


@dataclass(frozen=True)
class ExternalEvalConfig:
    queries: tuple[str, ...]
    target_total: int
    per_query_target: int
    sampling_seed: int
    max_per_context: int
    reddit: RedditCollectionConfig
    youtube: YouTubeCollectionConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ExternalEvalConfig:
        unknown = set(value) - {
            'queries', 'target_total', 'per_query_target', 'sampling_seed',
            'max_per_context', 'sources'}
        if unknown:
            raise ValueError(f'Unknown external-evaluation config fields: {sorted(unknown)}')
        raw_queries = value.get('queries')
        if not isinstance(raw_queries, list) or not raw_queries:
            raise ValueError('queries must be a non-empty list')
        queries = tuple(str(item).strip() for item in raw_queries)
        if any(not item for item in queries) or len(set(queries)) != len(queries):
            raise ValueError('queries must contain unique non-empty strings')
        target_total = _positive_int('target_total', value.get('target_total'), 500)
        per_query_target = _positive_int(
            'per_query_target', value.get('per_query_target'), 500)
        seed = value.get('sampling_seed', 42)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError('sampling_seed must be a non-negative integer')
        max_per_context = _positive_int(
            'max_per_context', value.get('max_per_context', 20), 500)
        sources = value.get('sources')
        if not isinstance(sources, Mapping):
            raise ValueError('sources must be an object')
        reddit = RedditCollectionConfig.from_mapping(sources.get('reddit', {}))
        youtube = YouTubeCollectionConfig.from_mapping(sources.get('youtube', {}))
        if not reddit.enabled and not youtube.enabled:
            raise ValueError('at least one collection source must be enabled')
        return cls(
            queries, target_total, per_query_target, seed, max_per_context,
            reddit, youtube)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value['queries'] = list(self.queries)
        value['sources'] = {
            'reddit': value.pop('reddit'),
            'youtube': value.pop('youtube'),
        }
        return value

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True,
            separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(payload).hexdigest()


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> ExternalEvalConfig:
    """Load and validate an external-evaluation JSON configuration."""

    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise FileNotFoundError(f'Collection config not found: {config_path}') from None
    except json.JSONDecodeError as exc:
        raise ValueError(f'Collection config is invalid JSON: {exc}') from exc
    if not isinstance(payload, Mapping):
        raise ValueError('Collection config root must be an object')
    return ExternalEvalConfig.from_mapping(payload)

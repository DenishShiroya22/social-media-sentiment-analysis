"""Platform-neutral social-post and sentiment-enriched record schemas."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import re
from typing import Any, Mapping

from src.config import LABELS
from src.inference import SentimentPrediction

SOURCE_PATTERN = re.compile(r'^[a-z][a-z0-9_-]*$')


def _utc_datetime(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError('created_at must be a timezone-aware datetime')
    return value.astimezone(timezone.utc)


def _validate_base(source: str, post_id: str, created_at: datetime,
                   text: str, query: str, engagement: Mapping[str, float]) -> None:
    if not isinstance(source, str) or not SOURCE_PATTERN.fullmatch(source):
        raise ValueError('source must be a lowercase platform identifier')
    for name, value in [('post_id', post_id), ('text', text), ('query', query)]:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{name} must be a non-empty string')
    _utc_datetime(created_at)
    if not isinstance(engagement, Mapping):
        raise ValueError('engagement must be a mapping')
    for key, value in engagement.items():
        if not isinstance(key, str) or not key:
            raise ValueError('engagement keys must be non-empty strings')
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('engagement values must be numeric')
        if not math.isfinite(float(value)):
            raise ValueError('engagement values must be finite')


@dataclass(frozen=True)
class SocialPost:
    """Minimal normalized public social-media record."""

    source: str
    post_id: str
    created_at: datetime
    text: str
    query: str
    url: str | None = None
    author_id_or_name: str | None = None
    parent_id: str | None = None
    engagement: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_base(self.source, self.post_id, self.created_at, self.text,
                       self.query, self.engagement)
        if self.author_id_or_name is not None:
            if (not isinstance(self.author_id_or_name, str) or
                    not self.author_id_or_name.strip()):
                raise ValueError('author_id_or_name must be non-empty when provided')
        if not isinstance(self.metadata, Mapping):
            raise ValueError('metadata must be a mapping')

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result['created_at'] = (
            _utc_datetime(self.created_at).isoformat().replace('+00:00', 'Z'))
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SocialPost:
        data = dict(value)
        created = data.get('created_at')
        if isinstance(created, str):
            data['created_at'] = datetime.fromisoformat(created.replace('Z', '+00:00'))
        return cls(**data)


@dataclass(frozen=True)
class AnalyzedPost:
    """Normalized post enriched with frozen-model sentiment output."""

    source: str
    post_id: str
    created_at: datetime
    text: str
    query: str
    clean_text: str
    sentiment: str
    confidence: float
    prob_negative: float
    prob_neutral: float
    prob_positive: float
    url: str | None = None
    author_id_or_name: str | None = None
    parent_id: str | None = None
    engagement: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_base(self.source, self.post_id, self.created_at, self.text,
                       self.query, self.engagement)
        if self.author_id_or_name is not None:
            if (not isinstance(self.author_id_or_name, str) or
                    not self.author_id_or_name.strip()):
                raise ValueError('author_id_or_name must be non-empty when provided')
        if not isinstance(self.metadata, Mapping):
            raise ValueError('metadata must be a mapping')
        if not isinstance(self.clean_text, str) or not self.clean_text:
            raise ValueError('clean_text must be a non-empty string')
        if self.sentiment not in LABELS:
            raise ValueError(f'sentiment must be one of {LABELS}')
        probabilities = (self.prob_negative, self.prob_neutral, self.prob_positive)
        if any(not math.isfinite(value) or value < 0 or value > 1
               for value in probabilities):
            raise ValueError('probabilities must be finite values in [0, 1]')
        if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError('probabilities must sum to one')
        if not math.isclose(self.confidence, max(probabilities),
                            rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError('confidence must equal the maximum probability')

    @classmethod
    def from_prediction(
        cls, post: SocialPost, prediction: SentimentPrediction
    ) -> AnalyzedPost:
        if post.text != prediction.text:
            raise ValueError('Prediction text does not match the source post')
        base = post.to_dict()
        base['created_at'] = post.created_at
        return cls(
            **base,
            clean_text=prediction.clean_text,
            sentiment=prediction.sentiment,
            confidence=prediction.confidence,
            prob_negative=prediction.probabilities['negative'],
            prob_neutral=prediction.probabilities['neutral'],
            prob_positive=prediction.probabilities['positive'],
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result['created_at'] = (
            _utc_datetime(self.created_at).isoformat().replace('+00:00', 'Z'))
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AnalyzedPost:
        data = dict(value)
        created = data.get('created_at')
        if isinstance(created, str):
            data['created_at'] = datetime.fromisoformat(created.replace('Z', '+00:00'))
        return cls(**data)

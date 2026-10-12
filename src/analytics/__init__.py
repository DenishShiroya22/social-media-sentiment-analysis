"""Reusable aggregate sentiment analytics for local application data."""

from .confidence import confidence_summary
from .summary import (
    representative_comments,
    sentiment_by_group,
    sentiment_summary,
)
from .topics import topic_report
from .trends import sentiment_trends

__all__ = [
    "confidence_summary",
    "representative_comments",
    "sentiment_by_group",
    "sentiment_summary",
    "sentiment_trends",
    "topic_report",
]

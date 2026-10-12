"""Offline tests for aggregate analytics and topic extraction."""
from __future__ import annotations

import pytest

from src.analytics.confidence import confidence_summary
from src.analytics.summary import (
    representative_comments,
    sentiment_by_group,
    sentiment_summary,
)
from src.analytics.topics import extract_terms, theme_summary, topic_report
from src.analytics.trends import sentiment_trends


def records():
    return [
        {
            "text": "Battery drains quickly and charging is slow",
            "sentiment": "negative",
            "confidence": 0.9,
            "query": "phone",
            "created_at": "2026-10-01T10:00:00Z",
        },
        {
            "text": "Battery quality is excellent and charging is fast",
            "sentiment": "positive",
            "confidence": 0.8,
            "query": "phone",
            "created_at": "2026-10-02T10:00:00Z",
        },
        {
            "text": "Delivery happened today",
            "sentiment": "neutral",
            "confidence": 0.6,
            "query": "delivery",
            "created_at": "2026-10-08T10:00:00Z",
        },
        {
            "text": "Delivery was late and customer support was terrible",
            "sentiment": "negative",
            "confidence": 0.7,
            "query": "delivery",
            "created_at": "",
        },
    ]


def test_summary_counts_percentages_ratio_and_confidence():
    result = sentiment_summary(records())
    assert result["total_posts"] == 4
    assert result["sentiment_counts"] == {
        "negative": 2,
        "neutral": 1,
        "positive": 1,
    }
    assert sum(result["sentiment_percentages"].values()) == pytest.approx(100)
    assert result["positive_negative_ratio"] == pytest.approx(0.5)
    assert result["average_confidence"] == pytest.approx(0.75)
    assert result["median_confidence"] == pytest.approx(0.75)


def test_empty_summary_and_grouped_summary_are_safe():
    empty = sentiment_summary([])
    assert empty["total_posts"] == 0
    assert sum(empty["sentiment_percentages"].values()) == 0
    assert empty["average_confidence"] is None
    grouped = sentiment_by_group(records())
    assert grouped["phone"]["total_posts"] == 2
    assert grouped["delivery"]["sentiment_counts"]["negative"] == 1


def test_confidence_summary_by_sentiment():
    result = confidence_summary(records())
    assert result["overall"]["count"] == 4
    assert result["overall"]["median"] == pytest.approx(0.75)
    assert result["by_sentiment"]["negative"]["average"] == pytest.approx(0.8)
    assert result["by_sentiment"]["positive"]["maximum"] == pytest.approx(0.8)


def test_daily_and_weekly_trends_skip_missing_timestamps():
    daily = sentiment_trends(records(), frequency="daily")
    assert [row["period"] for row in daily] == [
        "2026-10-01",
        "2026-10-02",
        "2026-10-08",
    ]
    assert daily[0]["negative_percentage"] == 100
    weekly = sentiment_trends(records(), frequency="weekly")
    assert [row["period"] for row in weekly] == ["2026-09-28", "2026-10-05"]
    assert weekly[0]["post_count"] == 2


def test_topic_output_schema_and_configurable_themes():
    negative = extract_terms(records(), sentiment="negative", top_n=5)
    assert negative
    assert {"term", "tfidf_score", "document_frequency"} == set(negative[0])
    themes = theme_summary(records())
    assert {row["topic"] for row in themes} >= {
        "battery",
        "delivery",
        "customer support",
    }
    report = topic_report(records(), top_n=5)
    assert {"negative_terms", "positive_terms", "themes", "method"} == set(report)


def test_representative_comments_are_bounded_and_private_payload_only():
    result = representative_comments(records(), limit=1)
    assert result["highest_confidence_negative"][0]["confidence"] == 0.9
    assert result["recent_positive"][0]["query"] == "phone"
    assert set(result["recent_positive"][0]) == {
        "text",
        "query",
        "created_at",
        "confidence",
    }

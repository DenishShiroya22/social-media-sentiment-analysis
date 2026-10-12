"""Sentiment totals, grouped summaries, and private representative examples."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import math
from statistics import mean, median
from typing import Mapping, Sequence

from src.config import LABELS


def _sentiment(record: Mapping) -> str:
    value = str(record.get("sentiment", "")).strip().lower()
    if value not in LABELS:
        raise ValueError(f"Unknown sentiment label: {value or '<blank>'}")
    return value


def _confidence_values(records: Sequence[Mapping]) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get("confidence")
        if value is None or value == "":
            continue
        number = float(value)
        if not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError("confidence must be a finite value between 0 and 1")
        values.append(number)
    return values


def sentiment_summary(records: Sequence[Mapping]) -> dict:
    """Return stable counts, percentages, ratios, and confidence summaries."""

    counts = Counter(_sentiment(record) for record in records)
    total = len(records)
    percentages = {
        label: (counts[label] / total * 100 if total else 0.0)
        for label in LABELS
    }
    confidences = _confidence_values(records)
    negative = counts["negative"]
    return {
        "total_posts": total,
        "sentiment_counts": {label: counts[label] for label in LABELS},
        "sentiment_percentages": percentages,
        "positive_negative_ratio": (
            counts["positive"] / negative if negative else None
        ),
        "average_confidence": mean(confidences) if confidences else None,
        "median_confidence": median(confidences) if confidences else None,
        "confidence_rows": len(confidences),
        "confidence_caveat": (
            "Model confidence is a relative score and is not guaranteed to be "
            "a calibrated probability."
        ),
    }


def sentiment_by_group(
    records: Sequence[Mapping], group_key: str = "query"
) -> dict[str, dict]:
    """Group records by a nonempty field and summarize each group."""

    groups: dict[str, list[Mapping]] = {}
    for record in records:
        group = str(record.get(group_key, "")).strip()
        if not group:
            group = "Unknown"
        groups.setdefault(group, []).append(record)
    return {
        group: sentiment_summary(group_records)
        for group, group_records in sorted(groups.items())
    }


def _timestamp(record: Mapping) -> datetime:
    raw = str(record.get("created_at", "")).strip()
    if not raw:
        return datetime.min
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return datetime.min


def representative_comments(
    records: Sequence[Mapping], *, limit: int = 3
) -> dict[str, list[dict]]:
    """Select local-only high-confidence and recent positive/negative examples."""

    if limit < 1:
        raise ValueError("limit must be positive")
    result: dict[str, list[dict]] = {}
    for label in ("negative", "positive"):
        matching = [record for record in records if _sentiment(record) == label]
        high = sorted(
            matching,
            key=lambda record: float(record.get("confidence") or 0.0),
            reverse=True,
        )[:limit]
        recent = sorted(matching, key=_timestamp, reverse=True)[:limit]
        for name, selected in (("highest_confidence", high), ("recent", recent)):
            result[f"{name}_{label}"] = [
                {
                    "text": str(record.get("text", "")),
                    "query": str(record.get("query", "")),
                    "created_at": str(record.get("created_at", "")),
                    "confidence": float(record.get("confidence") or 0.0),
                }
                for record in selected
            ]
    return result

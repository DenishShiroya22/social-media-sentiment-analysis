"""Daily and weekly sentiment trend aggregation."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Mapping, Sequence

from src.config import LABELS


def _period(value: object, frequency: str) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        moment = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    day = moment.date()
    if frequency == "weekly":
        day -= timedelta(days=day.weekday())
    return day.isoformat()


def sentiment_trends(
    records: Sequence[Mapping], *, frequency: str = "daily"
) -> list[dict]:
    """Aggregate available timestamps without filling artificial sparse periods."""

    if frequency not in {"daily", "weekly"}:
        raise ValueError("frequency must be 'daily' or 'weekly'")
    buckets: dict[str, Counter] = {}
    for record in records:
        period = _period(record.get("created_at"), frequency)
        if period is None:
            continue
        label = str(record.get("sentiment", "")).strip().lower()
        if label not in LABELS:
            raise ValueError(f"Unknown sentiment label: {label or '<blank>'}")
        buckets.setdefault(period, Counter())[label] += 1

    result = []
    for period, counts in sorted(buckets.items()):
        total = sum(counts.values())
        row = {"period": period, "post_count": total}
        for label in LABELS:
            row[f"{label}_count"] = counts[label]
            row[f"{label}_percentage"] = counts[label] / total * 100
        result.append(row)
    return result

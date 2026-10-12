"""Confidence summaries for frozen-model predictions."""
from __future__ import annotations

from statistics import mean, median
from typing import Mapping, Sequence

from src.config import LABELS


def confidence_summary(records: Sequence[Mapping]) -> dict:
    """Summarize confidence overall and by predicted sentiment."""

    values: list[float] = []
    grouped: dict[str, list[float]] = {label: [] for label in LABELS}
    for record in records:
        label = str(record.get("sentiment", "")).strip().lower()
        if label not in LABELS:
            raise ValueError(f"Unknown sentiment label: {label or '<blank>'}")
        value = float(record.get("confidence", 0.0))
        if not 0 <= value <= 1:
            raise ValueError("confidence must be between 0 and 1")
        values.append(value)
        grouped[label].append(value)

    def describe(items: list[float]) -> dict:
        return {
            "count": len(items),
            "average": mean(items) if items else None,
            "median": median(items) if items else None,
            "minimum": min(items) if items else None,
            "maximum": max(items) if items else None,
        }

    return {
        "overall": describe(values),
        "by_sentiment": {label: describe(grouped[label]) for label in LABELS},
        "calibration_warning": (
            "Confidence is the model's maximum class score; it is not a "
            "calibrated certainty estimate."
        ),
    }

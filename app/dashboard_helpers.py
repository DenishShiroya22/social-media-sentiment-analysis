"""Pure helpers used by the Streamlit dashboard and offline tests."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

from src.analytics.summary import (
    representative_comments,
    sentiment_by_group,
    sentiment_summary,
)
from src.analytics.topics import topic_report
from src.analytics.trends import sentiment_trends
from src.inference import SentimentPredictor
from src.social.youtube import YouTubeConnector


SUPPORTED_SUFFIXES = {".csv", ".jsonl"}


def discover_local_datasets(root: Path | str) -> list[Path]:
    """Return private local datasets suitable for interactive analysis."""

    root = Path(root)
    candidates = [
        *root.glob("external_eval/labels/external_eval_labeled.csv"),
        *root.glob("external_eval/prepared/external_prepared_*.jsonl"),
        *root.glob("analyzed/*.jsonl"),
        *root.glob("collected/*.jsonl"),
    ]
    return sorted(
        {path.resolve() for path in candidates if path.is_file()},
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def load_local_rows(path: Path | str) -> list[dict]:
    """Load CSV or JSONL rows and require nonempty text."""

    source = Path(path)
    if source.suffix.casefold() not in SUPPORTED_SUFFIXES:
        raise ValueError("Local dataset must be a CSV or JSONL file")
    if source.suffix.casefold() == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    else:
        rows = []
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSONL at line {line_number}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL line {line_number} is not an object")
                rows.append(value)
    if not rows:
        raise ValueError("The selected dataset is empty")
    sanitized = []
    for index, row in enumerate(rows, start=1):
        text = str(row.get("text", "")).strip()
        if not text:
            raise ValueError(f"Row {index} has no text")
        sanitized.append({
            key: value
            for key, value in row.items()
            if key not in {"author", "author_id_or_name", "username"}
        })
    return sanitized


def analyze_rows(
    rows: Sequence[Mapping], predictor: SentimentPredictor | None = None
) -> list[dict]:
    """Run one frozen-model batch and return private in-memory enriched rows."""

    if not rows:
        return []
    predictor = predictor or SentimentPredictor()
    predictions = predictor.predict_batch([str(row["text"]) for row in rows])
    return [
        {
            **dict(row),
            "sentiment": prediction.sentiment,
            "confidence": prediction.confidence,
            "probability_negative": prediction.probabilities["negative"],
            "probability_neutral": prediction.probabilities["neutral"],
            "probability_positive": prediction.probabilities["positive"],
        }
        for row, prediction in zip(rows, predictions, strict=True)
    ]


def collect_live_youtube(
    query: str,
    *,
    max_videos: int,
    comments_per_video: int,
    connector: YouTubeConnector | None = None,
) -> list[dict]:
    """Perform one explicitly bounded YouTube request sequence."""

    connector = connector or YouTubeConnector()
    posts = connector.search(
        query,
        limit=comments_per_video,
        max_videos_per_query=max_videos,
        max_comments_per_video=comments_per_video,
    )
    return [
        {
            key: value
            for key, value in post.to_dict().items()
            if key != "author_id_or_name"
        }
        for post in posts
    ]


def dashboard_payload(records: Sequence[Mapping], *, trend_frequency: str) -> dict:
    """Build all local dashboard views without persisting row-level text."""

    return {
        "summary": sentiment_summary(records),
        "per_query": sentiment_by_group(records),
        "trends": sentiment_trends(records, frequency=trend_frequency),
        "topics": topic_report(records),
        "representative_comments": representative_comments(records, limit=3),
    }

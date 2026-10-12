"""Offline tests for local dashboard data handling."""
from __future__ import annotations

import csv

import pytest

from app.dashboard_helpers import (
    analyze_rows,
    dashboard_payload,
    discover_local_datasets,
    load_local_rows,
)
from src.inference import SentimentPrediction


class FakePredictor:
    def predict_batch(self, texts):
        labels = ["negative", "positive"]
        return [
            SentimentPrediction(
                text=text,
                clean_text=text.casefold(),
                sentiment=labels[index % 2],
                confidence=0.8,
                probabilities={
                    "negative": 0.8 if labels[index % 2] == "negative" else 0.1,
                    "neutral": 0.1,
                    "positive": 0.8 if labels[index % 2] == "positive" else 0.1,
                },
                model_version="fixture",
                model_artifact_sha256="a" * 64,
                inference_timestamp="2026-10-12T00:00:00Z",
            )
            for index, text in enumerate(texts)
        ]


def test_load_csv_strips_author_fields(tmp_path):
    path = tmp_path / "comments.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["text", "query", "created_at", "author_id_or_name"]
        )
        writer.writeheader()
        writer.writerow({
            "text": "Synthetic comment",
            "query": "phone",
            "created_at": "2026-10-12T00:00:00Z",
            "author_id_or_name": "private-fixture",
        })
    rows = load_local_rows(path)
    assert rows[0]["text"] == "Synthetic comment"
    assert "author_id_or_name" not in rows[0]


def test_load_jsonl_rejects_malformed_or_empty_input(tmp_path):
    malformed = tmp_path / "bad.jsonl"
    malformed.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed JSONL"):
        load_local_rows(malformed)
    empty = tmp_path / "empty.csv"
    empty.write_text("text,query\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_local_rows(empty)


def test_analyze_and_dashboard_payload():
    rows = [
        {
            "text": "Bad battery",
            "query": "phone",
            "created_at": "2026-10-11T00:00:00Z",
        },
        {
            "text": "Great design",
            "query": "phone",
            "created_at": "2026-10-12T00:00:00Z",
        },
    ]
    analyzed = analyze_rows(rows, FakePredictor())
    assert [row["sentiment"] for row in analyzed] == ["negative", "positive"]
    payload = dashboard_payload(analyzed, trend_frequency="daily")
    assert payload["summary"]["total_posts"] == 2
    assert len(payload["trends"]) == 2
    assert "negative_terms" in payload["topics"]


def test_discover_local_datasets(tmp_path):
    labels = tmp_path / "external_eval" / "labels"
    labels.mkdir(parents=True)
    expected = labels / "external_eval_labeled.csv"
    expected.write_text("text\nSynthetic\n", encoding="utf-8")
    ignored = labels / "other.csv"
    ignored.write_text("text\nIgnored\n", encoding="utf-8")
    assert discover_local_datasets(tmp_path) == [expected.resolve()]

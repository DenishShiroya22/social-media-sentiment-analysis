"""Generate tracked aggregate analytics from a private local row-level dataset."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from src.config import PROJECT_ROOT
from src.inference import SentimentPredictor

from .confidence import confidence_summary
from .summary import sentiment_by_group, sentiment_summary
from .topics import topic_report
from .trends import sentiment_trends


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "analytics"


def load_private_csv(path: Path | str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Input dataset contains no rows")
    for index, row in enumerate(rows, start=2):
        if not str(row.get("text", "")).strip():
            raise ValueError(f"Row {index} has no text")
    return rows


def enrich_records(rows: Sequence[dict], predictor: SentimentPredictor) -> list[dict]:
    predictions = predictor.predict_batch([str(row["text"]) for row in rows])
    return [
        {
            **row,
            "sentiment": prediction.sentiment,
            "confidence": prediction.confidence,
            "probabilities": prediction.probabilities,
        }
        for row, prediction in zip(rows, predictions, strict=True)
    ]


def generate_aggregate_reports(
    input_path: Path | str,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    predictor: SentimentPredictor | None = None,
) -> dict:
    rows = load_private_csv(input_path)
    predictor = predictor or SentimentPredictor()
    records = enrich_records(rows, predictor)
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    analytics = {
        "status": "aggregate analytics complete",
        "generated_at": generated_at,
        "prediction_basis": "frozen model predictions",
        "model_version": predictor.model_version,
        "model_artifact_sha256": predictor.artifact_sha256,
        "summary": sentiment_summary(records),
        "confidence": confidence_summary(records),
        "per_query": sentiment_by_group(records),
        "daily_trends": sentiment_trends(records, frequency="daily"),
        "weekly_trends": sentiment_trends(records, frequency="weekly"),
        "privacy": "No row-level text or representative comments are stored here.",
    }
    topics = {
        "status": "aggregate topic extraction complete",
        "generated_at": generated_at,
        "prediction_basis": "frozen model predictions",
        **topic_report(records),
        "privacy": "Only aggregate terms and theme counts are stored here.",
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    analytics_path = destination / "external_analytics.json"
    topics_path = destination / "external_topics.json"
    analytics_path.write_text(
        json.dumps(analytics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    topics_path.write_text(
        json.dumps(topics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "analytics": analytics,
        "topics": topics,
        "paths": {"analytics": analytics_path, "topics": topics_path},
        "records": records,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    try:
        result = generate_aggregate_reports(args.input, output_dir=args.output_dir)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Rows analyzed: {result['analytics']['summary']['total_posts']}")
    print(f"Reports: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

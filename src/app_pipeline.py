"""Bounded social ingestion, batch sentiment enrichment, and local persistence."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import logging
from typing import Sequence

from .config import LABELS
from .inference import SentimentPredictor
from .social.base import ConnectorError, SocialConnector
from .social.reddit import RedditConnector
from .social.schemas import AnalyzedPost, SocialPost
from .storage.repository import (
    JsonlRepository, RunManifest, create_run_id,
)

logger = logging.getLogger(__name__)


def analyze_posts(
    posts: Sequence[SocialPost], predictor: SentimentPredictor
) -> list[AnalyzedPost]:
    """Batch-predict normalized posts and attach sentiment fields in order."""

    predictions = predictor.predict_batch([post.text for post in posts])
    if len(predictions) != len(posts):
        raise RuntimeError('Predictor returned the wrong result count')
    return [
        AnalyzedPost.from_prediction(post, prediction)
        for post, prediction in zip(posts, predictions, strict=True)
    ]


def summarize_posts(posts: Sequence[AnalyzedPost]) -> dict:
    """Return minimal sentiment counts, percentages, and mean confidence."""

    total = len(posts)
    counts = Counter(post.sentiment for post in posts)
    return {
        'analyzed_count': total,
        'sentiment_counts': {label: counts.get(label, 0) for label in LABELS},
        'sentiment_percentages': {
            label: (counts.get(label, 0)/total*100 if total else 0.0)
            for label in LABELS},
        'average_confidence': (
            sum(post.confidence for post in posts)/total if total else 0.0),
    }


def connector_version(connector: SocialConnector) -> str:
    if connector.source == 'reddit':
        return f"praw-{importlib.metadata.version('praw')}"
    return f'{connector.__class__.__module__}.{connector.__class__.__name__}'


def run_application(
    connector: SocialConnector,
    *,
    query: str,
    limit: int = 50,
    sort: str = 'relevance',
    time_filter: str = 'all',
    predictor: SentimentPredictor | None = None,
    storage: JsonlRepository | None = None,
    now: datetime | None = None,
) -> dict:
    """Fetch, persist, batch-analyze, persist, and manifest one bounded run."""

    predictor = predictor or SentimentPredictor()
    storage = storage or JsonlRepository()
    model_sha256 = predictor.artifact_sha256
    model_version = predictor.model_version
    created = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    run_id = create_run_id(created)
    posts = connector.search(
        query, limit=limit, sort=sort, time_filter=time_filter)
    collected_path = storage.write_records(
        'collected', connector.source, query, run_id, posts)
    analyzed = analyze_posts(posts, predictor)
    analyzed_path = storage.write_records(
        'analyzed', connector.source, query, run_id, analyzed)
    manifest = RunManifest(
        run_id=run_id,
        source=connector.source,
        query=query,
        retrieved_count=len(posts),
        analyzed_count=len(analyzed),
        created_at=created.isoformat().replace('+00:00', 'Z'),
        model_sha256=model_sha256,
        model_version=model_version,
        connector_version=connector_version(connector),
        collected_file=storage.relative_path(collected_path),
        output_file=storage.relative_path(analyzed_path),
    )
    manifest_path = storage.write_manifest(manifest)
    return {
        'run_id': run_id,
        'manifest': asdict(manifest),
        'manifest_file': storage.relative_path(manifest_path),
        'summary': summarize_posts(analyzed),
    }


def print_summary(query: str, result: dict) -> None:
    summary = result['summary']
    print(f'Query: {query}')
    print(f"Posts analyzed: {summary['analyzed_count']}")
    for label in LABELS:
        count = summary['sentiment_counts'][label]
        percentage = summary['sentiment_percentages'][label]
        print(f'{label.title()}: {count} ({percentage:.1f}%)')
    print(f"Average confidence: {summary['average_confidence']:.3f}")
    print(f"Saved: {result['manifest']['output_file']}")
    print(f"Manifest: {result['manifest_file']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Collect bounded public posts and run frozen-model sentiment inference.')
    parser.add_argument('--source', choices=['reddit'], default='reddit')
    parser.add_argument('--query', required=True)
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--sort', choices=['relevance', 'new', 'top'],
                        default='relevance')
    parser.add_argument('--time-filter',
                        choices=['day', 'week', 'month', 'year', 'all'],
                        default='all')
    parser.add_argument(
        '--include-authors', action='store_true',
        help='Store public Reddit author names; disabled by default for privacy.')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        connector = RedditConnector(collect_author=args.include_authors)
        result = run_application(
            connector, query=args.query, limit=args.limit,
            sort=args.sort, time_filter=args.time_filter)
    except (ConnectorError, ValueError, TypeError, FileNotFoundError) as exc:
        logger.error('%s', exc)
        return 2
    print_summary(args.query, result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

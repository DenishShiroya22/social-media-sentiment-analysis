"""API-first collection orchestration for external-domain evaluation."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Mapping, Sequence

from src.config import PROJECT_ROOT
from src.social.base import ConnectorError, SocialConnector
from src.social.reddit import RedditConnector
from src.social.schemas import SocialPost
from src.social.youtube import YouTubeConnector
from src.storage.repository import create_run_id

from .config import DEFAULT_CONFIG_PATH, ExternalEvalConfig, load_config

logger = logging.getLogger(__name__)
DEFAULT_DATA_ROOT = PROJECT_ROOT / 'data' / 'external_eval'


def _private_minimized(post: SocialPost) -> SocialPost:
    return replace(post, author_id_or_name=None)


def _write_jsonl(path: Path, posts: Sequence[SocialPost]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        for post in posts:
            handle.write(post.to_json() + '\n')


def _write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')


def collect_external_posts(
    config: ExternalEvalConfig,
    *,
    connectors: Mapping[str, SocialConnector] | None = None,
    output_root: Path | str = DEFAULT_DATA_ROOT,
    now: datetime | None = None,
) -> dict:
    """Collect bounded public records and write raw JSONL plus a safe manifest."""

    supplied = dict(connectors or {})
    root = Path(output_root)
    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    run_id = create_run_id(moment)
    requested_counts: dict[str, dict[str, int]] = {}
    retrieved_counts: dict[str, dict[str, int]] = {}
    all_posts: list[SocialPost] = []

    if config.reddit.enabled:
        connector = supplied.get('reddit') or RedditConnector(collect_author=False)
        requested_counts['reddit'] = {}
        retrieved_counts['reddit'] = {}
        for query in config.queries:
            expected = config.reddit.per_query_limit * (
                1 + (config.reddit.max_comments_per_submission
                     if config.reddit.include_comments else 0))
            requested_counts['reddit'][query] = expected
            posts = connector.search(
                query,
                limit=config.reddit.per_query_limit,
                sort=config.reddit.sort,
                time_filter=config.reddit.time_filter,
                include_comments=config.reddit.include_comments,
                max_comments_per_submission=(
                    config.reddit.max_comments_per_submission),
            )
            minimized = [_private_minimized(post) for post in posts]
            all_posts.extend(minimized)
            retrieved_counts['reddit'][query] = len(minimized)
            logger.info('source=reddit query=%r retrieved=%d', query, len(minimized))

    if config.youtube.enabled:
        connector = supplied.get('youtube') or YouTubeConnector()
        requested_counts['youtube'] = {}
        retrieved_counts['youtube'] = {}
        for query in config.queries:
            expected = (config.youtube.max_videos_per_query *
                        config.youtube.max_comments_per_video)
            requested_counts['youtube'][query] = expected
            posts = connector.search(
                query,
                limit=config.youtube.max_comments_per_video,
                max_videos_per_query=config.youtube.max_videos_per_query,
                max_comments_per_video=config.youtube.max_comments_per_video,
            )
            minimized = [_private_minimized(post) for post in posts]
            all_posts.extend(minimized)
            retrieved_counts['youtube'][query] = len(minimized)
            logger.info('source=youtube query=%r retrieved=%d', query, len(minimized))

    raw_path = root / 'raw' / f'external_collection_{run_id}.jsonl'
    manifest_path = root / 'manifests' / f'collection_{run_id}.json'
    _write_jsonl(raw_path, all_posts)
    manifest = {
        'run_id': run_id,
        'timestamp': moment.isoformat().replace('+00:00', 'Z'),
        'status': 'collected_waiting_for_preparation',
        'sources': sorted(retrieved_counts),
        'queries': list(config.queries),
        'requested_counts': requested_counts,
        'retrieved_counts': retrieved_counts,
        'retrieved_total': len(all_posts),
        'prepared_count': 0,
        'deduplicated_count': 0,
        'sampling_seed': config.sampling_seed,
        'configuration_sha256': config.sha256,
        'output_paths': {
            'raw_jsonl': raw_path.relative_to(root.parent.parent).as_posix(),
            'manifest': manifest_path.relative_to(root.parent.parent).as_posix(),
        },
        'privacy': {
            'author_identity_collected': False,
            'private_content_collected': False,
        },
    }
    _write_json(manifest_path, manifest)
    return {'posts': all_posts, 'manifest': manifest,
            'raw_path': raw_path, 'manifest_path': manifest_path}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Collect bounded public content for external evaluation.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        config = load_config(args.config)
        result = collect_external_posts(config)
    except (ConnectorError, FileNotFoundError, TypeError, ValueError) as exc:
        logger.error('%s', exc)
        return 2
    print(f"Collected: {result['manifest']['retrieved_total']}")
    print(f"Raw data: {result['manifest']['output_paths']['raw_jsonl']}")
    print(f"Manifest: {result['manifest']['output_paths']['manifest']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

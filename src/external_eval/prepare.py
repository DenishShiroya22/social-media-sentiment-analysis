"""Filter, deduplicate, and reproducibly sample collected public posts."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from statistics import mean, median
from typing import Iterable, Sequence

from src.config import PROJECT_ROOT
from src.social.schemas import SocialPost
from src.storage.repository import create_run_id

from .config import DEFAULT_CONFIG_PATH, ExternalEvalConfig, load_config
from .sampling import stable_sample_id, stratified_sample

logger = logging.getLogger(__name__)
DEFAULT_DATA_ROOT = PROJECT_ROOT / 'data' / 'external_eval'
PLACEHOLDERS = frozenset({
    '[deleted]', '[removed]', 'deleted', 'removed',
    'this comment was deleted', 'this comment was removed',
})
URL_ONLY_PATTERN = re.compile(r'^(?:https?://\S+\s*)+$', re.IGNORECASE)


def load_raw_posts(path: Path | str) -> list[SocialPost]:
    posts: list[SocialPost] = []
    with Path(path).open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                posts.append(SocialPost.from_dict(value))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f'Invalid raw record on line {line_number}: {exc}') from exc
    return posts


def filter_content(posts: Iterable[SocialPost]) -> tuple[list[SocialPost], dict[str, int]]:
    accepted: list[SocialPost] = []
    counts: Counter[str] = Counter()
    for post in posts:
        text = post.text.strip()
        reason = None
        if not text:
            reason = 'blank'
        elif text.casefold() in PLACEHOLDERS:
            reason = 'placeholder'
        elif URL_ONLY_PATTERN.fullmatch(text):
            reason = 'url_only'
        if reason:
            counts[reason] += 1
        else:
            accepted.append(post)
    return accepted, dict(sorted(counts.items()))


def _text_key(text: str) -> str:
    return ' '.join(text.casefold().split())


def deduplicate_posts(
    posts: Iterable[SocialPost],
) -> tuple[list[SocialPost], dict[str, int]]:
    """Remove stable-ID duplicates, then exact whitespace-normalized text duplicates."""

    seen_ids: set[tuple[str, str]] = set()
    seen_texts: set[str] = set()
    unique: list[SocialPost] = []
    id_duplicates = 0
    text_duplicates = 0
    for post in posts:
        identifier = (post.source, post.post_id)
        if identifier in seen_ids:
            id_duplicates += 1
            continue
        seen_ids.add(identifier)
        text_key = _text_key(post.text)
        if text_key in seen_texts:
            text_duplicates += 1
            continue
        seen_texts.add(text_key)
        unique.append(post)
    return unique, {
        'id_duplicate_count': id_duplicates,
        'exact_text_duplicate_count': text_duplicates,
        'removed_duplicate_count': id_duplicates + text_duplicates,
    }


def language_review_flag(text: str) -> str:
    """Flag clearly non-Latin text for human review without silently dropping it."""

    letters = [character for character in text if character.isalpha()]
    if len(letters) < 5:
        return ''
    latin = sum(('a' <= character.casefold() <= 'z') for character in letters)
    return 'language_review' if latin / len(letters) < 0.7 else ''


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(
                row, ensure_ascii=False, sort_keys=True,
                separators=(',', ':')) + '\n')


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')


def _manifest_path(path: Path | str, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root.parent.parent.resolve()).as_posix()
    except ValueError:
        return resolved.name
def prepare_external_dataset(
    raw_path: Path | str,
    config: ExternalEvalConfig,
    *,
    output_root: Path | str = DEFAULT_DATA_ROOT,
    now: datetime | None = None,
) -> dict:
    """Prepare a diverse unlabeled sample without using model predictions."""

    root = Path(output_root)
    posts = load_raw_posts(raw_path)
    filtered, filter_counts = filter_content(posts)
    unique, duplicate_counts = deduplicate_posts(filtered)
    sampled = stratified_sample(
        unique,
        target_total=config.target_total,
        per_query_target=config.per_query_target,
        max_per_context=config.max_per_context,
        seed=config.sampling_seed,
    )
    rows: list[dict] = []
    for post in sampled:
        row = post.to_dict()
        row['author_id_or_name'] = None
        row['sample_id'] = stable_sample_id(post)
        row['review_flag'] = language_review_flag(post.text)
        rows.append(row)
    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    run_id = create_run_id(moment)
    prepared_path = root / 'prepared' / f'external_prepared_{run_id}.jsonl'
    manifest_path = root / 'manifests' / f'preparation_{run_id}.json'
    _write_jsonl(prepared_path, rows)
    lengths = [len(post.text) for post in sampled]
    manifest = {
        'run_id': run_id,
        'timestamp': moment.isoformat().replace('+00:00', 'Z'),
        'status': 'prepared_waiting_for_human_labeling',
        'input_file': _manifest_path(raw_path, root),
        'output_file': _manifest_path(prepared_path, root),
        'configuration_sha256': config.sha256,
        'sampling_seed': config.sampling_seed,
        'initial_count': len(posts),
        'content_filter_counts': filter_counts,
        **duplicate_counts,
        'deduplicated_count': len(unique),
        'prepared_count': len(sampled),
        'rows_per_source': dict(sorted(Counter(
            post.source for post in sampled).items())),
        'rows_per_query': dict(sorted(Counter(
            post.query for post in sampled).items())),
        'language_review_count': sum(
            bool(row['review_flag']) for row in rows),
        'average_text_length': mean(lengths) if lengths else 0.0,
        'median_text_length': median(lengths) if lengths else 0.0,
        'human_labels_present': False,
    }
    raw_file = Path(raw_path)
    prefix = 'external_collection_'
    collection_manifest_path = None
    if raw_file.stem.startswith(prefix):
        collection_run_id = raw_file.stem[len(prefix):]
        collection_manifest_path = (
            root / 'manifests' / f'collection_{collection_run_id}.json')
        manifest['source_collection_run_id'] = collection_run_id
    _write_json(manifest_path, manifest)
    if collection_manifest_path and collection_manifest_path.is_file():
        collection_manifest = json.loads(
            collection_manifest_path.read_text(encoding='utf-8'))
        collection_manifest.update({
            'status': 'prepared_waiting_for_human_labeling',
            'prepared_count': len(sampled),
            'deduplicated_count': len(unique),
            **duplicate_counts,
        })
        collection_manifest.setdefault('output_paths', {})[
            'prepared_jsonl'] = _manifest_path(prepared_path, root)
        collection_manifest['output_paths'][
            'preparation_manifest'] = _manifest_path(manifest_path, root)
        collection_manifest_path.write_text(
            json.dumps(
                collection_manifest, ensure_ascii=False,
                indent=2, sort_keys=True) + '\n',
            encoding='utf-8')
    return {
        'rows': rows, 'manifest': manifest,
        'prepared_path': prepared_path, 'manifest_path': manifest_path,
    }


def latest_raw_file(root: Path = DEFAULT_DATA_ROOT) -> Path:
    paths = sorted((root / 'raw').glob('external_collection_*.jsonl'))
    if not paths:
        raise FileNotFoundError('No external collection file found; run collect first')
    return paths[-1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Filter, deduplicate, and sample external public content.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--input', type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        config = load_config(args.config)
        raw_path = args.input or latest_raw_file()
        result = prepare_external_dataset(raw_path, config)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        logger.error('%s', exc)
        return 2
    print(f"Prepared: {result['manifest']['prepared_count']}")
    print(f"Removed duplicates: {result['manifest']['removed_duplicate_count']}")
    print(f"Output: {result['prepared_path']}")
    print('Next: export a blank labeling CSV and obtain human labels.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

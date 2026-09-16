"""Deterministic source/query-balanced sampling for external evaluation."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
import random
from typing import Iterable

from src.social.schemas import SocialPost


def stable_sample_id(post: SocialPost) -> str:
    payload = f'{post.source}\0{post.post_id}'.encode('utf-8')
    return hashlib.sha256(payload).hexdigest()[:20]


def context_key(post: SocialPost) -> str:
    metadata = post.metadata
    context = (
        metadata.get('video_id') or metadata.get('submission_id') or
        post.parent_id or post.post_id)
    return f'{post.source}:{context}'


def stratified_sample(
    posts: Iterable[SocialPost],
    *,
    target_total: int,
    per_query_target: int,
    max_per_context: int,
    seed: int = 42,
) -> list[SocialPost]:
    """Round-robin source/query strata with a cap on any thread or video."""

    if min(target_total, per_query_target, max_per_context) < 1:
        raise ValueError('sampling limits must be positive')
    strata: dict[tuple[str, str], list[SocialPost]] = defaultdict(list)
    for post in posts:
        strata[(post.source, post.query)].append(post)
    queues: dict[tuple[str, str], deque[SocialPost]] = {}
    for key, values in sorted(strata.items()):
        ordered = sorted(values, key=lambda post: stable_sample_id(post))
        random.Random(f'{seed}:{key[0]}:{key[1]}').shuffle(ordered)
        queues[key] = deque(ordered)
    selected: list[SocialPost] = []
    query_counts: Counter[str] = Counter()
    context_counts: Counter[str] = Counter()
    while len(selected) < target_total:
        progressed = False
        for key in sorted(queues):
            if len(selected) >= target_total:
                break
            queue = queues[key]
            if query_counts[key[1]] >= per_query_target:
                continue
            while queue:
                candidate = queue.popleft()
                candidate_context = context_key(candidate)
                if context_counts[candidate_context] >= max_per_context:
                    continue
                selected.append(candidate)
                query_counts[candidate.query] += 1
                context_counts[candidate_context] += 1
                progressed = True
                break
        if not progressed:
            break
    return selected

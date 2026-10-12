"""Interpretable TF-IDF terms and configurable product-theme summaries."""
from __future__ import annotations

from collections import Counter
import re
from typing import Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import LABELS


DEFAULT_THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "battery": ("battery", "charging", "charger", "charge"),
    "pricing": ("price", "pricing", "expensive", "cost", "cheap", "value"),
    "delivery": ("delivery", "delivered", "shipping", "late", "courier"),
    "customer support": ("support", "service", "customer care", "helpdesk"),
    "quality": ("quality", "durable", "broken", "defect", "fake"),
    "bugs": ("bug", "bugs", "crash", "error", "issue", "glitch"),
    "refunds": ("refund", "return", "replacement", "money back"),
    "performance": ("performance", "slow", "fast", "lag", "speed", "heat"),
    "design": ("design", "look", "style", "color", "display", "screen"),
    "availability": ("available", "availability", "stock", "sold out", "launch"),
}


def _records_for_sentiment(
    records: Sequence[Mapping], sentiment: str
) -> list[Mapping]:
    return [
        record
        for record in records
        if str(record.get("sentiment", "")).strip().lower() == sentiment
        and str(record.get("text", "")).strip()
    ]


def extract_terms(
    records: Sequence[Mapping],
    *,
    sentiment: str,
    top_n: int = 12,
    ngram_range: tuple[int, int] = (1, 2),
) -> list[dict]:
    """Return top mean TF-IDF terms for one sentiment subset."""

    if sentiment not in LABELS:
        raise ValueError(f"sentiment must be one of {LABELS}")
    if top_n < 1:
        raise ValueError("top_n must be positive")
    selected = _records_for_sentiment(records, sentiment)
    if not selected:
        return []
    texts = [str(record["text"]) for record in selected]
    min_df = 2 if len(texts) >= 8 else 1
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=0.95 if len(texts) >= 20 else 1.0,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9']+\b",
        max_features=3000,
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return []
    terms = vectorizer.get_feature_names_out()
    scores = np.asarray(matrix.mean(axis=0)).ravel()
    document_frequency = np.asarray((matrix > 0).sum(axis=0)).ravel()
    order = np.argsort(scores)[::-1][:top_n]
    return [
        {
            "term": str(terms[index]),
            "tfidf_score": float(scores[index]),
            "document_frequency": int(document_frequency[index]),
        }
        for index in order
        if scores[index] > 0
    ]


def theme_summary(
    records: Sequence[Mapping],
    *,
    themes: Mapping[str, Sequence[str]] | None = None,
) -> list[dict]:
    """Count configurable keyword themes and show sentiment within each theme."""

    configured = themes or DEFAULT_THEME_KEYWORDS
    compiled = {
        name: [re.compile(rf"\b{re.escape(keyword.casefold())}\b") for keyword in words]
        for name, words in configured.items()
    }
    totals: Counter = Counter()
    sentiments: dict[str, Counter] = {name: Counter() for name in configured}
    for record in records:
        text = str(record.get("text", "")).casefold()
        label = str(record.get("sentiment", "")).strip().lower()
        if label not in LABELS:
            raise ValueError(f"Unknown sentiment label: {label or '<blank>'}")
        for name, patterns in compiled.items():
            if any(pattern.search(text) for pattern in patterns):
                totals[name] += 1
                sentiments[name][label] += 1

    rows = []
    for name, total in totals.most_common():
        counts = {label: sentiments[name][label] for label in LABELS}
        rows.append({
            "topic": name,
            "frequency": total,
            "sentiment_counts": counts,
            "sentiment_percentages": {
                label: counts[label] / total * 100 for label in LABELS
            },
        })
    return rows


def topic_report(records: Sequence[Mapping], *, top_n: int = 12) -> dict:
    """Build aggregate negative terms, positive terms, and theme summaries."""

    return {
        "negative_terms": extract_terms(
            records, sentiment="negative", top_n=top_n
        ),
        "positive_terms": extract_terms(
            records, sentiment="positive", top_n=top_n
        ),
        "themes": theme_summary(records),
        "method": (
            "Terms use mean TF-IDF over predicted sentiment subsets. Themes use "
            "configurable keyword matching and are descriptive, not definitive."
        ),
    }

"""Local sentiment monitoring dashboard over the frozen production model."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from app.dashboard_helpers import (
    analyze_rows,
    collect_live_youtube,
    dashboard_payload,
    discover_local_datasets,
    load_local_rows,
)
from src.config import PROJECT_ROOT
from src.social.base import ConnectorError


st.set_page_config(page_title="Brand sentiment monitor", layout="wide")
st.title("Brand sentiment monitor")
st.caption(
    "Analyze local comments or perform one bounded YouTube collection with the "
    "frozen TweetEval-trained Logistic Regression model."
)

source_mode = st.sidebar.radio(
    "Data source", ["Local dataset", "Live YouTube collection"]
)
trend_frequency = st.sidebar.selectbox("Trend period", ["daily", "weekly"])
rows = None

if source_mode == "Local dataset":
    datasets = discover_local_datasets(PROJECT_ROOT / "data")
    if not datasets:
        st.warning("No supported local CSV or JSONL dataset was found.")
    else:
        selected = st.selectbox(
            "Local dataset",
            datasets,
            format_func=lambda path: str(path.relative_to(PROJECT_ROOT)),
        )
        if st.button("Analyze local dataset", type="primary"):
            try:
                rows = load_local_rows(selected)
            except (OSError, TypeError, ValueError) as exc:
                st.error(f"Could not load the local dataset: {exc}")
else:
    st.info("Reddit live collection is unavailable. YouTube is the active source.")
    query = st.text_input("Brand or product query", value="Samsung Galaxy")
    first, second = st.columns(2)
    max_videos = first.number_input(
        "Videos", min_value=1, max_value=5, value=2, step=1
    )
    comments_per_video = second.number_input(
        "Comments per video", min_value=1, max_value=20, value=10, step=1
    )
    estimated_requests = 1 + int(max_videos)
    confirmed = st.checkbox(
        f"Use the YouTube API for up to {estimated_requests} bounded requests"
    )
    if st.button("Collect and analyze", type="primary", disabled=not confirmed):
        if not os.environ.get("YOUTUBE_API_KEY", "").strip():
            st.error(
                "YOUTUBE_API_KEY is missing from this process. The application "
                "does not load .env automatically."
            )
        else:
            try:
                rows = collect_live_youtube(
                    query,
                    max_videos=int(max_videos),
                    comments_per_video=int(comments_per_video),
                )
            except (ConnectorError, OSError, TypeError, ValueError) as exc:
                st.error(f"YouTube collection failed: {exc}")

if rows is not None:
    if not rows:
        st.warning("The selected source returned no comments.")
    else:
        try:
            records = analyze_rows(rows)
            payload = dashboard_payload(records, trend_frequency=trend_frequency)
        except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
            st.error(f"Sentiment analysis could not run: {exc}")
        else:
            summary = payload["summary"]
            st.header("Overview")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Comments", summary["total_posts"])
            col2.metric(
                "Negative", f"{summary['sentiment_percentages']['negative']:.1f}%"
            )
            col3.metric(
                "Neutral", f"{summary['sentiment_percentages']['neutral']:.1f}%"
            )
            col4.metric(
                "Positive", f"{summary['sentiment_percentages']['positive']:.1f}%"
            )
            confidence = summary["average_confidence"]
            st.caption(
                f"Average model confidence: {confidence:.3f}. "
                "Confidence is not a calibrated certainty estimate."
                if confidence is not None
                else "No confidence values are available."
            )

            st.header("Sentiment distribution")
            distribution = pd.DataFrame(
                {
                    "sentiment": list(summary["sentiment_counts"]),
                    "count": list(summary["sentiment_counts"].values()),
                }
            ).set_index("sentiment")
            st.bar_chart(distribution)

            st.subheader("Sentiment by query")
            query_rows = [
                {
                    "query": query_name,
                    **group["sentiment_percentages"],
                    "comments": group["total_posts"],
                }
                for query_name, group in payload["per_query"].items()
            ]
            st.dataframe(pd.DataFrame(query_rows), hide_index=True, use_container_width=True)

            st.header("Trends")
            trends = pd.DataFrame(payload["trends"])
            if trends.empty:
                st.info("No usable timestamps are available for trend analysis.")
            else:
                trend_chart = trends.set_index("period")[[
                    "negative_percentage",
                    "neutral_percentage",
                    "positive_percentage",
                ]]
                st.line_chart(trend_chart)
                st.bar_chart(trends.set_index("period")[["post_count"]])

            st.header("Topics and complaints")
            topic_data = payload["topics"]
            left, right = st.columns(2)
            left.subheader("Negative terms")
            left.dataframe(
                pd.DataFrame(topic_data["negative_terms"]),
                hide_index=True,
                use_container_width=True,
            )
            right.subheader("Positive terms")
            right.dataframe(
                pd.DataFrame(topic_data["positive_terms"]),
                hide_index=True,
                use_container_width=True,
            )
            themes = pd.DataFrame([
                {
                    "topic": row["topic"],
                    "frequency": row["frequency"],
                    **row["sentiment_percentages"],
                }
                for row in topic_data["themes"]
            ])
            if not themes.empty:
                st.subheader("Configured product themes")
                st.dataframe(themes, hide_index=True, use_container_width=True)
            st.caption(topic_data["method"])

            st.header("Representative comments")
            st.caption("Shown only in this local session; comments are not tracked in Git.")
            for section, comments in payload["representative_comments"].items():
                with st.expander(section.replace("_", " ").title()):
                    if not comments:
                        st.write("No matching comments.")
                    for comment in comments:
                        st.write(comment["text"])
                        st.caption(
                            f"{comment['query']} · confidence {comment['confidence']:.3f}"
                        )

            st.header("Model information")
            metrics_path = PROJECT_ROOT / "reports" / "external_eval" / "external_metrics.json"
            if metrics_path.is_file():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                st.write("Model: frozen TF-IDF Logistic Regression")
                st.write("Official TweetEval TEST macro-F1: 0.6199")
                st.write(
                    "External human-labeled macro-F1: "
                    f"{metrics['metrics']['macro_f1']:.4f}"
                )
                st.caption(metrics["distribution_caveat"])
            else:
                st.write("Model: frozen TF-IDF Logistic Regression")
                st.write("Official TweetEval TEST macro-F1: 0.6199")
                st.info("External-domain metrics are not available.")

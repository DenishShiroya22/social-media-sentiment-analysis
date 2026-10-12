# Application inference and analytics

This layer turns bounded public comments into normalized records, frozen-model sentiment predictions, aggregate analytics, topic summaries, and a local Streamlit dashboard. It does not retrain or tune the model or rerun the official benchmark.

## Architecture

    User query
        ↓
    SocialConnector
        ↓
    Normalized SocialPost records
        ↓
    data/collected/*.jsonl
        ↓
    SentimentPredictor
        ↓
    AnalyzedPost records
        ↓
    data/analyzed/*.jsonl
        ↓
    RunManifest + basic sentiment summary
        ↓
    aggregate analytics, topics, and local dashboard

The modules have intentionally small responsibilities:

- src.inference.SentimentPredictor validates and cleans raw application text, verifies and caches the final artifact, and performs batch prediction.
- src.social.base.SocialConnector is the interface for bounded public-data connectors.
- src.social.schemas defines platform-neutral collected and analyzed records.
- src.social.reddit.RedditConnector performs read-only public submission search and optional bounded top-level comment collection through PRAW.
- src.social.youtube.YouTubeConnector searches bounded public videos and retrieves bounded top-level comments through YouTube Data API v3.
- src.storage.repository.JsonlRepository writes exclusive run-specific JSONL and manifest files.
- src.app_pipeline connects collection, storage, inference, and the console summary.
- src.analytics provides aggregate summaries, confidence views, trends, and interpretable topic extraction.
- app.streamlit_app provides the local dashboard while app.dashboard_helpers keeps data loading and analysis testable offline.

## Final model contract

Application inference loads models/final/final_model.joblib, the immutable combined word-and-character TF-IDF plus balanced Logistic Regression pipeline. The artifact contains feature extraction and classification, but it does not contain raw-text preprocessing. It was trained on clean_text, so SentimentPredictor calls the existing src.preprocessing.clean_text exactly once before sending a batch of cleaned strings to the fitted pipeline.

The loader checks the artifact SHA-256 recorded in models/final/final_model_metadata.json, verifies that the current preprocessing configuration matches the frozen metadata, verifies the pipeline steps and class order, and retains the loaded model for later calls. It never fits or retrains a model.

predict_one(text) returns a SentimentPrediction. predict_batch(texts) preserves order and calls predict and predict_proba on whole batches. Each result contains the original text, cleaned text, negative/neutral/positive label, maximum predicted probability as confidence, all three predicted probabilities, model version, artifact SHA-256, and UTC inference timestamp. Empty, whitespace-only, non-string, None, preprocessing-empty, and over-100,000-character inputs are rejected. An empty batch returns an empty list.

Example:

    python -m src.inference --text "I love the battery life"

Predicted probabilities are model outputs, not guaranteed calibrated probabilities.

## Social-post schema

SocialPost has these fields:

- source, post_id, timezone-aware created_at, text, and query
- optional url, author_id_or_name, and parent_id
- numeric engagement
- non-sensitive platform metadata

AnalyzedPost adds clean_text, sentiment, confidence, and the three flat probability fields. Both schemas serialize datetimes as UTC ISO 8601 strings and round-trip through JSON.

Author collection is disabled by default. The prototype does not request or store email addresses, phone numbers, private messages, hidden profile data, or private content. The optional author field exists for future requirements that justify it; use the CLI flag only after considering whether identity is actually needed.

## Reddit connector

The connector uses PRAW 8.0.3 in read-only mode and searches public submissions from r/all. Register an appropriate Reddit application and copy .env.example as a local reference or set these variables through your environment:

    REDDIT_CLIENT_ID=
    REDDIT_CLIENT_SECRET=
    REDDIT_USER_AGENT=

The repository does not load .env automatically and never logs these values. Configure the process environment with a descriptive user agent before running.

Supported arguments:

- limit: 50 by default, 1–500
- sort: relevance, new, or top
- time_filter: day, week, month, year, or all

Each submission becomes one generic record. External collection may also request a bounded number of top-level comments per submission. Submission text combines the title and non-deleted self-text. Engagement contains score and comment count; metadata contains subreddit and record type. Downstream code receives no PRAW objects. Authentication failures, rate limiting, and remote request failures raise application-specific connector errors. Individual malformed records are logged and skipped. An empty search returns an empty list and still produces an auditable empty run.

The connector never streams indefinitely or recursively crawls massive threads. External collection defaults to at most 20 submissions and 10 top-level comments per submission for each query, with both limits configurable and bounded. It does not bypass API controls or perform mass collection. Follow Reddit’s current developer terms and public-content policies for the intended use.

## Application CLI

With credentials in the process environment:

    python -m src.app_pipeline --source reddit --query "Samsung Galaxy S26" --limit 50 --sort relevance --time-filter month

Public author names remain omitted. --include-authors is an explicit privacy-sensitive opt-in.

A successful run:

1. queries the connector;
2. writes normalized collected posts;
3. batch-cleans and predicts them;
4. writes analyzed records;
5. writes a run manifest;
6. prints counts, percentages, average model confidence, and output paths.

## Storage and manifests

Real application data is local and Git-ignored:

    data/collected/<source>_<query>_<run-id>.jsonl
    data/analyzed/<source>_<query>_<run-id>.jsonl
    data/manifests/<run-id>.json

Files use UTF-8 and exclusive creation, so an existing run is never overwritten. A run ID combines a UTC timestamp and random suffix. The manifest records the source, query, retrieved and analyzed counts, creation time, model SHA-256 and version, connector version, and both output paths. Model metadata lives at run level instead of being duplicated in every analyzed record.

## Error handling and logging

Invalid local inputs raise TypeError or ValueError. Missing or altered model artifacts fail before inference. Missing Reddit credentials raise ConnectorConfigurationError naming the absent environment variables, without exposing values. Authentication, rate limits, and network/server failures use distinct connector exceptions. The CLI logs concise failures and exits with status 2; successful summaries use normal console output.

## Model choice and limitations

The application intentionally retains the official Logistic Regression model. Post-benchmark stacking reached VALIDATION macro-F1 0.6781 versus 0.6728 for LR, a difference of about +0.0052, but the paired bootstrap interval [-0.0109, +0.0211] crossed zero. XGBoost was substantially weaker. Neither replaces the completed official model.

The model learned from historical English TweetEval posts. Modern brand and product comments can differ in vocabulary, platform behavior, topic mix, and class balance. Sarcasm, conversational context, ambiguous language, code-switching, and new products remain difficult. Confidence values are not reliability guarantees. Use predictions as aggregate decision support, not psychological inference, individual profiling, or a replacement for human moderation.

## External collection and evaluation

The external-domain workflow is separate from routine application inference:

    public APIs → normalized private local records → exact deduplication
        → seeded source/query-balanced sampling → blank labeling CSV
        → human labeling → strict validation → frozen-model evaluation

The tracked `config/external_eval_queries.json` now uses six YouTube queries, a 400-row target, seed 42, and a 20-row per-video cap. Reddit remains supported in code but is disabled because live Reddit access is unavailable. YouTube requires `YOUTUBE_API_KEY` in the process environment. Both connectors omit authors in external storage. Raw, prepared, and label files remain under Git-ignored `data/external_eval/` paths. Use the project `.venv` interpreter on Windows; `.env` is ignored but is not loaded automatically by the application.

The completed YouTube-only collection retrieved 478 rows and prepared 400 after removing three exact-text duplicates. Human review excluded 69 language-ineligible rows, leaving 331 validated labels: 56 negative, 184 neutral, and 91 positive. Frozen-model evaluation achieved 0.5982 accuracy and 0.5406 macro-F1. The result measures transfer to this sampled YouTube brand domain and does not change the official model.

Run:

    .\.venv\Scripts\python.exe -m src.external_eval.collect --config config/external_eval_queries.json
    .\.venv\Scripts\python.exe -m src.external_eval.prepare --config config/external_eval_queries.json
    .\.venv\Scripts\python.exe -m src.external_eval.labeling export

At this point, stop for human labeling under `docs/EXTERNAL_LABELING_GUIDE.md`. The evaluator rejects blank or invalid labels before model loading:

    .\.venv\Scripts\python.exe -m src.external_eval.labeling validate --file data/external_eval/labels/external_eval_labeled.csv --expected-min 300 --expected-max 500
    .\.venv\Scripts\python.exe -m src.external_eval.evaluate --labels data/external_eval/labels/external_eval_labeled.csv

Evaluation uses `SentimentPredictor`, makes no fit/tuning/threshold changes, and saves aggregate metrics, a negative/neutral/positive confusion matrix, a dataset summary, and a domain-gap comparison under `reports/external_eval/`. Raw text and row-level predictions are not placed in public reports. Read `docs/DATA_COLLECTION_POLICY.md` for the API-first and privacy rules and `docs/EXTERNAL_EVALUATION.md` for the full protocol.

## Analytics and dashboard

`src/analytics/` provides sentiment totals, percentages, positive/negative ratio, confidence summaries, per-query results, daily and weekly trends, TF-IDF terms, configurable topic themes, and local representative-comment selection. Generate aggregate tracked reports with:

    .\.venv\Scripts\python.exe -m src.analytics.report --input data/external_eval/labels/external_eval_labeled.csv

Start the local dashboard with:

    .\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py

The dashboard analyzes a previously collected local CSV/JSONL dataset or performs a small YouTube collection after the user checks the API-use confirmation. Live collection is capped at five videos and twenty comments per video. It reports missing credentials, quota or request failures, empty results, malformed files, and missing model artifacts without exposing secrets. Representative comments stay in the local session. See `DASHBOARD.md`.

The official TweetEval TEST evaluation remains complete and must not be rerun. External results are measurement evidence, not a trigger for model changes.

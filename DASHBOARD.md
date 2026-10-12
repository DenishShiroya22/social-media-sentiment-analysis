# Local sentiment dashboard

The Streamlit dashboard analyzes private local comments with the frozen production model. It can also perform one small, explicitly confirmed YouTube collection. It does not retrain, tune, recalibrate, or replace the model.

## Start

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

The command starts a local Streamlit server and opens the interface in a browser. No hosted deployment is configured.

## Local dataset mode

The dashboard discovers supported private files under:

- `data/external_eval/labels/external_eval_labeled.csv`
- `data/external_eval/prepared/external_prepared_*.jsonl`
- `data/collected/*.jsonl`
- `data/analyzed/*.jsonl`

Select a file and choose **Analyze local dataset**. The dashboard validates that rows contain text, removes author fields from its in-memory view, runs one batched prediction, and displays aggregate results. Malformed, empty, or unsupported files produce a readable error.

## Live YouTube mode

Set `YOUTUBE_API_KEY` in the process environment before starting Streamlit. The application does not load `.env` automatically and never displays the key.

Choose a query, one to five videos, and one to twenty comments per video. The interface shows the bounded request estimate and keeps the collection button disabled until the API-use checkbox is selected. A run uses one search request plus at most one comment request per selected video. Reddit remains unavailable for live use.

## Sections

- **Overview:** comment count, sentiment percentages, and average confidence.
- **Sentiment distribution:** counts and per-query percentages.
- **Trends:** daily or weekly sentiment percentages and comment volume.
- **Topics and complaints:** mean TF-IDF terms plus configurable keyword themes such as battery, pricing, delivery, support, quality, bugs, refunds, performance, design, and availability.
- **Representative comments:** high-confidence and recent positive/negative examples shown only in the local session.
- **Model information:** official TweetEval performance, external human-labeled performance, and domain limitations.

Topic terms and theme matches are descriptive summaries. They can miss context or combine different meanings. Model confidence is the maximum class score and is not guaranteed to be calibrated.

## Privacy and storage

Real raw, prepared, labeled, collected, and analyzed rows stay in Git-ignored directories. The dashboard does not persist representative comments or API credentials. Tracked analytics reports contain aggregates only. Author names are removed from dashboard inputs and are not collected by the YouTube workflow.

## Troubleshooting

- **Missing API key:** set `YOUTUBE_API_KEY` in the same PowerShell session used to start Streamlit.
- **Quota exhausted or request failed:** wait for quota recovery or use a local dataset.
- **No comments:** try a different query or fewer videos with public comments enabled.
- **Model missing or changed:** restore the frozen model and metadata from the repository.
- **Malformed local file:** use CSV or JSONL with a nonempty `text` field on every row.

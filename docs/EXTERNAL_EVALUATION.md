# External brand-domain evaluation workflow

This workflow measures how the frozen TweetEval-trained model transfers to current public brand/product discussions. It is separate from TweetEval TRAIN, VALIDATION, and TEST. It never retrains, tunes, recalibrates, or changes the official benchmark.

## Completed human-label checkpoint

The bounded YouTube run retrieved 478 rows across Samsung Galaxy, iPhone, Zomato, Swiggy, Nike, and Netflix. Preparation removed three exact-text duplicates and sampled 400 rows across 28 video contexts. Human review marked 69 rows as language-ineligible because the annotator could not understand them; these were preserved in a private ignored exclusion file and were not counted as neutral sentiment. The remaining 331 rows passed validation with 56 negative, 184 neutral, and 91 positive labels. Aggregate counts are recorded in `reports/external_eval/`; raw and row-level data remain ignored.

The full run required 6 `search.list` calls and at most 30 `commentThreads.list` calls, for 36 bounded requests without pagination. Current quota behavior is documented by the [YouTube quota calculator](https://developers.google.com/youtube/v3/determine_quota_cost).

## 1. Configure

Edit `config/external_eval_queries.json`. The tracked configuration targets 400 rows from six configurable YouTube queries with sampling seed 42 and caps each video context at 20 rows. Reddit remains supported but is disabled while live Reddit access is unavailable. Collection content can change as public feeds change; code, configuration, and sampling behavior are reproducible.

Set credentials only in the process environment. YouTube uses `YOUTUBE_API_KEY`. The application does not automatically load `.env`; in PowerShell set `$env:YOUTUBE_API_KEY="..."` for the current session. Keep the value outside Git. Use `.\.venv\Scripts\python.exe` so all declared dependencies, including `emoji`, come from the project environment.

## 2. Collect

```powershell
.\.venv\Scripts\python.exe -m src.external_eval.collect --config config/external_eval_queries.json
```

The API-first collector writes privacy-minimized JSONL under `data/external_eval/raw/` and a text-free run manifest under `data/external_eval/manifests/`. It makes bounded read-only calls and never fabricates records when a source fails.

## 3. Prepare

```powershell
.\.venv\Scripts\python.exe -m src.external_eval.prepare --config config/external_eval_queries.json
```

Preparation removes blank/deleted/removed/URL-only placeholders, deduplicates by source ID and exact normalized text, and records each count. It flags clearly non-Latin text for review rather than silently dropping it. Seeded round-robin sampling balances source/query strata and caps any one thread or video.

## 4. Export and label

```powershell
.\.venv\Scripts\python.exe -m src.external_eval.labeling export
```

This creates `data/external_eval/labels/external_eval_labeling.csv` with `sample_id`, `source`, `post_id`, `created_at`, `query`, `text`, blank `label`, `review_flag`, and `notes`. No username is included. Follow `docs/EXTERNAL_LABELING_GUIDE.md`, save the completed copy as `external_eval_labeled.csv`, and do not expose model output to annotators.

STOP until human labeling is complete.

## 5. Validate

```powershell
.\.venv\Scripts\python.exe -m src.external_eval.labeling validate --file data/external_eval/labels/external_eval_labeled.csv --expected-min 300 --expected-max 500
```

Validation requires complete labels from `negative`, `neutral`, and `positive`; nonempty text/source/query; unique sample IDs and platform IDs; and the requested row-count range.

## 6. Evaluate once

```powershell
.\.venv\Scripts\python.exe -m src.external_eval.evaluate --labels data/external_eval/labels/external_eval_labeled.csv
```

The command validates labels before loading the frozen model, uses the exact production preprocessing and inference path, predicts once, and writes only aggregate outputs under `reports/external_eval/`:

- `external_metrics.json`
- `external_summary.md`
- `external_confusion_matrix.png`
- `dataset_summary.json`

Metrics include accuracy, macro precision/recall/F1, weighted F1, per-class precision/recall/F1/support, and a confusion matrix ordered negative/neutral/positive with actual rows and predicted columns. The report compares accuracy, macro-F1, and per-class F1 with the historical TweetEval TEST metrics while stating that the distributions are not interchangeable.

Poor results should be reported as possible domain shift, slang, product context, sarcasm, mixed sentiment, or platform differences. Strong results are promising only for this sampled dataset. Neither outcome changes the official benchmark or triggers retraining in this workflow.

## 7. Result

The frozen model was evaluated once on 331 validated rows:

| Metric | External result |
|---|---:|
| Accuracy | 0.5982 |
| Macro precision | 0.5374 |
| Macro recall | 0.5462 |
| Macro-F1 | 0.5406 |
| Weighted F1 | 0.6041 |
| Negative F1 | 0.2951 |
| Neutral F1 | 0.6742 |
| Positive F1 | 0.6524 |

Compared with TweetEval TEST, external accuracy is 0.0246 lower and macro-F1 is 0.0793 lower. Negative F1 is 0.3455 lower, while neutral and positive F1 are 0.0553 and 0.0522 higher. The datasets are not directly comparable. The differences can reflect YouTube versus Twitter behavior, historical versus current language, brand-specific vocabulary, slang, sarcasm, missing video context, and the sample's query and class composition.

## 8. Analytics and dashboard

Generate aggregate analytics and descriptive topics with:

```powershell
.\.venv\Scripts\python.exe -m src.analytics.report --input data/external_eval/labels/external_eval_labeled.csv
```

Run the local dashboard with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

The tracked reports contain no row-level comment text. Representative comments appear only in the local dashboard session. Topic labels are descriptive and imperfect; confidence values are not calibrated certainty estimates.

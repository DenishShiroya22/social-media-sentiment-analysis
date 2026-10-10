# External brand-domain evaluation workflow

This workflow measures how the frozen TweetEval-trained model transfers to current public brand/product discussions. It is separate from TweetEval TRAIN, VALIDATION, and TEST. It never retrains, tunes, recalibrates, or changes the official benchmark.

## Current human-label checkpoint

The bounded YouTube run retrieved 478 rows across Samsung Galaxy, iPhone, Zomato, Swiggy, Nike, and Netflix. Preparation removed three exact-text duplicates and sampled 400 rows across 28 video contexts. Thirty-eight rows require language review. The local labeling CSV contains 400 blank labels, so external evaluation is not yet permitted. Aggregate counts are recorded in `reports/external_eval/collection_summary.json`; raw and row-level data remain ignored.

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

# Social Media Sentiment Analysis

This public project builds a reproducible sentiment-modeling foundation for future brand and product monitoring. It acquires the official Cardiff NLP TweetEval sentiment splits, validates and conservatively cleans them, compares nine classical TF-IDF baselines, and tunes the two leading linear pipelines using TRAIN-only cross-validation. Live collection, analytics and a dashboard remain future work.

## Official final benchmark result

The frozen combined word-and-character TF-IDF plus Logistic Regression procedure was committed and CI-passing before TEST was unlocked. It was then fitted once on TRAIN+VALIDATION (47,615 rows) and evaluated once on untouched TEST (12,284 rows). The primary **TEST macro-F1 is 0.6199**; accuracy is **0.6228**.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| negative | 0.5945 | 0.6944 | 0.6406 | 3,972 |
| neutral | 0.6710 | 0.5744 | 0.6189 | 5,937 |
| positive | 0.5782 | 0.6240 | 0.6002 | 2,375 |

Macro precision is 0.6146, macro recall 0.6309, and weighted F1 0.6223. The previous TRAIN-only candidate had VALIDATION macro-F1 0.6728; TEST macro-F1 is 0.0529 lower after the frozen configuration was refitted on TRAIN+VALIDATION. This is a descriptive difference, not a reason to retune. No SVM or alternate model received a TEST score.

The [final evaluation summary](reports/final_evaluation_summary.md), [structured metrics](reports/final_test_metrics.json), [confusion matrix](reports/figures/final_test_confusion_matrix.png), [manifest](reports/final_evaluation_manifest.json), and [model card](models/final/MODEL_CARD.md) document the official result. Historical [baseline](reports/model_comparison.csv), [preprocessing](reports/preprocessing_impact.md), [tuning](reports/tuning_summary.md), and [uncertainty](reports/model_uncertainty.md) evidence remain available. These are historical tweet benchmark results, not guaranteed performance for current brand data.

## Data and workflow

Source: [Cardiff NLP TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval), sentiment configuration, immutable revision recorded in [the acquisition manifest](data/raw/manifest.json). Citation: Francesco Barbieri, Jose Camacho-Collados, Luis Espinosa Anke and Leonardo Neves (2020), [TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification](https://aclanthology.org/2020.findings-emnlp.148/). The source card lists the dataset license as unknown; review upstream terms before redistribution. Raw and processed dataset CSVs are deliberately excluded from Git and can be regenerated.

Official split sizes: TRAIN 45,615; VALIDATION 2,000; TEST 12,284. Labels are negative, neutral and positive. No resplitting or oversampling occurs. The pipeline preserves original text, labels, order and split membership. It normalizes only known escaped punctuation such as literal backslash-u2019 and backslash-u002c, plus actual Unicode punctuation, while retaining contractions, emoji, hashtags and sentiment punctuation. See [data foundation](DATA_FOUNDATION.md).

The completed sequence is data foundation → preprocessing → classical modeling → TRAIN-only CV tuning → VALIDATION selection → frozen procedure commit and CI → TRAIN+VALIDATION refit → one TEST evaluation. Next application work is a reusable inference service, responsible public social-media ingestion, aggregation, trend analysis and a dashboard.

## Reproduce

Use Python 3.11 or newer. From the repository root:

    python -m venv .venv
    .\.venv\Scripts\python -m pip install -r requirements.txt
    .\.venv\Scripts\python -m src.data_acquisition
    .\.venv\Scripts\python -m src.pipeline
    .\.venv\Scripts\python -m src.model_pipeline
    .\.venv\Scripts\python -m src.tuning_pipeline
    .\.venv\Scripts\python -m pytest -q
    .\.venv\Scripts\python scripts\create_notebooks.py
    .\.venv\Scripts\python scripts\create_model_notebook.py
    .\.venv\Scripts\python scripts\create_final_notebook.py
    .\.venv\Scripts\python scripts\execute_notebooks.py
    .\.venv\Scripts\python -m src.final_evaluation

On Linux/macOS, use .venv/bin/python instead. The acquisition command reuses checksum-verified local files or downloads the pinned source revision. The test suite uses offline fixtures; [GitHub Actions](.github/workflows/tests.yml) runs it on pushes and pull requests without downloading TweetEval. requirements-lock.txt records the tested local package set, while requirements.txt specifies portable ranges.

The saved [frozen candidate metadata](models/candidates/frozen_candidate_metadata.json) records the development selection. The [final model metadata](models/final/final_model_metadata.json) records the 47,615-row refit and TEST result. Joblib artifacts contain vectorizer state and classifier coefficients, not training-row arrays; load them only from trusted sources. The local label-only prediction file is excluded from Git, as are raw and processed dataset CSVs. There is no software LICENSE yet; public visibility does not itself grant reuse rights.

## One-time final evaluation guard

The [pre-TEST procedure](FINAL_EVALUATION_PROTOCOL.md) was frozen in commit 7922527d9599567887de2a5836e84a4e7e89458e, pushed and CI-passing before the one official evaluation. A normal invocation of src.final_evaluation only points to the saved result. Its explicit unlock requires that pre-TEST SHA, a clean synchronized checkout and no prior result. The completed manifest now prevents accidental repeated scoring. The explicit unlock command is recorded for audit in the protocol document; it is not a routine reproduction step.

[GitHub Actions](.github/workflows/tests.yml) runs offline tests on pushes and pull requests. It never downloads TweetEval or recomputes official TEST metrics.

## Repository map and limits

- src/: acquisition, validation, cleaning, feature extraction, modeling, tuning and evaluation.
- data/raw and data/processed: ignored split CSVs plus tracked provenance manifests.
- notebooks/: executed data, preprocessing, model-development and read-only final-results notebooks.
- reports/: executed data quality, baseline, tuning and final evaluation results.
- models/candidates and models/final: development and final fitted artifacts with metadata and a model card.
- tests/: offline behavior and leakage-boundary checks.

Historical English tweets may differ sharply from present-day brand comments. Sarcasm, missing conversational context, annotation ambiguity, class imbalance and domain shift remain. Usernames are unnecessary for sentiment modeling; future collection should respect platform terms and avoid unnecessary personal data. Predictions are fallible and should not be treated as objective psychological facts.

## Experimental post-benchmark research

After the one official TEST evaluation, a separate research track compares frozen TRAIN-only Logistic Regression and LinearSVC configurations with hard voting, TRAIN-CV-selected calibrated soft voting, out-of-fold stacking, and sparse XGBoost. This track trains on TRAIN (45,615 rows), selects settings with TRAIN-only CV, and compares fixed candidates on VALIDATION (2,000 rows). It produces **no new TEST score** and does not replace the official benchmark model.

The [experimental comparison](reports/experimental/ensemble_summary.md), [machine-readable metrics](reports/experimental/ensemble_metrics.json), [XGBoost metrics](reports/experimental/xgboost_metrics.json), and [read-only results notebook](notebooks/05_ensemble_experiments.ipynb) document the methods and development evidence. New artifacts, if a candidate shows a meaningful, stable improvement over frozen LR on VALIDATION, live only in models/experimental. Their status is EXPERIMENTAL DEVELOPMENT MODEL. The official TweetEval TEST result above remains the sole official benchmark.

With processed TRAIN and VALIDATION CSVs already available, reproduce this development work separately:

    .\.venv\Scripts\python -m src.experimental.ensemble
    .\.venv\Scripts\python -m src.experimental.xgboost_experiment
    .\.venv\Scripts\python -m src.experimental.report
    .\.venv\Scripts\python scripts\create_experimental_notebook.py

The XGBoost search is intentionally CPU-intensive and is excluded from CI. The offline synthetic tests cover leakage boundaries without downloading data. Future assessment of an experimental model should use a newly collected, manually labeled brand-comment dataset.

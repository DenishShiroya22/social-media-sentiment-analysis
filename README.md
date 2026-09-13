# Social Media Sentiment Analysis

This public project builds a reproducible sentiment-modeling foundation for future brand and product monitoring. It acquires the official Cardiff NLP TweetEval sentiment splits, validates and conservatively cleans them, compares nine classical TF-IDF baselines, and tunes the two leading linear pipelines using TRAIN-only cross-validation. Live collection, analytics and a dashboard remain future work.

## Current development result

The **frozen development candidate** is combined word-and-character TF-IDF with Logistic Regression. On the official VALIDATION split it reached macro-F1 **0.6728** and accuracy **0.6925**; negative-class F1 was **0.5850**. Its configuration was selected after stratified CV within TRAIN. The LinearSVC finalist reached macro-F1 0.6726. Their paired-bootstrap difference is small and uncertain. **TEST HAS NOT BEEN EVALUATED.** No TEST predictions, predictive metrics, confusion matrices or error analysis were generated.

The corrected nine-baseline comparison is in [model comparison](reports/model_comparison.csv). The [preprocessing impact](reports/preprocessing_impact.md), [tuning summary](reports/tuning_summary.md), [uncertainty](reports/model_uncertainty.md), [errors](reports/error_analysis.md), and [confidence/margins](reports/model_confidence.md) provide the evidence behind selection. These are historical tweet-validation results, not a performance claim for current brand data.

## Data and workflow

Source: [Cardiff NLP TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval), sentiment configuration, immutable revision recorded in [the acquisition manifest](data/raw/manifest.json). Citation: Francesco Barbieri, Jose Camacho-Collados, Luis Espinosa Anke and Leonardo Neves (2020), [TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification](https://aclanthology.org/2020.findings-emnlp.148/). The source card lists the dataset license as unknown; review upstream terms before redistribution. Raw and processed dataset CSVs are deliberately excluded from Git and can be regenerated.

Official split sizes: TRAIN 45,615; VALIDATION 2,000; TEST 12,284. Labels are negative, neutral and positive. No resplitting or oversampling occurs. The pipeline preserves original text, labels, order and split membership. It normalizes only known escaped punctuation such as literal backslash-u2019 and backslash-u002c, plus actual Unicode punctuation, while retaining contractions, emoji, hashtags and sentiment punctuation. See [data foundation](DATA_FOUNDATION.md).

The sequence is data foundation → preprocessing → classical modeling → focused TRAIN CV tuning → VALIDATION checkpoint → frozen development candidate. A final methodology review and one locked TEST evaluation are future work; live social-media integration and analytics come after that.

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
    .\.venv\Scripts\python scripts\execute_notebooks.py

On Linux/macOS, use .venv/bin/python instead. The acquisition command reuses checksum-verified local files or downloads the pinned source revision. The test suite uses offline fixtures; [GitHub Actions](.github/workflows/tests.yml) runs it on pushes and pull requests without downloading TweetEval. requirements-lock.txt records the tested local package set, while requirements.txt specifies portable ranges.

The saved [frozen candidate metadata](models/candidates/frozen_candidate_metadata.json) records preprocessing, TF-IDF, weights, classifier, CV, versions, validation metrics and artifact SHA-256. The fitted joblib artifact contains vectorizer state and classifier coefficients, not training-row arrays. Load joblib only from trusted sources; regenerate it if library versions differ. There is no software LICENSE yet; public visibility does not itself grant reuse rights.

## Frozen final evaluation protocol

The [one-time final evaluation procedure](FINAL_EVALUATION_PROTOCOL.md) fits the frozen configuration on TRAIN+VALIDATION only after its code is committed, pushed and CI-passing. It requires an explicit unlock flag and pre-TEST commit SHA; a normal invocation does not evaluate TEST. No official TEST score is present in this pre-evaluation checkpoint.

## Repository map and limits

- src/: acquisition, validation, cleaning, feature extraction, modeling, tuning and evaluation.
- data/raw and data/processed: ignored split CSVs plus tracked provenance manifests.
- notebooks/: executed data, preprocessing and model-development notebooks.
- reports/: executed data quality, baseline, tuning, uncertainty and error results.
- models/candidates/: fitted development artifacts and metadata.
- tests/: offline behavior and leakage-boundary checks.

Historical English tweets may differ sharply from present-day brand comments. Sarcasm, missing conversational context, annotation ambiguity, class imbalance and domain shift remain. Usernames are unnecessary for sentiment modeling; future collection should respect platform terms and avoid unnecessary personal data. Predictions are fallible and should not be treated as objective psychological facts.

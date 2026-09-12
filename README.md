# Social Media Sentiment Analysis

## Problem and objective

Build a reproducible sentiment-analysis foundation for future brand/product monitoring. Public social-media language is noisy and context-dependent; a benchmark classifier helps establish measurable behavior before any live collection is attempted. This repository acquires Cardiff NLP TweetEval sentiment, validates the official splits, cleans text conservatively, and compares classical classifiers on the official validation set.

The intended product flow is Brand/Product → Public Social Media → Collection → Cleaning → Sentiment Model → Negative / Neutral / Positive → Analytics → Dashboard. Collection, analytics, dashboard and deployment are future work.

## Current status and result

Nine controlled full-TRAIN experiments have run with word and character TF-IDF. Combined word (1,2) + character (3,5) TF-IDF with Logistic Regression is the current VALIDATION-SELECTED CANDIDATE: macro-F1 0.6646, accuracy 0.7050. It is not final test performance. TEST remains locked; there are no test predictions or test metrics. Full comparison, per-class results, confusion matrices and limitations are in reports/model_summary.md and MODELING.md.

## Dataset and ethics

Dataset: [Cardiff NLP TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval), configuration sentiment. The recorded source revision and hashes are in data/raw/manifest.json. Official splits: TRAIN 45,615; VALIDATION 2,000; TEST 12,284. Labels: 0 negative, 1 neutral, 2 positive. No random resplit or class rebalancing.

Citation: Francesco Barbieri, Jose Camacho-Collados, Luis Espinosa Anke, Leonardo Neves (2020), [TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification](https://aclanthology.org/2020.findings-emnlp.148/), Findings of EMNLP. The upstream dataset card lists the license as unknown; review original terms before redistribution.

Future collection should use public content only and respect platform terms. Usernames are unnecessary for sentiment modeling. Predictions are fallible, especially with sarcasm and missing context; social-media users do not represent all customers. Labels are not objective psychological facts. Avoid storing unnecessary personally identifiable information.

## Repository layout

    src/                  acquisition, validation, preprocessing, TF-IDF, models, evaluation, pipelines
    tests/                offline unit and fixture integration tests
    notebooks/            data understanding, preprocessing, model development
    data/raw/             official split CSVs and provenance manifest
    data/processed/       clean split CSVs and processing manifest
    models/candidates/    saved validation-selected candidate and metadata
    reports/              executed quality and modeling results
    reports/figures/      TRAIN EDA and VALIDATION model plots
    scripts/              notebook generation and execution

Large datasets and Hugging Face caches are excluded from Git. The 4.4 MB candidate artifact is included with checksum metadata so the pushed repository contains an inspectable result; regenerate it from source if needed. Only load joblib artifacts from trusted repositories.

## Installation

Python 3.11 or newer. From the repository root on Windows PowerShell:

    python -m venv .venv
    .\.venv\Scripts\python -m pip install -r requirements.txt

For Linux/macOS, activate a virtual environment with source .venv/bin/activate and use python in the commands below. requirements-lock.txt records the tested Windows/Python 3.14 package set; requirements.txt is the portable dependency specification.

## Reproduce data and models

    .\.venv\Scripts\python -m src.data_acquisition
    .\.venv\Scripts\python -m src.pipeline
    .\.venv\Scripts\python -m src.model_pipeline
    .\.venv\Scripts\python -m pytest -q
    .\.venv\Scripts\python scripts\execute_notebooks.py

Acquisition reuses checksum-verified local files and otherwise downloads the configured immutable source revision. To refresh deliberately, use python -m src.data_acquisition --force --revision with a reviewed commit hash. If download fails, the command raises an error; no unrelated dataset is substituted.

The modeling command requires processed TRAIN, VALIDATION and TEST files. It loads TRAIN and VALIDATION labels, mechanically checks TEST existence/schema/split/count/checksum, fits TF-IDF and classifiers on TRAIN only, ranks by VALIDATION macro-F1, writes reports and plots, then saves and reload-verifies the candidate. It never computes TEST predictions. Re-running is deterministic apart from recorded wall-clock times.

## Data contract and modeling choices

Processed files data/processed/train_clean.csv, validation_clean.csv and test_clean.csv contain unchanged original text, clean_text, standardized sentiment and official split membership. Optional metadata is preserved. Cleaning lowercases, normalizes Unicode/HTML/whitespace, removes URLs and mentions, and retains hashtag words. Emoji, negation, contractions, punctuation and repeated letters remain. Stopword filtering, stemming and lemmatization are disabled.

Word TF-IDF uses a contraction-preserving token pattern; character TF-IDF can capture punctuation and noisy spelling. The compared algorithms are Logistic Regression, LinearSVC, MultinomialNB and a bounded Random Forest. Macro-F1 is primary because negative is the minority class. See MODELING.md for exact parameters and the controlled experiment matrix.

## Leakage policy and limitations

Vocabulary, IDF and classifier weights are fitted only on TRAIN. VALIDATION supports model comparison and error analysis. TEST is reserved for one final evaluation after procedure freeze; neither refitting on TRAIN+VALIDATION nor final TEST evaluation is implemented here. The dataset is historical English Twitter text, not brand-specific current customer data. The winner is a development candidate, and validation performance may overestimate future-domain performance. Exact duplicate checks do not detect paraphrases, and tokenization choices may still omit some Unicode emoji cues.

See [data-foundation technical record](PHASE1.md) and [modeling methodology](MODELING.md) for implementation details.

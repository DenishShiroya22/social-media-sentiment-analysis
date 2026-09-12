# Social Media Sentiment Analysis

## Problem Statement
Brands need to understand public reactions to their products, but social-media language is noisy, context-dependent and difficult to summarize reliably.

## Project Objective
Build a reproducible sentiment-analysis system for brand/product monitoring. This repository implements **Phase 1 only: data foundation, TRAIN EDA and conservative NLP preprocessing**.

## Business Motivation
Eventually help identify recurring customer concerns and changes in public sentiment. This benchmark establishes engineering foundations; it does not establish brand-specific accuracy or business outcomes.

## Final Planned System
Brand/Product → Public Social Media → Data Collection → Preprocessing → Sentiment Model → Positive / Neutral / Negative → Analytics → Dashboard

## Current Status
Phase 1 is implemented and exercised on all 59,899 official TweetEval sentiment rows. No features are fit and no models are trained. See [executed summary](reports/phase1_summary.md), [Phase 1 guide](PHASE1.md), and reports/ for machine-readable evidence.

## Dataset
- Maintainer: Cardiff NLP.
- Source: [cardiffnlp/tweet_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval), configuration `sentiment`.
- Labels: 0 = negative, 1 = neutral, 2 = positive.
- Official train = 45,615; validation = 2,000; test = 12,284. No random resplit.
- Acquisition records the immutable source revision, Hugging Face fingerprints and CSV SHA-256 hashes in `data/raw/manifest.json`.
- Citation: Francesco Barbieri, Jose Camacho-Collados, Luis Espinosa Anke, Leonardo Neves (2020). [TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification](https://aclanthology.org/2020.findings-emnlp.148/), Findings of EMNLP, pp. 1644–1650. The sentiment task derives from SemEval Twitter sentiment data; consult the upstream card and original task terms for reuse. The Hugging Face card lists its license as unknown, so public availability should not be interpreted as unrestricted redistribution permission.

## Repository Structure
```text
data/raw/             official CSVs and provenance manifest
data/processed/       clean split CSVs and processing manifest
notebooks/            EDA and preprocessing teaching notebooks
src/                  configuration, acquisition, I/O, validation, EDA, cleaning, pipeline
tests/                unit and isolated integration tests
scripts/              notebook generation and execution
reports/figures/      five TRAIN charts
reports/              executed metrics, validation and summary
```

## Installation
Use Python 3.11 or newer. From the project root on Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
On Linux/macOS activate with `source .venv/bin/activate`. Runtime versions from the executed run are recorded in `reports/phase1_metrics.json`; `requirements-lock.txt` records the tested Windows/Python 3.14 environment. Use requirements.txt for portable Python 3.11+ resolution; the lock may contain platform-specific dependencies.

## Dataset Acquisition
```powershell
python -m src.data_acquisition
```
No credentials or paid services required. Initial acquisition needs network access. Complete local files are reused after checksum and schema validation. To regenerate the exact recorded snapshot:
```powershell
python -m src.data_acquisition --force --revision b3a375baf0f409c77e6bc7aa35102b7b3534f8be
```
`--force` explicitly refreshes local copies; new acquisition defaults to the immutable snapshot in src/config.py. An explicit `--revision` allows a deliberate source update. Corrupted local files fail with an actionable error. Acquisition errors are raised clearly; no fallback dataset is silently substituted. Retry the same command after restoring access. No sample dataset was needed for this run. Tests use small explicitly synthetic fixtures in temporary directories only.

## Running Phase 1
```powershell
python -m src.pipeline
python scripts/execute_notebooks.py
```
The pipeline acquires/reuses raw files, validates all splits, performs TRAIN EDA, cleans all splits, validates and reloads output CSVs, and generates reports. It can run from any working directory if the project is on Python's import path; the documented commands run at the root. Notebooks find the root from either the root or notebooks directory. To rebuild unexecuted notebooks, use `python scripts/create_notebooks.py`.

## Running Tests
```powershell
python -m pytest -q
```
Tests are offline and isolated. The actual full-data smoke test is `python -m src.pipeline`, including CSV round-trip assertions; notebook execution also runs this pipeline.

## Processed Dataset Schema
Each `data/processed/{train,validation,test}_clean.csv` contains `text` (unchanged original), `sentiment` (negative/neutral/positive), `split` (official membership), and `clean_text`. Optional future metadata is preserved by reusable functions. No index column is written. CSV has UTF-8 encoding. Empty CSV fields are loaded as empty strings so literal `NA` and `null` text is preserved; validation separately detects blank fields.

## Preprocessing Decisions
Defaults are centralized in the frozen `PreprocessingConfig` in src/preprocessing.py. Lowercase, normalize NFC Unicode/apostrophes, decode HTML entities and strip tags, remove URLs and mentions, retain hashtag words, and collapse whitespace. Preserve Unicode emoji, negation, contractions, punctuation and repeated letters. No stopword removal, stemming or lemmatization. Optional mention tokens/preservation, emoji-to-text conversion, case preservation and repeat reduction are available. Raw text remains available for future train-only experiments. Future word tokenizers must explicitly accommodate sentiment cues; preservation here does not guarantee a default TF-IDF tokenizer will consume them.

## Data Leakage Policy
No concatenation for development, random resplitting, feature fitting, model training or test-label EDA occurs. TRAIN is the only input to exploratory charts/token analysis; validation receives descriptive summaries. All-split checks are mechanical validation and label-blind text overlap. Test labels are checked only for valid vocabulary/preservation, never for development decisions or performance. Keep official duplicates and class proportions. Phase 2 must fit features/models on TRAIN, select on VALIDATION and evaluate TEST only once the final procedure is fixed.

## Limitations
Historical English tweets differ from contemporary brand/product feedback. Sarcasm, context, slang, label ambiguity and sampling bias remain. Exact overlap checks do not detect near duplicates. Lowercasing removes capitalization intensity; Unicode emoji preservation is verified with illustrative fixtures because this TRAIN snapshot has no detected emojis. The pipeline deliberately fails on unusable clean text rather than silently removing official rows. Generic HTML/URL regex handling is conservative, not a full browser parser. A refresh with changed expected row counts requires investigation, not automatic acceptance.

## Ethics / Privacy
Future collection should use public content only, respect platform terms, and avoid unnecessary personally identifiable information. Usernames are not required for sentiment modeling. Predictions can be wrong, especially for sarcasm and missing context; labels are not objective psychological facts. NLP models may contain bias, and social-media users do not represent all customers. Raw and processed datasets and caches are excluded from Git; regenerate them locally. Small manifests, code, notebooks, reports and figures are retained. Notebook examples come only from TRAIN.

## Future Phases
1. Data acquisition, validation, EDA and preprocessing (implemented).
2. TF-IDF, baseline models and comparison.
3. Hyperparameter tuning and final classical model.
4. Real social-media integration.
5. Trends, topics/aspects and business insights.
6. Streamlit dashboard.
7. Deployment, final report and presentation.

# Data and NLP foundation

## Objective and scope
Prepare clean, reproducible official TweetEval sentiment splits for future classical sentiment modeling. Acquisition, integrity checks, descriptive analysis, conservative cleaning, notebooks, tests and reports are included. Model fitting, TF-IDF, tuning, APIs, collection integrations, dashboards and deployment are excluded.

## Architecture and responsibilities
Hugging Face → acquisition → normalized official raw splits → validation → TRAIN EDA → stateless preprocessing → validated processed splits → classical modeling.

| Module | Responsibility |
|---|---|
| src/config.py | Project-relative pathlib paths, split names, labels, expected sizes |
| src/data_acquisition.py | Resolve source commit, verify upstream schema/semantics, map labels, save CSVs, checksum reuse |
| src/data_loader.py | UTF-8 CSV I/O, useful errors, atomic per-file writes and no index |
| src/data_validation.py | Configurable label mapping, errors/warnings, dates, duplicates, label-blind overlap |
| src/preprocessing.py | Frozen configuration, modular cleaning, original text/metadata preservation |
| src/eda.py | TRAIN figures, artifact and term counts, guarded descriptive summaries |
| src/pipeline.py | Orchestration, output verification, manifests and generated reports |

## Dataset choice and source
TweetEval sentiment is a manageable three-class public English benchmark with established official splits. Source and citation are in README.md. Pin the recorded revision to reproduce identical source content; acquisition records hashes and fingerprints. This is a generic tweet sentiment foundation, not a dataset of labeled brand-specific aspects.

## Contract and split policy
Raw: `text`, `sentiment`, `split`. Processed: these columns plus `clean_text`. Preserve optional post_id/platform/created_at/brand/product/engagement/source_url metadata. Numeric conventions are explicit configurable mappings; TweetEval uses 0 negative / 1 neutral / 2 positive. No rows, original text, benchmark labels, order or split memberships are modified. TRAIN is for EDA and future fitting; VALIDATION for future tuning/selection; TEST for final evaluation only.

## Acquisition and validation
Verify exact official split set, text/label columns, ClassLabel names and expected counts before writing raw files. Complete manifests permit offline reuse only after hash/schema checks. Network failures raise a clear error with no silent substitution. Required columns, empty frames, missing/non-string/blank text, invalid or missing labels/splits and expected split mismatches are critical errors. Duplicate rows/text, <3-character or >500-character text and invalid optional dates are warnings. Thresholds are configurable. Raw and processed exact cross-split overlaps count distinct nonblank strings and use no labels. Full data-quality JSON is saved; rows are never removed automatically.

## EDA philosophy
All charts, examples, token/bigram counts and per-sentiment term analysis use TRAIN. Validation summary is reported without affecting rules. Test receives only mechanical contract/duplicate checks; no test distribution or examples are exposed. Conflicting requests for all-split class distributions are resolved in favor of test isolation. Five charts cover class balance, text lengths, social artifacts, top tokens and cleaning length changes. Artifacts count posts containing matches. Token counts include stopwords and are not modeling features.

## Preprocessing philosophy
Use identical deterministic defaults for all splits, based on general NLP principles: preserve negation, contractions, emoji, hashtag words, repeated letters and punctuation. Remove mentions/URLs, normalize whitespace/Unicode/HTML, lowercase. No stemming, lemmatization or stopword filtering. Optional variants are available but do not change generated primary outputs. Missing scalar text cleans to an empty string and fails downstream validation. Do not casually drop rows that become empty; review a failure before any policy change.

## Execution and outputs
```text
python -m pip install -r requirements.txt
python -m src.data_acquisition
python -m src.pipeline
python -m pytest -q
python scripts/execute_notebooks.py
```
See README.md for virtual environments and exact revision download commands. Outputs: three raw CSVs; raw provenance manifest; three processed CSVs; processed checksum/config/environment manifest; reports/data_quality.json; reports/phase1_metrics.json; reports/phase1_summary.md; five PNG figures; two executed notebooks; notebook execution record. Data CSVs/caches are excluded from Git for size/privacy; manifests and reproducible code remain. The generated summary and JSON provide actual counts, including changed/removed rows.

## Leakage prevention and modeling handoff
Consume `data/processed/train_clean.csv` and `validation_clean.csv`; reserve `test_clean.csv`. Use clean_text as the initial model input and sentiment as the target. Fit vocabulary/IDF and models on TRAIN only; apply fitted transforms to VALIDATION. Choose methods/settings using VALIDATION. Access TEST performance only after final selection. Do not concatenate splits or deduplicate across boundaries. Preserve original text for traceability. Modeling should consider negation-aware tokenization and explicitly include preserved emoji/punctuation if desired; no such learned or fitted transformation exists here.

## Known limitations
Historical English benchmark, generic sentiment rather than aspect labels, class imbalance, sarcasm/context difficulty, demographic/platform bias, imperfect annotation and exact-only duplicate checks. Lowercasing loses capitalization signals. The observed zero emoji prevalence limits empirical assessment of emoji handling on this snapshot. Network is needed for initial acquisition, not verified local reruns. Multi-file publication is not transactional, but the processed manifest is written last and records each output hash; rerun after an interrupted pipeline. See README ethics/privacy and source terms before redistribution or future collection.

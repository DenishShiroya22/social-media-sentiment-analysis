# Data foundation executed summary

Dataset: `cardiffnlp/tweet_eval`, configuration `sentiment`.
Source: https://huggingface.co/datasets/cardiffnlp/tweet_eval
Immutable revision: `b3a375baf0f409c77e6bc7aa35102b7b3534f8be`.

## Split integrity and outputs

| Split | Raw rows | Processed rows | Changed text | Removed |
|---|---:|---:|---:|---:|
| train | 45615 | 45615 | 45150 | 0 |
| validation | 2000 | 2000 | 1984 | 0 |
| test | 12284 | 12284 | not explored | 0 |

Raw columns: text, sentiment, split. Processed columns add clean_text.
All saved CSVs were reloaded and compared with in-memory data. Original text, order, labels and split membership are unchanged.

## Quality findings

Counts below are mechanical integrity checks, not exploratory test-label analysis. Duplicate counts exclude the first occurrence; overlap counts are distinct nonblank strings.

- raw/train: errors `{}`; warnings `{'duplicate_rows': 26, 'duplicate_text': 29}`; missing cells `0`; blank text `0`.
- raw/validation: errors `{}`; warnings `{}`; missing cells `0`; blank text `0`.
- processed/train: errors `{}`; warnings `{'duplicate_rows': 26, 'duplicate_text': 43}`; missing cells `0`; blank text `0`.
- processed/validation: errors `{}`; warnings `{}`; missing cells `0`; blank text `0`.

Exact raw text overlap: `{'train__validation': 0}`.
Cleaned text overlap: `{'train__validation': 0}`.
TEST receives only schema, row-count, split-membership and checksum checks. Official duplicates are preserved. No deduplication or rebalancing was performed.

## TRAIN findings

| Sentiment | Rows | Percent |
|---|---:|---:|
| negative | 7093 | 15.55 |
| neutral | 20673 | 45.321 |
| positive | 17849 | 39.13 |

Characters per training post: mean 106.93, median 113, range 10–200.
Words per training post: mean 19.24, median 20.
Artifact prevalence: `{'url': {'rows': 88, 'percent': 0.193}, 'mention': {'rows': 13390, 'percent': 29.354}, 'hashtag': {'rows': 8526, 'percent': 18.691}, 'emoji': {'rows': 0, 'percent': 0.0}}`.
Top training tokens (stopwords included): `[('the', 38002), ('to', 20952), ('user', 16840), ('in', 13710), ('i', 13612), ('on', 13119), ('a', 13006), ('and', 12699), ('of', 10896), ('for', 9992)]`.
Top training bigrams: `[('in the', 3589), ('user user', 2678), ('of the', 2566), ('on the', 2365), ('for the', 2350), ('going to', 2223), ('at the', 1974), ('to the', 1840), ('may be', 1497), ('will be', 1283)]`.
Suspiciously short TRAIN examples (<3 characters): `[]`.
Suspiciously long TRAIN examples (>500 characters): `[]`.
Class imbalance motivates macro-F1 and per-class precision/recall alongside accuracy. Frequent tokens are descriptive counts, not fitted modeling features.
See phase1_metrics.json for validation summary, full training statistics and tokens by sentiment; figures/ contains five training charts.

## Fixed preprocessing decisions

NFC Unicode and curly-apostrophe normalization; HTML tags removed and entities decoded; lowercase; URLs and mentions removed; hashtag words retained. Unicode emojis, contractions, negation, punctuation and repeated characters retained. Whitespace collapsed. No stopword removal, stemming or lemmatization. Optional mention tokens, demojizing and repeat reduction are available but unused in primary outputs.
No rows removed. Missing or cleaning-empty text is reported as a critical error and blocks publication instead of silently filtering benchmark rows.

## Risks and modeling handoff

Historical English tweets are not representative of all customers or contemporary brand discourse. Sarcasm, context, annotation ambiguity and platform/demographic bias remain. Exact duplicate auditing does not detect paraphrases. Lowercasing loses capitalization intensity, but raw text is retained.
Default word-based TF-IDF tokenization may discard emoji and punctuation and fragment contractions. Fit future feature extractors on TRAIN only and validate choices on VALIDATION.
Consume data/processed/{train,validation,test}_clean.csv with text, clean_text, sentiment, split. Fit features and models only on TRAIN; tune with TRAIN cross-validation; use VALIDATION for development confirmation and TEST only for final evaluation. Test class distributions/examples are intentionally absent.

## Reproducibility

Run `python -m src.pipeline`. Raw and processed manifests record SHA-256 checksums, source revision, configuration and runtime versions. Run `python -m pytest -q` and `python scripts/execute_notebooks.py` for verification.

# Classical sentiment modeling

## Contract and leakage boundary

Official TweetEval sentiment TRAIN has 45,615 posts: negative 7,093, neutral 20,673, positive 17,849. VALIDATION has 2,000. TEST has 12,284 and is locked. Its only permitted checks are existence, schema, count, split membership and checksum. No TEST text or labels enter model development. Vocabulary, IDF and classifier fitting occur on TRAIN; CV refits the full sklearn Pipeline inside each TRAIN fold. Each tuned finalist is evaluated once on VALIDATION after its configuration is fixed by TRAIN CV. Earlier fixed baselines were also compared on VALIDATION.

Macro-F1 is the primary metric because the negative class is less common. Accuracy, macro precision/recall, weighted F1 and per-class precision/recall/F1 are also recorded. Random seed: 42.

## Corrected preprocessing and baselines

Literal escaped punctuation affected 5,111 TRAIN rows and 206 VALIDATION rows. Explicit mappings fix escaped apostrophes, quotation marks and commas without blanket Unicode decoding. Corrected learned features contain zero u2019/u002c artifacts; don't, can't wait and isn't appear in the TRAIN-fitted vocabulary. The full audit and old/new deltas are in [preprocessing impact](reports/preprocessing_impact.md).

| Experiment | Features | Classifier | Accuracy | Macro-F1 | Negative F1 |
|---|---|---|---:|---:|---:|
| exp04 | combined | logistic_regression | 0.7000 | 0.6580 | 0.5130 |
| exp07 | combined | linear_svm | 0.6830 | 0.6536 | 0.5451 |
| exp03 | character | logistic_regression | 0.6900 | 0.6463 | 0.4972 |
| exp06 | character | linear_svm | 0.6705 | 0.6382 | 0.5219 |
| exp02 | word_bigram | logistic_regression | 0.6810 | 0.6367 | 0.4921 |
| exp05 | word_bigram | linear_svm | 0.6680 | 0.6337 | 0.5115 |
| exp01 | word_unigram | logistic_regression | 0.6730 | 0.6230 | 0.4600 |
| exp08 | word_bigram | naive_bayes | 0.6260 | 0.4592 | 0.0190 |
| exp09 | word_bigram | random_forest | 0.5405 | 0.3651 | 0.0000 |

The corrected ranking retains combined word (1,2) plus character (3,5) TF-IDF Logistic Regression first and LinearSVC second. Character-only linear models follow. The bounded Random Forest and MultinomialNB are weak on negative F1 and were not tuned. These rankings compare specific configurations, not all possible algorithms.

## TRAIN-only focused search

Three-fold StratifiedKFold, shuffled with seed 42, scores f1_macro. Three folds and serial execution bound runtime and memory for sparse combined matrices. Each classifier searched C = 0.25, 0.5, 1, 2, 4 and class_weight = None or balanced: 10 settings per model. A second stage compared five feature settings near the best classifier: equal weights, character weight 0.75 or 1.25, word min_df 3, or character n-grams (3,6). This is 30 configurations and 90 fold fits total. There is no exhaustive Cartesian feature search.

| Model | Best TRAIN-CV configuration | Mean macro-F1 | CV std |
|---|---|---:|---:|
| logistic_regression | {'classifier__C': 1.0, 'classifier__class_weight': 'balanced', 'features__transformer_weights': {'character': 1.25, 'word': 1.0}} | 0.6545 | 0.0021 |
| linear_svm | {'classifier__C': 0.25, 'classifier__class_weight': 'balanced'} | 0.6517 | 0.0016 |

All candidate parameters, ranks, fold scores and runtimes are in [tuning results](reports/tuning_results.csv). TF-IDF is inside each CV Pipeline, preventing feature fitting on a held-out fold. No validation or TEST row enters CV.

## Finalist VALIDATION checkpoint

| Model | Accuracy | Macro precision | Macro recall | Macro-F1 | Negative F1 | Neutral F1 | Positive F1 | Fit s | Predict s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.6925 | 0.6642 | 0.6891 | 0.6728 | 0.5850 | 0.6861 | 0.7475 | 31.2 | 0.480 |
| linear_svm | 0.6975 | 0.6693 | 0.6771 | 0.6726 | 0.5710 | 0.6994 | 0.7473 | 16.9 | 0.508 |

Full per-class precision and recall, confusion matrices and version details are in [tuning metrics](reports/tuning_metrics.json).
LR minus SVM macro-F1 is +0.0003; a 2,000-draw paired row bootstrap gives a 95% percentile interval [-0.0117, +0.0128]. It crosses zero, so this validation sample does not distinguish the two reliably. It is a sensitivity diagnostic, not formal proof. See [uncertainty analysis](reports/model_uncertainty.md).

The frozen development candidate is **logistic_regression** by validation macro-F1, with CV stability, negative-class F1 and runtime considered as secondary evidence. Its validation macro-F1 is 0.6728; negative recall is 0.6731. The small LR advantage should not be interpreted as a firm generalization advantage. **TEST HAS NOT BEEN EVALUATED.**

## Error and confidence review

Both finalists have all six off-diagonal confusion counts and descriptive validation cue error rates in [error analysis](reports/error_analysis.md). LR gets more actual negative posts correct but also calls more neutral posts negative; SVM has slightly higher accuracy and neutral F1. Negation error rates are equal in this descriptive cue grouping; contraction and emoticon groups slightly favor SVM. Four posts have five words or fewer, too few for a meaningful short-text comparison.
The [confidence report](reports/model_confidence.md) uses LR maximum probability and SVM top-versus-second decision margin separately. Incorrect predictions have lower median values for each. Neither measure guarantees calibration, and some high-value predictions are wrong. No SVM probability calibration was added.

## Frozen artifact and reproducibility

Artifact: models/candidates/frozen_candidate.joblib (4,329,149 bytes; SHA-256 ef5e641c7368cacc2d17938fc2d05eaad86f45df538417c800c6a9afe1c8af0b). Reloaded VALIDATION predictions exactly match the pre-save predictions. [Machine-readable metadata](models/candidates/frozen_candidate_metadata.json) fixes preprocessing, word/character TF-IDF, feature weights, classifier, CV strategy, selection rule, versions and validation metrics.
Run python -m src.data_acquisition, python -m src.pipeline, python -m src.model_pipeline and python -m src.tuning_pipeline in that order from the repository root; then python -m pytest -q and python scripts/execute_notebooks.py. The tuning search is intentionally CPU intensive.

## Limits and next work

These are development results on historical English tweets. Brand-domain shift, sarcasm, absent context, ambiguous labels and class imbalance remain. Selection on a single validation split can be optimistic. Raw and processed dataset CSVs are excluded from Git; upstream rights need review before redistribution. Next: review the frozen methodology, decide whether to refit on TRAIN+VALIDATION, and perform exactly one locked TEST evaluation in a separate task. No TEST predictive result exists in this work.

## Frozen one-time final evaluation protocol

The pre-TEST [final evaluation protocol](FINAL_EVALUATION_PROTOCOL.md) is implemented in src/final_evaluation.py. It will fit the exact frozen Pipeline on TRAIN+VALIDATION (47,615 rows) and, only after this procedure is pushed and CI passes, evaluate untouched TEST once. The ordinary command is read-only; explicit unlock requires the pre-TEST commit SHA and a clean synchronized checkout. This checkpoint contains no official TEST metrics.

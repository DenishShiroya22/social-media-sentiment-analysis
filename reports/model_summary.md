# Classical sentiment modeling: executed VALIDATION summary

Objective: compare sparse TF-IDF representations and classical classifiers using official TweetEval TRAIN and VALIDATION.
TRAIN fits vocabulary, IDF and classifiers. VALIDATION selects by macro-F1. TEST remains locked: only existence, schema, row count, split and checksum were checked. No TEST metrics or predictions exist.

## Data and imbalance

TRAIN: 45615; VALIDATION: 2000; locked TEST rows: 12284.
TRAIN class counts: negative 7093; neutral 20673; positive 17849. No oversampling or resplitting.

## Features and models

TF-IDF multiplies term frequency within a document by inverse document frequency, reducing the influence of very common terms. In not good, word unigrams are not and good; the bigram not good retains their relation.
Word features capture vocabulary and word pairs, using a contraction-preserving token pattern. Character n-grams (3–5) can capture n't, !!!, repeated letters and fragments of noisy words. Combined features concatenate word (1,2) and character (3,5) sparse matrices. Character benefits are judged by VALIDATION results below.
All representations use min_df=2, max_df=.95, sublinear_tf=True, L2 normalization and no additional lowercasing. Word max_features=100,000; character max_features=120,000. No exhaustive search or class weighting.
Models: Logistic Regression (C=1, lbfgs, max_iter=350), LinearSVC (C=1, max_iter=3000), MultinomialNB (alpha=1), RandomForest (80 trees, depth 24, leaf minimum 2, sqrt features, 2 jobs). Random seed 42 where applicable.

## Validation comparison

| ID | Features | Model | Accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 | Fit seconds |
|---|---|---|---:|---:|---:|---:|---:|---:|
| exp04 | combined | logistic_regression | 0.7000 | 0.6820 | 0.6455 | 0.6580 | 0.6961 | 29.43 |
| exp07 | combined | linear_svm | 0.6830 | 0.6617 | 0.6475 | 0.6536 | 0.6819 | 8.12 |
| exp03 | character | logistic_regression | 0.6900 | 0.6736 | 0.6333 | 0.6463 | 0.6856 | 11.53 |
| exp06 | character | linear_svm | 0.6705 | 0.6506 | 0.6299 | 0.6382 | 0.6687 | 6.24 |
| exp02 | word_bigram | logistic_regression | 0.6810 | 0.6770 | 0.6207 | 0.6367 | 0.6750 | 4.90 |
| exp05 | word_bigram | linear_svm | 0.6680 | 0.6486 | 0.6244 | 0.6337 | 0.6658 | 1.28 |
| exp01 | word_unigram | logistic_regression | 0.6730 | 0.6619 | 0.6084 | 0.6230 | 0.6660 | 1.62 |
| exp08 | word_bigram | naive_bayes | 0.6260 | 0.7580 | 0.4955 | 0.4592 | 0.5760 | 0.09 |
| exp09 | word_bigram | random_forest | 0.5405 | 0.4392 | 0.4206 | 0.3651 | 0.4646 | 1.14 |

## Leading models

- logistic_regression: exp04 combined; macro-F1 0.6580; accuracy 0.7000; negative/neutral/positive F1 0.5130/0.7128/0.7481; fit 29.43s.
- linear_svm: exp07 combined; macro-F1 0.6536; accuracy 0.6830; negative/neutral/positive F1 0.5451/0.6838/0.7320; fit 8.12s.
- random_forest: exp09 word_bigram; macro-F1 0.3651; accuracy 0.5405; negative/neutral/positive F1 0.0000/0.6464/0.4488; fit 1.14s.
- naive_bayes: exp08 word_bigram; macro-F1 0.4592; accuracy 0.6260; negative/neutral/positive F1 0.0190/0.6670/0.6917; fit 0.09s.

## Selected candidate and errors

VALIDATION-SELECTED CANDIDATE: exp04 combined + logistic_regression. Macro-F1 0.6580; accuracy 0.7000. THIS IS NOT YET THE FINAL TEST-EVALUATED MODEL.
Winner VALIDATION confusion rows (negative, neutral, positive): [[138, 136, 38], [62, 665, 142], [26, 196, 597]]. Off-diagonal cells count misclassifications; see error_analysis.md for controlled examples and cue counts.
The top learned linear coefficients in model_metrics.json are TRAIN-fitted feature associations, not causal explanations. Positive coefficients favor the corresponding class relative to alternatives.
Random Forest used all TRAIN rows with bounded depth and 80 trees. See its negative recall and fit time above; the constrained configuration is not a full-capacity forest.

## Limits and next work

Historical English tweets, class imbalance, sarcasm, context loss and benchmark annotation ambiguity remain. Lowercased clean text loses case cues. Default word vectorization omits emoji, while character features may represent them indirectly. No brand-specific or prospective evaluation is claimed.
Next: focused validation-guided tuning of the strongest candidates and tokenizer checks, freeze the procedure, then perform one locked TEST evaluation. Do not call this artifact production-ready.
Full per-class precision/recall/F1, confusion matrices, feature/model parameters, versions and times are in model_metrics.json. Model artifact regeneration: python -m src.model_pipeline.

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
| exp04 | combined | logistic_regression | 0.7050 | 0.6895 | 0.6515 | 0.6646 | 0.7012 | 15.94 |
| exp07 | combined | linear_svm | 0.6905 | 0.6726 | 0.6560 | 0.6631 | 0.6893 | 6.84 |
| exp03 | character | logistic_regression | 0.6920 | 0.6835 | 0.6349 | 0.6500 | 0.6871 | 12.73 |
| exp06 | character | linear_svm | 0.6745 | 0.6530 | 0.6304 | 0.6393 | 0.6723 | 4.86 |
| exp02 | word_bigram | logistic_regression | 0.6805 | 0.6762 | 0.6203 | 0.6363 | 0.6745 | 4.23 |
| exp05 | word_bigram | linear_svm | 0.6700 | 0.6511 | 0.6268 | 0.6362 | 0.6677 | 1.66 |
| exp01 | word_unigram | logistic_regression | 0.6725 | 0.6631 | 0.6086 | 0.6236 | 0.6657 | 1.35 |
| exp08 | word_bigram | naive_bayes | 0.6255 | 0.7025 | 0.4964 | 0.4627 | 0.5771 | 0.09 |
| exp09 | word_bigram | random_forest | 0.5270 | 0.4269 | 0.4097 | 0.3520 | 0.4483 | 1.16 |

## Leading models

- logistic_regression: exp04 combined; macro-F1 0.6646; accuracy 0.7050; negative/neutral/positive F1 0.5251/0.7160/0.7525; fit 15.94s.
- linear_svm: exp07 combined; macro-F1 0.6631; accuracy 0.6905; negative/neutral/positive F1 0.5626/0.6923/0.7344; fit 6.84s.
- random_forest: exp09 word_bigram; macro-F1 0.3520; accuracy 0.5270; negative/neutral/positive F1 0.0000/0.6375/0.4184; fit 1.16s.
- naive_bayes: exp08 word_bigram; macro-F1 0.4627; accuracy 0.6255; negative/neutral/positive F1 0.0314/0.6660/0.6905; fit 0.09s.

## Selected candidate and errors

VALIDATION-SELECTED CANDIDATE: exp04 combined + logistic_regression. Macro-F1 0.6646; accuracy 0.7050. THIS IS NOT YET THE FINAL TEST-EVALUATED MODEL.
Winner VALIDATION confusion rows (negative, neutral, positive): [[141, 135, 36], [59, 667, 143], [25, 192, 602]]. Off-diagonal cells count misclassifications; see error_analysis.md for controlled examples and cue counts.
The top learned linear coefficients in model_metrics.json are TRAIN-fitted feature associations, not causal explanations. Positive coefficients favor the corresponding class relative to alternatives.
Random Forest used all TRAIN rows with bounded depth and 80 trees. See its negative recall and fit time above; the constrained configuration is not a full-capacity forest.

## Limits and next work

Historical English tweets, class imbalance, sarcasm, context loss and benchmark annotation ambiguity remain. Lowercased clean text loses case cues. Default word vectorization omits emoji, while character features may represent them indirectly. No brand-specific or prospective evaluation is claimed.
Next: focused validation-guided tuning of the strongest candidates and tokenizer checks, freeze the procedure, then perform one locked TEST evaluation. Do not call this artifact production-ready.
Full per-class precision/recall/F1, confusion matrices, feature/model parameters, versions and times are in model_metrics.json. Model artifact regeneration: python -m src.model_pipeline.

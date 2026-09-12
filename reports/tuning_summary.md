# Focused TRAIN-only tuning

3-fold stratified CV (shuffle, seed 42), macro-F1 scoring. Three folds limit runtime and memory for 216,107-dimensional combined sparse features. TF-IDF and IDF are fitted inside each training fold by the sklearn Pipeline. VALIDATION is excluded from the search; TEST has no predictive use.

30 configurations: 10 classifier settings and 5 focused feature settings per model, each run on 3 folds. CV executes serially to limit memory duplication.

## TRAIN CV winners

- logistic_regression: mean macro-F1 0.6545 ± 0.0021; parameters {'classifier__C': 1.0, 'classifier__class_weight': 'balanced', 'features__transformer_weights': {'character': 1.25, 'word': 1.0}}.
- linear_svm: mean macro-F1 0.6517 ± 0.0016; parameters {'classifier__C': 0.25, 'classifier__class_weight': 'balanced'}.

## VALIDATION checkpoint

| Model | Accuracy | Macro precision | Macro recall | Macro-F1 | Negative F1 | Neutral F1 | Positive F1 | Fit s | Predict s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.6925 | 0.6642 | 0.6891 | 0.6728 | 0.5850 | 0.6861 | 0.7475 | 31.2 | 0.480 |
| linear_svm | 0.6975 | 0.6693 | 0.6771 | 0.6726 | 0.5710 | 0.6994 | 0.7473 | 16.9 | 0.508 |

Selected: **logistic_regression** by VALIDATION macro-F1, considering CV stability, negative F1 and runtime as secondary evidence.
This is a frozen development candidate; TEST has not been evaluated. The paired bootstrap interval in model_uncertainty.md describes remaining selection uncertainty.
Full parameters, scores, per-class metrics and versions are in tuning_metrics.json and frozen_candidate_metadata.json.

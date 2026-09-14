# Post-benchmark experimental sentiment modeling

This is development research on TRAIN (45,615 rows) and VALIDATION (2,000 rows). No new TEST predictions or metrics were generated. The one official benchmark and its final model remain frozen.

## Method

- Frozen balanced LR and LinearSVC combined word (1,2) plus character (3,5) TF-IDF pipelines.
- Three-fold shuffled stratified TRAIN CV, seed 42, macro-F1 selection.
- Hard voting with two estimators ties to LR, so it is exactly LR on every row.
- Soft voting tests five LR/SVM weights. LinearSVC sigmoid probabilities are fitted with inner TRAIN-only calibration CV; CV selects the weight.
- Stacking trains a six-dimensional LR-probability/SVM-margin meta-model from inner out-of-fold predictions, with no TF-IDF passthrough. Four meta configurations are selected by outer TRAIN CV.
- XGBoost uses 10 selected tree/weighting configurations, 30 fold fits, sparse CSR TF-IDF and no dense conversion. Each fold fits its own vocabulary and IDF.

## Comparison

| Model | CV F1 | CV std | VAL accuracy | VAL macro precision | VAL macro recall | VAL macro-F1 | Negative F1 | Neutral F1 | Positive F1 | Fit s | Predict s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.6545 | 0.0021 | 0.6925 | 0.6642 | 0.6891 | 0.6728 | 0.5850 | 0.6861 | 0.7475 | 21.7 | 0.302 |
| linear_svm | 0.6517 | 0.0016 | 0.6975 | 0.6693 | 0.6771 | 0.6726 | 0.5710 | 0.6994 | 0.7473 | 12.3 | 0.301 |
| hard_vote | 0.6545 | 0.0021 | 0.6925 | 0.6642 | 0.6891 | 0.6728 | 0.5850 | 0.6861 | 0.7475 | 34.1 | 0.603 |
| soft_vote | 0.6558 | 0.0017 | 0.7015 | 0.6726 | 0.6845 | 0.6776 | 0.5787 | 0.7010 | 0.7531 | 53.1 | 1.196 |
| stacking | 0.6480 | 0.0019 | 0.7060 | 0.6916 | 0.6692 | 0.6781 | 0.5754 | 0.7109 | 0.7479 | 119.0 | 0.602 |
| xgboost | 0.5657 | 0.0011 | 0.5630 | 0.5569 | 0.5627 | 0.5412 | 0.4450 | 0.6087 | 0.5699 | 69.0 | 0.167 |

## Diagnostics

LR/SVM both correct: 1329; both wrong: 549; LR only correct: 56; SVM only correct: 66. Disagreements: 138 (6.90%).

Selected soft weights: LR 0.70, SVM 0.30. Sigmoid calibration used 3 inner TRAIN folds. Soft voting improved on the frozen LR VALIDATION baseline; this is a combined prediction result, not evidence that calibration by itself improved classification.

Stack base estimators: ['logistic_regression', 'linear_svm']; outer/inner OOF folds: 3/3; stack methods: ['predict_proba', 'decision_function']; meta-feature dimensions: 6; best meta parameters: {'C': 1.0, 'class_weight': None}.

XGBoost 3.2.0 best parameters: {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 40, 'subsample': 0.8, 'colsample_bytree': 0.5, 'min_child_weight': 3, 'reg_lambda': 1, 'sample_weight_policy': 'balanced'}. CV macro-F1 0.5657; VALIDATION macro-F1 0.5412; final fit 69.0s, prediction 0.167s; all CV model fits 2095.0s. The TF-IDF matrices remained CSR sparse.

## Paired bootstrap and decision

Best experimental model: **stacking**, VALIDATION macro-F1 0.6781, difference from frozen LR +0.0052. The paired bootstrap for stacking minus LR gives +0.0052, 95% percentile interval [-0.0109, +0.0211] over 2000 seeded draws. Selection on this same holdout limits inference; this is a sensitivity diagnostic.

Decision: **No stable, meaningful improvement; retain the simpler Logistic Regression for application work.**

Saved fitted experimental artifacts: none; the observed improvement did not satisfy the meaningful/stable threshold.

All new models remain EXPERIMENTAL DEVELOPMENT MODELS. The official TweetEval TEST result is a completed historical result and is not reused.

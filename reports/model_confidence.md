# VALIDATION confidence and decision margins

Logistic Regression uses maximum predicted probability. LinearSVC uses the gap between the highest and second-highest decision scores. Their scales are different; SVM margins are not probabilities.

## logistic_regression

Measure: maximum predicted probability.
Median correct: 0.7328; median incorrect: 0.5864.
Error rate below model-specific median: 0.4420; at/above median: 0.1730.
Incorrect predictions at/above median: 173.

## linear_svm

Measure: top-versus-second decision margin.
Median correct: 0.8145; median incorrect: 0.3759.
Error rate below model-specific median: 0.4420; at/above median: 0.1630.
Incorrect predictions at/above median: 163.

These are diagnostics on VALIDATION, not calibrated reliability guarantees. High-confidence errors warrant manual review before application use.

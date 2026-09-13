# Frozen final evaluation protocol

The model-development decision is closed. The frozen candidate is the combined word (1,2) and character (3,5) TF-IDF Pipeline with Logistic Regression, C=1, balanced class weights, lbfgs, word weight 1.0, character weight 1.25, and all remaining parameters from models/candidates/frozen_candidate_metadata.json. The pre-TEST procedure verifies that metadata, its artifact checksum, preprocessing configuration, random seed and scikit-learn version before fitting a fresh Pipeline. It does not reuse the TRAIN-only fitted candidate as the final estimator.

The official final fit concatenates unchanged TRAIN (45,615) and VALIDATION (2,000) rows in that order, preserving all 47,615 rows and their official labels. No deduplication, resplitting or manual rebalancing occurs. TEST (12,284 rows) is checked mechanically for existence, schema, split membership, count and checksum before any predictive access.

The final evaluation is one explicitly unlocked action, permitted only after this procedure and its synthetic tests have been committed, pushed to main, and passed GitHub Actions. Its command requires the full SHA of that pre-TEST commit:

    python -m src.final_evaluation --evaluate-locked-test --pretest-sha FULL_PRETEST_COMMIT_SHA

A normal invocation only reports whether an official result already exists; it does not evaluate TEST. The explicit action also requires a clean checkout whose HEAD and origin/main equal the supplied SHA. It creates an exclusive attempt marker immediately before opening TEST labels. A prior attempt or completed manifest blocks a repeat. If an attempt fails after the marker is created, the marker remains and requires a manual technical audit; the tool does not silently retry.

The one official pipeline predicts TEST once and calculates one metric set: accuracy, macro precision/recall/F1, weighted F1, per-class precision/recall/F1/support, and a 3×3 confusion matrix in negative, neutral, positive order. The final joblib artifact is saved and reloaded. Its prediction on the same TEST inputs is compared for exact equality as a persistence check only; no second metric set is calculated. Row-level label/index predictions stay under ignored data/processed, and public reports contain aggregate metrics only, with no raw TEST text.

No classifier, parameter, preprocessing, feature, weighting or threshold decision may change based on TEST. Validation-versus-TEST differences are descriptive. The completed evaluation writes a manifest containing the pre-TEST SHA, processed-TEST checksum, model checksum and metrics checksum. The final result may be accepted even if it is below VALIDATION performance.

The offline test suite and synthetic dry run exercise this procedure with small fixture splits; CI never downloads TweetEval or scores the official TEST. After the one evaluation, default commands and notebooks read persisted results. Do not use the explicit unlock command for routine reproduction or experimentation.

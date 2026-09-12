# Candidate artifact

best_candidate.joblib is a VALIDATION-SELECTED CANDIDATE, not a final test-evaluated model. It contains the fitted TF-IDF FeatureUnion and Logistic Regression classifier. best_candidate_metadata.json records the exact experiment, validation score, SHA-256, file size and reload-prediction check. Only load this joblib file from a trusted checkout.

Regenerate from official processed data with python -m src.model_pipeline. The candidate is approximately 4.4 MB and is included in Git for reviewability. No TEST predictions or metrics were produced.

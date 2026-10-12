# External evaluation reports

The YouTube-only collection, human labeling, and frozen-model external evaluation are complete. Safe aggregate records in this directory include the environment check, live smoke status, collection summary, label-validation summary, protected artifact hashes, metrics, dataset summary, confusion matrix, and text-free run manifests under `data/external_eval/manifests/`.

Human review excluded 69 language-ineligible rows from the 400-row sample. The remaining 331 labels passed validation. The frozen model achieved 0.5982 accuracy and 0.5406 macro-F1. The official TweetEval benchmark remains unchanged.

These public files must not contain raw comment text, usernames, credentials, or row-level predictions.

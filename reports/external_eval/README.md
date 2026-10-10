# External evaluation reports

The YouTube-only collection and preparation workflow is complete and waiting for human labeling. Safe aggregate records in this directory include the environment check, live smoke status, collection summary, protected artifact hashes, and text-free run manifests under `data/external_eval/manifests/`.

No external metric has been generated. The local 400-row labeling CSV contains blank labels, and the official TweetEval benchmark remains unchanged. After human labels pass validation, expected aggregate outputs are `external_metrics.json`, `external_summary.md`, `external_confusion_matrix.png`, and `dataset_summary.json`.

These public files must not contain raw comment text, usernames, credentials, or row-level predictions.

# External brand-domain evaluation data

Real external-evaluation records remain local and are ignored by Git:

- `raw/`: privacy-minimized API collection JSONL
- `prepared/`: filtered, deduplicated, sampled JSONL
- `labels/`: blank labeling exports and completed human labels
- `manifests/`: text-free collection and preparation audit records

Do not derive this set from TweetEval TEST or commit raw comments, usernames, credentials, or row-level model output. Track only code, configuration templates, synthetic fixtures, guidelines, reviewed text-free manifests, and aggregate reports.

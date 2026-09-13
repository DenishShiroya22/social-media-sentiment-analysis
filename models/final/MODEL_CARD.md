# Final TweetEval sentiment model card

## Intended use

A research baseline for three-class sentiment on TweetEval-style English posts and a starting point for later brand-monitoring evaluation. It is not a substitute for human moderation and is not suitable for psychological inference.

## Dataset and training

Cardiff NLP TweetEval sentiment, pinned source revision recorded in data/raw/manifest.json. Official TRAIN (45,615) and VALIDATION (2,000) rows were combined after model selection, giving 47,615 final fitting rows. No deduplication, resplitting or resampling. One official TEST evaluation used 12,284 untouched rows.

## Frozen pipeline

Deterministic preprocessing with targeted escaped-punctuation normalization; word (1,2) and char_wb (3,5) TF-IDF; word weight 1.0 and character weight 1.25; balanced Logistic Regression (C=1, lbfgs, max_iter=350, seed 42). Exact parameters and environment are in final_model_metadata.json.

## One official TEST result

Accuracy 0.6228; macro precision 0.6146; macro recall 0.6309; macro-F1 0.6199; weighted F1 0.6223.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| negative | 0.5945 | 0.6944 | 0.6406 | 3,972 |
| neutral | 0.6710 | 0.5744 | 0.6189 | 5,937 |
| positive | 0.5782 | 0.6240 | 0.6002 | 2,375 |

## Limits and ethics

Historical tweets may not transfer to modern, brand-specific comments. Sarcasm, short or missing context, mixed sentiment, annotation ambiguity and class imbalance remain. Do not treat labels as objective psychological facts. Future collection should respect platform terms and avoid unnecessary personal information. Raw/processed dataset CSVs and row-level TEST predictions are not published. Check upstream dataset rights before redistribution.

Pre-TEST procedure commit: 7922527d9599567887de2a5836e84a4e7e89458e. No tuning followed the TEST result.

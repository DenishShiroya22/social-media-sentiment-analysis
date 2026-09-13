# One official locked-TEST sentiment evaluation

Pre-TEST procedure commit: 7922527d9599567887de2a5836e84a4e7e89458e. Protocol version: 1.0.
The frozen combined TF-IDF plus Logistic Regression procedure was fitted once on TRAIN+VALIDATION, then evaluated once on untouched TEST. No competing model or TEST-driven tuning was performed.

Fit rows: TRAIN 45,615 + VALIDATION 2,000 = 47,615. TEST rows: 12,284.
Final fitting took 23.76 seconds; one official prediction took 1.62 seconds.

## Official TEST metrics

Accuracy 0.6228; macro precision 0.6146; macro recall 0.6309; **macro-F1 0.6199**; weighted F1 0.6223.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| negative | 0.5945 | 0.6944 | 0.6406 | 3,972 |
| neutral | 0.6710 | 0.5744 | 0.6189 | 5,937 |
| positive | 0.5782 | 0.6240 | 0.6002 | 2,375 |

Confusion rows are actual and columns predicted, ordered negative, neutral, positive:

[[2758, 1003, 211], [1657, 3410, 870], [224, 669, 1482]]

## Descriptive VALIDATION versus TEST

| Metric | VALIDATION | TEST | TEST − VALIDATION |
|---|---:|---:|---:|
| Accuracy | 0.6925 | 0.6228 | -0.0697 |
| Macro-F1 | 0.6728 | 0.6199 | -0.0529 |
| Negative F1 | 0.5850 | 0.6406 | +0.0556 |
| Neutral F1 | 0.6861 | 0.6189 | -0.0672 |
| Positive F1 | 0.7475 | 0.6002 | -0.1473 |

The VALIDATION score belongs to the development candidate fitted on TRAIN only. The TEST score belongs to the same frozen configuration refitted on TRAIN+VALIDATION. This gap is descriptive, not a basis for changing the model.

## Artifact and safeguards

Final fitted artifact: models/final/final_model.joblib; 4,418,974 bytes; SHA-256 a3e375cc0e75c68f2ac8de8a35d46369e566c3e0821d31c778018f4edc83451e. Reloaded predictions matched the original vector exactly. The second prediction call is solely an artifact-integrity comparison; no second metric set was calculated.
The label-only prediction file is kept locally under data/processed and excluded from Git. No raw TEST tweet examples appear in this report.
An evaluation manifest blocks accidental reruns. The recorded pre-TEST commit was pushed and CI-passing before TEST was unlocked.

## Limits and next work

TweetEval contains historical English tweets. Sarcasm, missing context, annotation ambiguity, class imbalance and modern brand-domain shift limit real-world use. This benchmark result is not a guarantee for current brand comments or psychological inference. No post-TEST parameter, preprocessing or model changes were made. Next: build a reusable inference service, then evaluate responsibly on real public social-media data and design aggregate analytics.

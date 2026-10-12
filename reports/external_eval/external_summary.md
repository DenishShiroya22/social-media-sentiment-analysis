# External brand-domain evaluation

Rows evaluated: 331

The frozen TweetEval-trained Logistic Regression model was evaluated without retraining, threshold changes, recalibration, or model selection.

## External metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.5982 |
| Macro precision | 0.5374 |
| Macro recall | 0.5462 |
| Macro-F1 | 0.5406 |
| Weighted F1 | 0.6041 |

## Per-class metrics

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| negative | 0.2727 | 0.3214 | 0.2951 | 56 |
| neutral | 0.7041 | 0.6467 | 0.6742 | 184 |
| positive | 0.6354 | 0.6703 | 0.6524 | 91 |

## Domain-gap comparison

| Metric | TweetEval TEST | External domain | Difference |
|---|---:|---:|---:|
| accuracy | 0.6228 | 0.5982 | -0.0246 |
| macro_f1 | 0.6199 | 0.5406 | -0.0793 |
| negative_f1 | 0.6406 | 0.2951 | -0.3455 |
| neutral_f1 | 0.6189 | 0.6742 | +0.0553 |
| positive_f1 | 0.6002 | 0.6524 | +0.0522 |

TweetEval TEST and the external brand/product sample come from different distributions and are not directly interchangeable. These results describe external-domain transfer only.

The gap can reflect a platform shift from historical tweets to YouTube comments, newer language and slang, brand/product-specific vocabulary, sarcasm, missing video or conversation context, and the external sample's different class and query composition. The large negative-class difference is evidence about this sample and does not trigger changes to the frozen model.

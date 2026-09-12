# Finalist VALIDATION errors

Rows are true labels and columns predictions in negative, neutral, positive order. Cue associations are descriptive, not causal. TEST is locked.

## logistic_regression

Total errors: 615.
Confusion matrix: [[210, 72, 30], [139, 577, 153], [57, 164, 598]].

| Cue | Rows | Errors | Error rate |
|---|---:|---:|---:|
| negation | 330 | 123 | 0.373 |
| contraction | 767 | 256 | 0.334 |
| emoticon | 52 | 11 | 0.212 |
| punctuation | 619 | 184 | 0.297 |
| short_5_words | 4 | 0 | 0.000 |

## linear_svm

Total errors: 605.
Confusion matrix: [[187, 93, 32], [109, 612, 148], [47, 176, 596]].

| Cue | Rows | Errors | Error rate |
|---|---:|---:|---:|
| negation | 330 | 123 | 0.373 |
| contraction | 767 | 245 | 0.319 |
| emoticon | 52 | 10 | 0.192 |
| punctuation | 619 | 185 | 0.299 |
| short_5_words | 4 | 0 | 0.000 |

## Off-diagonal comparison

| True → predicted | Logistic Regression | LinearSVC |
|---|---:|---:|
| negative → neutral | 72 | 93 |
| negative → positive | 30 | 32 |
| neutral → negative | 139 | 109 |
| neutral → positive | 153 | 148 |
| positive → negative | 57 | 47 |
| positive → neutral | 164 | 176 |

LR catches more negative posts (210 vs 187 true negatives in the negative class row) but also predicts negative more often for neutral posts (139 vs 109). LinearSVC handles the neutral class more accurately. Cue-specific error rates are close and remain descriptive.
Cue groups overlap and have different denominators. Short context, negation, contractions, emoticons and punctuation do not establish causes of error.

# Escaped Unicode punctuation correction

Measured on TRAIN and VALIDATION raw text only. TEST content was not inspected. Literal escaped punctuation was decoded by an explicit mapping; unrelated escapes and existing Unicode are preserved.

TRAIN affected rows: 5,111; VALIDATION affected rows: 206.
Common TRAIN occurrences: literal \\u002c 4,664; literal \\u2019 4,408.
Common VALIDATION occurrences: literal \\u002c 179; literal \\u2019 181.

TRAIN examples: literal can\u2019t → can't; literal red\u002c → red,.

## Same-config baseline comparison

| Experiment | Old macro-F1 | New macro-F1 | Delta | New accuracy | Old negative F1 | New negative F1 |
|---|---:|---:|---:|---:|---:|---:|
| exp04 | 0.6646 | 0.6580 | -0.0066 | 0.7000 | 0.5251 | 0.5130 |
| exp07 | 0.6631 | 0.6536 | -0.0095 | 0.6830 | 0.5626 | 0.5451 |
| exp03 | 0.6500 | 0.6463 | -0.0037 | 0.6900 | 0.5097 | 0.4972 |
| exp06 | 0.6393 | 0.6382 | -0.0010 | 0.6705 | 0.5133 | 0.5219 |
| exp02 | 0.6363 | 0.6367 | +0.0004 | 0.6810 | 0.4921 | 0.4921 |
| exp05 | 0.6362 | 0.6337 | -0.0025 | 0.6680 | 0.5160 | 0.5115 |
| exp01 | 0.6236 | 0.6230 | -0.0006 | 0.6730 | 0.4640 | 0.4600 |
| exp08 | 0.4627 | 0.4592 | -0.0034 | 0.6260 | 0.0314 | 0.0190 |
| exp09 | 0.3520 | 0.3651 | +0.0131 | 0.5405 | 0.0000 | 0.0000 |

Old ranking: exp04, exp07, exp03, exp06, exp02, exp05, exp01, exp08, exp09.
New ranking: exp04, exp07, exp03, exp06, exp02, exp05, exp01, exp08, exp09.
The top two remain combined Logistic Regression and combined LinearSVC. Character-only models remain next; combined features remain strongest within each linear family. Lower corrected scores suggest the escape artifacts had become predictive on validation. These changes reflect preprocessing, not a new dataset split.

## Learned-feature check

Artificial u2019/u002c feature count in fitted combined vocabulary: 0.
Contraction representation in fitted vocabulary: {"don't": True, "can't wait": True, "isn't": True}.
Feature presence reflects TRAIN vocabulary; coefficients do not establish causal importance.

# Classical sentiment modeling

## Objective and data contract

Compare TRAIN-fitted sparse TF-IDF representations and classical classifiers on the official TweetEval VALIDATION split. Primary selection metric: macro-F1, giving negative, neutral and positive equal weight. Secondary metrics: accuracy, macro precision/recall, weighted-F1, per-class precision/recall/F1, and confusion matrix. Input CSVs have text, clean_text, sentiment and split; clean_text is used for modeling.

TRAIN has 45,615 rows (negative 7,093; neutral 20,673; positive 17,849); VALIDATION has 2,000. TEST has 12,284 rows and remains locked. No random splitting, oversampling, feature fitting on validation, or test-label analysis occurs. The test file is read only for header, split membership, row count and checksum; its labels and text are never passed to development functions.

## TF-IDF and n-grams

TF-IDF combines term frequency in a post with inverse document frequency across the TRAIN corpus. A common term gets less weight than a discriminative term. Word unigrams for not good are not and good; the bigram not good retains phrase context. Character 3–5-grams can represent fragments of contractions, repeated letters, emoticons and punctuation that word tokenization may miss.

Compared representations:

| Name | Analyzer | N-grams | Minimum document frequency | Maximum document frequency | Feature cap |
|---|---|---|---:|---:|---:|
| word_unigram | word | 1 | 2 | 0.95 | 100,000 |
| word_bigram | word | 1–2 | 2 | 0.95 | 100,000 |
| character | char_wb | 3–5 | 2 | 0.95 | 120,000 |
| combined | FeatureUnion of word_bigram + character | as above | as above | as above | 220,000 maximum nominal |

All use sublinear TF, L2 normalization, float32, and lowercase=False because clean_text is already lowercased. Word token pattern retains apostrophe contractions. No custom tokenizer. Word trigrams are deferred because the controlled matrix already covers the main representation differences at reasonable resource cost.

## Algorithms and experiment matrix

| ID | Features | Model |
|---|---|---|
| exp01 | word_unigram | Logistic Regression |
| exp02 | word_bigram | Logistic Regression |
| exp03 | character | Logistic Regression |
| exp04 | combined | Logistic Regression |
| exp05 | word_bigram | LinearSVC |
| exp06 | character | LinearSVC |
| exp07 | combined | LinearSVC |
| exp08 | word_bigram | MultinomialNB |
| exp09 | word_bigram | Random Forest |

Logistic Regression: C=1, lbfgs, max_iter=350. LinearSVC: C=1, max_iter=3000. Naive Bayes: alpha=1. Random Forest: 80 trees, depth 24, minimum leaf 2, sqrt feature sampling, two jobs. Applicable seeds use 42. These are baselines, not exhaustive hyperparameter tuning. Full TRAIN is used for every experiment, including Random Forest.

## Executed validation results

| Experiment | Model | Features | Accuracy | Macro-F1 | Negative F1 | Neutral F1 | Positive F1 | Fit seconds |
|---|---|---|---:|---:|---:|---:|---:|---:|
| exp04 | logistic_regression | combined | 0.7050 | 0.6646 | 0.5251 | 0.7160 | 0.7525 | 15.94 |
| exp07 | linear_svm | combined | 0.6905 | 0.6631 | 0.5626 | 0.6923 | 0.7344 | 6.84 |
| exp03 | logistic_regression | character | 0.6920 | 0.6500 | 0.5097 | 0.7076 | 0.7329 | 12.73 |
| exp06 | linear_svm | character | 0.6745 | 0.6393 | 0.5133 | 0.6824 | 0.7221 | 4.86 |
| exp02 | logistic_regression | word_bigram | 0.6805 | 0.6363 | 0.4921 | 0.7002 | 0.7166 | 4.23 |
| exp05 | linear_svm | word_bigram | 0.6700 | 0.6362 | 0.5160 | 0.6776 | 0.7151 | 1.66 |
| exp01 | logistic_regression | word_unigram | 0.6725 | 0.6236 | 0.4640 | 0.6921 | 0.7146 | 1.35 |
| exp08 | naive_bayes | word_bigram | 0.6255 | 0.4627 | 0.0314 | 0.6660 | 0.6905 | 0.09 |
| exp09 | random_forest | word_bigram | 0.5270 | 0.3520 | 0.0000 | 0.6375 | 0.4184 | 1.16 |

These numbers were produced by the executed full-data run; reports/model_comparison.csv and reports/model_metrics.json contain unrounded values, all requested metrics, parameters and environment versions. In this run, combined features improved both leading linear models over their word-only variants. Random Forest was fast under the bounded configuration but missed every negative validation example, so its macro-F1 was poor. Naive Bayes was also weak on negative recall (0.0160). This is a comparison of these fixed configurations only, not a general ranking of algorithms.

## Validation-selected candidate and interpretation

exp04, combined TF-IDF plus Logistic Regression, is selected by VALIDATION macro-F1 0.664563 (accuracy 0.7050). The saved models/candidates/best_candidate.joblib is a fitted sklearn Pipeline. Metadata includes SHA-256 and confirms predictions after reload exactly matched pre-save VALIDATION predictions. This is not the final test-evaluated model.

Winner VALIDATION confusion matrix, rows true and columns predicted in negative, neutral, positive order:

    [[141, 135, 36],
     [ 59, 667, 143],
     [ 25, 192, 602]]

The largest off-diagonal counts are positive→neutral (192), neutral→positive (143) and negative→neutral (135). Negative recall is 0.4519, so minority-class performance needs focused attention. Validation error_analysis.md gives controlled examples. Negative examples include negative affect and negation within event/price context; neutral/positive boundaries can depend on unstated attitude or future expectations. Samples are illustrative, not proof of causes. The error report counts 102 misclassified posts with explicit negation tokens and 178 with exclamation/question marks.

TRAIN-learned Logistic Regression coefficients associate negative with sad, not, worst and :( fragments; positive with can't wait, happy, good, great and :) fragments. Neutral top coefficients include do you and last day. These are associations in fitted TRAIN features, not causal explanations; some weights reflect dataset topics or annotation patterns.

## Artifacts, execution and limitations

Run from repository root:

    python -m src.data_acquisition
    python -m src.pipeline
    python -m src.model_pipeline
    python -m pytest -q
    python scripts/execute_notebooks.py

Outputs: reports/model_comparison.csv; reports/model_metrics.json; reports/model_summary.md; reports/error_analysis.md; modeling figures under reports/figures/modeling; models/candidates/best_candidate.joblib and metadata; notebooks/03_model_development.ipynb. The 4.4 MB artifact is checked into this repository. Only load trusted joblib files; regenerate with the modeling command if library versions change.

Historical English tweets differ from contemporary brand comments. Sarcasm, context, short messages, domain drift and annotation ambiguity remain. Macro-F1 improves class balance awareness but negative recall remains modest. A validation-selected winner may still disappoint on untouched data. Next work: focused tuning of the leading linear candidates, evaluate minority-class tradeoffs on VALIDATION, freeze the full procedure, then run one final locked TEST evaluation. Do not refit on TRAIN+VALIDATION or compute TEST metrics during the current development iteration.

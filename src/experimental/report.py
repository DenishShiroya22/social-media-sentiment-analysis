"""Publish a human-readable comparison from the persisted development reports."""
import json
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT
from src.experimental.ensemble import assert_official_unchanged


def write_summary(root=PROJECT_ROOT):
    root = Path(root)
    assert_official_unchanged(root)
    out = root/'reports/experimental'
    ensemble = json.loads((out/'ensemble_metrics.json').read_text(encoding='utf-8'))
    xgboost = json.loads((out/'xgboost_metrics.json').read_text(encoding='utf-8'))
    bootstrap = json.loads((out/'bootstrap_comparison.json').read_text(encoding='utf-8'))
    table = pd.read_csv(out/'ensemble_comparison.csv')
    if set(table['model']) != {'logistic_regression', 'linear_svm', 'hard_vote',
                                'soft_vote', 'stacking', 'xgboost'}:
        raise ValueError('Comparison table is incomplete')
    baseline = table.set_index('model').loc['logistic_regression']
    best = table.set_index('model').loc[
        table[~table['model'].isin(('logistic_regression', 'linear_svm', 'hard_vote'))]
        .sort_values('validation_macro_f1', ascending=False).iloc[0]['model']]
    best_name = best.name
    delta = best['validation_macro_f1'] - baseline['validation_macro_f1']
    ci = bootstrap['ci_95_percentile']
    if delta >= 0.01 and ci[0] > 0:
        decision = 'Meaningful development improvement; validate on a new external labeled brand-comment dataset.'
    elif delta >= 0.005 and ci[0] > 0:
        decision = 'Small experimental improvement; validate on a new external labeled brand-comment dataset.'
    else:
        decision = 'No stable, meaningful improvement; retain the simpler Logistic Regression for application work.'
    lines = [
        '# Post-benchmark experimental sentiment modeling',
        '',
        'This is development research on TRAIN (45,615 rows) and VALIDATION (2,000 rows). '
        'No new TEST predictions or metrics were generated. The one official benchmark '
        'and its final model remain frozen.',
        '',
        '## Method',
        '',
        '- Frozen balanced LR and LinearSVC combined word (1,2) plus character (3,5) TF-IDF pipelines.',
        '- Three-fold shuffled stratified TRAIN CV, seed 42, macro-F1 selection.',
        '- Hard voting with two estimators ties to LR, so it is exactly LR on every row.',
        '- Soft voting tests five LR/SVM weights. LinearSVC sigmoid probabilities are fitted '
        'with inner TRAIN-only calibration CV; CV selects the weight.',
        '- Stacking trains a six-dimensional LR-probability/SVM-margin meta-model from '
        'inner out-of-fold predictions, with no TF-IDF passthrough. Four meta configurations '
        'are selected by outer TRAIN CV.',
        '- XGBoost uses 10 selected tree/weighting configurations, 30 fold fits, '
        'sparse CSR TF-IDF and no dense conversion. Each fold fits its own vocabulary and IDF.',
        '',
        '## Comparison',
        '',
        '| Model | CV F1 | CV std | VAL accuracy | VAL macro precision | VAL macro recall | '
        'VAL macro-F1 | Negative F1 | Neutral F1 | Positive F1 | Fit s | Predict s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for _, row in table.iterrows():
        lines.append('| ' + ' | '.join([
            str(row['model']), f"{row['cv_macro_f1']:.4f}",
            f"{row['cv_std']:.4f}", f"{row['validation_accuracy']:.4f}",
            f"{row['validation_macro_precision']:.4f}",
            f"{row['validation_macro_recall']:.4f}",
            f"{row['validation_macro_f1']:.4f}",
            f"{row['negative_f1']:.4f}", f"{row['neutral_f1']:.4f}",
            f"{row['positive_f1']:.4f}", f"{row['fit_seconds']:.1f}",
            f"{row['predict_seconds']:.3f}"]) + ' |')
    diversity = ensemble['diversity']
    soft = ensemble['records']['soft_vote']
    stack = ensemble['records']['stacking']
    xgb = xgboost['record']
    lines += [
        '',
        '## Diagnostics',
        '',
        f"LR/SVM both correct: {diversity['both_correct']}; both wrong: "
        f"{diversity['both_wrong']}; LR only correct: {diversity['lr_only_correct']}; "
        f"SVM only correct: {diversity['svm_only_correct']}. "
        f"Disagreements: {diversity['prediction_disagreements']} "
        f"({diversity['disagreement_rate']:.2%}).",
        '',
        f"Selected soft weights: LR {soft['details']['lr_weight']:.2f}, "
        f"SVM {soft['details']['svm_weight']:.2f}. Sigmoid calibration used "
        f"{soft['details']['calibration_cv_folds']} inner TRAIN folds. "
        f"Soft voting {'improved' if soft['validation']['macro_f1'] > baseline['validation_macro_f1'] else 'did not improve'} "
        'on the frozen LR VALIDATION baseline; this is a combined prediction result, '
        'not evidence that calibration by itself improved classification.',
        '',
        f"Stack base estimators: {stack['details']['base_estimators']}; "
        f"outer/inner OOF folds: {stack['details']['outer_cv_folds']}/"
        f"{stack['details']['inner_oof_folds']}; stack methods: "
        f"{stack['details']['stack_method']}; meta-feature dimensions: "
        f"{stack['details']['meta_feature_dimensions']}; "
        f"best meta parameters: {stack['details']['meta_params']}.",
        '',
        f"XGBoost {xgb['details']['xgboost_version']} best parameters: "
        f"{xgb['details']['params']}. CV macro-F1 {xgb['cv_macro_f1']:.4f}; "
        f"VALIDATION macro-F1 {xgb['validation']['macro_f1']:.4f}; "
        f"final fit {xgb['fit_seconds']:.1f}s, prediction {xgb['predict_seconds']:.3f}s; "
        f"all CV model fits {sum(r['total_cv_fit_seconds'] for r in xgboost['search']):.1f}s. "
        'The TF-IDF matrices remained CSR sparse.',
        '',
        '## Paired bootstrap and decision',
        '',
        f"Best experimental model: **{best_name}**, VALIDATION macro-F1 "
        f"{best['validation_macro_f1']:.4f}, difference from frozen LR "
        f"{delta:+.4f}. The paired bootstrap for {bootstrap['candidate']} "
        f"minus LR gives {bootstrap['difference_candidate_minus_lr']:+.4f}, "
        f"95% percentile interval [{ci[0]:+.4f}, {ci[1]:+.4f}] "
        f"over {bootstrap['draws']} seeded draws. "
        'Selection on this same holdout limits inference; this is a sensitivity diagnostic.',
        '',
        f"Decision: **{decision}**",
        '',
        'Saved fitted experimental artifacts: ' +
        (', '.join(p.name for p in sorted((root/'models/experimental').glob('*.joblib')))
         or 'none; the observed improvement did not satisfy the meaningful/stable threshold') + '.',
        '',
        'All new models remain EXPERIMENTAL DEVELOPMENT MODELS. '
        'The official TweetEval TEST result is a completed historical result and is not reused.',
        '',
    ]
    path = out/'ensemble_summary.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    assert_official_unchanged(root)
    return {'best_model': best_name, 'difference_from_lr': float(delta),
            'bootstrap': bootstrap, 'decision': decision, 'summary_path': str(path)}


if __name__ == '__main__':
    print(write_summary())

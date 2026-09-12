"""Train-only TF-IDF fitting and validation-only model comparison."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import PROJECT_ROOT, EXPECTED_ROWS, LABELS, RANDOM_STATE
from .data_loader import load_csv
from .data_validation import require_valid, validate_dataframe
from .evaluation import evaluate_validation, select_winner
from .features import make_features, fit_train_transform_validation, feature_parameters
from .models import EXPERIMENTS, make_model

logger = logging.getLogger(__name__)

def check_locked_test(path: Path, expected_rows: int, expected_sha256: str | None = None) -> dict:
    """Read only header/split/count; never expose or inspect TEST labels/text."""
    if not path.is_file():
        raise FileNotFoundError(f'Locked TEST file not found: {path}')
    if expected_sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('Locked TEST checksum mismatch')
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'text', 'clean_text', 'sentiment', 'split'} <= set(reader.fieldnames or []):
            raise ValueError('Locked TEST schema mismatch')
        count = 0
        for row in reader:
            if row['split'] != 'test':
                raise ValueError('Locked TEST split membership mismatch')
            count += 1
    if count != expected_rows:
        raise ValueError(f'Locked TEST row count mismatch: {count} != {expected_rows}')
    return {'exists': True, 'schema_valid': True, 'split_valid': True,
            'rows': count, 'checksum_valid': expected_sha256 is not None}

def load_development_data(root: Path, expected_rows: dict = EXPECTED_ROWS):
    """Load TRAIN and VALIDATION labels; test is only checked mechanically."""
    processed_dir = root / 'data' / 'processed'
    frames = {}
    for name in ('train', 'validation'):
        frame = load_csv(processed_dir / f'{name}_clean.csv')
        require_valid(validate_dataframe(frame, name))
        require_valid(validate_dataframe(frame, name, text_column='clean_text'))
        if len(frame) != expected_rows[name]:
            raise ValueError(f'{name} row count mismatch')
        frames[name] = frame
    test_path = processed_dir / 'test_clean.csv'
    manifest_path = processed_dir / 'manifest.json'
    expected_sha = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        expected_sha = manifest['outputs']['test']['sha256']
    test_integrity = check_locked_test(test_path, expected_rows['test'], expected_sha)
    return frames['train'], frames['validation'], test_integrity

def _flatten(spec, feature_params, model, feature_seconds, fit_seconds,
             predict_seconds, dimensions, train_rows, validation_rows, metrics):
    record = {
        'experiment_id': spec.id, 'feature': spec.feature, 'feature_params': feature_params,
        'model': spec.model, 'model_params': model.get_params(deep=False),
        'random_state': RANDOM_STATE, 'train_rows': train_rows,
        'validation_rows': validation_rows, 'feature_dimensions': dimensions,
        'feature_fit_seconds': round(feature_seconds, 4),
        'fit_seconds': round(fit_seconds, 4),
        'prediction_seconds': round(predict_seconds, 4),
    }
    for key in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1'):
        record[f'validation_{key}'] = metrics[key]
    for label in LABELS:
        for key in ('precision', 'recall', 'f1'):
            record[f'{label}_{key}'] = metrics['per_class'][label][key]
    return record

def _top_linear_features(extractor, model, top_n=12):
    if not hasattr(model, 'coef_'):
        return None
    names = extractor.get_feature_names_out()
    result = {}
    for i, label in enumerate(model.classes_):
        weights = model.coef_[i]
        indices = np.argsort(weights)[-top_n:][::-1]
        result[str(label)] = [{'feature': str(names[j]), 'weight': float(weights[j])}
                              for j in indices]
    return result

def _write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n',
                    encoding='utf-8')

def _comparison_rows(records):
    keys = ['experiment_id', 'feature', 'model', 'feature_dimensions',
            'fit_seconds', 'prediction_seconds', 'validation_accuracy',
            'validation_macro_precision', 'validation_macro_recall',
            'validation_macro_f1', 'validation_weighted_f1']
    for label in LABELS:
        keys.extend(f'{label}_{metric}' for metric in ('precision', 'recall', 'f1'))
    return pd.DataFrame(records).loc[:, keys]

def _plot_results(records, diagnostics, figures_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda r: r['validation_macro_f1'])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([r['experiment_id'] + ' ' + r['feature'] + '/' + r['model'] for r in ordered],
            [r['validation_macro_f1'] for r in ordered], color='#3979a8')
    ax.set(xlabel='VALIDATION macro-F1', title='Controlled TF-IDF model comparison')
    fig.tight_layout()
    fig.savefig(figures_dir / 'validation_macro_f1.png', dpi=150)
    plt.close(fig)
    winners = {}
    for model in ('logistic_regression', 'linear_svm', 'random_forest'):
        candidates = [r for r in records if r['model'] == model]
        if candidates:
            winners[model] = select_winner(candidates)
    overall = select_winner(records)
    winners['overall'] = overall
    seen = set()
    for label, record in winners.items():
        if record['experiment_id'] in seen:
            continue
        seen.add(record['experiment_id'])
        cm = np.asarray(diagnostics[record['experiment_id']]['confusion_matrix'])
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        image = ax.imshow(cm, cmap='Blues')
        ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS,
               yticklabels=LABELS, xlabel='Predicted', ylabel='True',
               title=f"VALIDATION confusion matrix: {record['experiment_id']}")
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, str(value), ha='center', va='center',
                    color='white' if value > cm.max()/2 else 'black')
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(figures_dir / f"validation_confusion_{record['experiment_id']}.png", dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = list(winners)
    x = np.arange(len(labels))
    for i, sentiment in enumerate(LABELS):
        vals = [diagnostics[winners[label]['experiment_id']]['per_class'][sentiment]['f1']
                for label in labels]
        ax.bar(x + (i-1)*.24, vals, width=.24, label=sentiment)
    ax.set(xticks=x, xticklabels=labels, ylabel='VALIDATION F1', ylim=(0, 1),
           title='Per-class F1 of leading candidates')
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / 'validation_per_class_f1.png', dpi=150)
    plt.close(fig)

def _error_analysis(validation, predictions, metrics):
    true = validation['sentiment'].to_numpy()
    text = validation['clean_text'].to_numpy()
    lines = ['# VALIDATION error analysis', '',
             'Errors come from the validation-selected candidate only. Samples are ordered, small and illustrative, not representative estimates.',
             'TEST is locked. No TEST predictions or performance were generated.', '',
             '## Confusion counts', '', 'Rows are true labels; columns are predictions in negative, neutral, positive order.', '',
             str(metrics['confusion_matrix']), '']
    mismatches = true != predictions
    lines += [f"Total errors: {int(mismatches.sum())} of {len(true)} validation rows.", '']
    for actual in LABELS:
        for predicted in LABELS:
            if actual == predicted:
                continue
            indices = np.flatnonzero((true == actual) & (predictions == predicted))
            lines += [f'## {actual} to {predicted}: {len(indices)}', '']
            for idx in indices[:2]:
                excerpt = str(text[idx]).replace('|', '/').replace('\n', ' ')
                lines.append(f'- Row {int(idx)}: {excerpt[:220]}')
            lines.append('')
    errors = [str(value) for value in text[mismatches]]
    negation = sum(any(word in value.split() for word in ("not", "no", "never", "don't", "can't", "won't")) for value in errors)
    punctuation = sum(('!' in value or '?' in value) for value in errors)
    short = sum(len(value.split()) <= 5 for value in errors)
    lines += ['## Descriptive cues in misclassified validation text', '',
              f'- Explicit negation token: {negation} errors.',
              f'- Exclamation or question mark: {punctuation} errors.',
              f'- Five words or fewer: {short} errors.', '',
              'Illustrative reading: negative-to-neutral examples can express dissatisfaction through price or access details rather than a simple negative adjective. Positive-to-neutral examples can depend on future expectations or event context. Neutral-to-positive examples may contain named events without an explicit opinion. These are hypotheses from a few displayed posts, not measured error causes.',
              'These counts do not establish causes. Sarcasm, mixed sentiment, named entities, tense and missing conversation context require manual review. The displayed examples should guide focused future hypotheses, then be tested on validation without unlocking TEST.', '']
    return '\n'.join(lines)

def _summary(records, diagnostics, test_integrity, winner, train_counts, synthetic=False):
    lines = ['# Classical sentiment modeling: executed VALIDATION summary', '',
             'Objective: compare sparse TF-IDF representations and classical classifiers using official TweetEval TRAIN and VALIDATION.',
             'TRAIN fits vocabulary, IDF and classifiers. VALIDATION selects by macro-F1. TEST remains locked: only existence, schema, row count, split and checksum were checked. No TEST metrics or predictions exist.',
             '', '## Data and imbalance', '',
             f"TRAIN: {records[0]['train_rows']}; VALIDATION: {records[0]['validation_rows']}; locked TEST rows: {test_integrity['rows']}.",
             f"TRAIN class counts: negative {train_counts['negative']}; neutral {train_counts['neutral']}; positive {train_counts['positive']}. No oversampling or resplitting.",
             '', '## Features and models', '',
             'TF-IDF multiplies term frequency within a document by inverse document frequency, reducing the influence of very common terms. In not good, word unigrams are not and good; the bigram not good retains their relation.',
             "Word features capture vocabulary and word pairs, using a contraction-preserving token pattern. Character n-grams (3–5) can capture n't, !!!, repeated letters and fragments of noisy words. Combined features concatenate word (1,2) and character (3,5) sparse matrices. Character benefits are judged by VALIDATION results below.",
             'All representations use min_df=2, max_df=.95, sublinear_tf=True, L2 normalization and no additional lowercasing. Word max_features=100,000; character max_features=120,000. No exhaustive search or class weighting.',
             'Models: Logistic Regression (C=1, lbfgs, max_iter=350), LinearSVC (C=1, max_iter=3000), MultinomialNB (alpha=1), RandomForest (80 trees, depth 24, leaf minimum 2, sqrt features, 2 jobs). Random seed 42 where applicable.',
             '', '## Validation comparison', '',
             '| ID | Features | Model | Accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 | Fit seconds |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted(records, key=lambda r: -r['validation_macro_f1']):
        lines.append('| ' + ' | '.join([r['experiment_id'], r['feature'], r['model']] +
            [f"{r['validation_'+metric]:.4f}" for metric in ('accuracy','macro_precision','macro_recall','macro_f1','weighted_f1')] +
            [f"{r['fit_seconds']:.2f}"]) + ' |')
    lines += ['', '## Leading models', '']
    for model in ('logistic_regression', 'linear_svm', 'random_forest', 'naive_bayes'):
        r = select_winner([item for item in records if item['model'] == model])
        per = diagnostics[r['experiment_id']]['per_class']
        lines.append(f"- {model}: {r['experiment_id']} {r['feature']}; macro-F1 {r['validation_macro_f1']:.4f}; accuracy {r['validation_accuracy']:.4f}; negative/neutral/positive F1 {per['negative']['f1']:.4f}/{per['neutral']['f1']:.4f}/{per['positive']['f1']:.4f}; fit {r['fit_seconds']:.2f}s.")
    cm = diagnostics[winner['experiment_id']]['confusion_matrix']
    lines += ['', '## Selected candidate and errors', '',
              f"VALIDATION-SELECTED CANDIDATE: {winner['experiment_id']} {winner['feature']} + {winner['model']}. Macro-F1 {winner['validation_macro_f1']:.4f}; accuracy {winner['validation_accuracy']:.4f}. THIS IS NOT YET THE FINAL TEST-EVALUATED MODEL.",
              f'Winner VALIDATION confusion rows (negative, neutral, positive): {cm}. Off-diagonal cells count misclassifications; see error_analysis.md for controlled examples and cue counts.',
              'The top learned linear coefficients in model_metrics.json are TRAIN-fitted feature associations, not causal explanations. Positive coefficients favor the corresponding class relative to alternatives.',
              'Random Forest used all TRAIN rows with bounded depth and 80 trees. See its negative recall and fit time above; the constrained configuration is not a full-capacity forest.',
              '', '## Limits and next work', '',
              'Historical English tweets, class imbalance, sarcasm, context loss and benchmark annotation ambiguity remain. Lowercased clean text loses case cues. Default word vectorization omits emoji, while character features may represent them indirectly. No brand-specific or prospective evaluation is claimed.',
              'These are corrected-preprocessing baselines. Focused TRAIN-only CV and the frozen development candidate are reported in tuning_summary.md. TEST remains locked for a separate final methodology review and one future evaluation.',
              'Full per-class precision/recall/F1, confusion matrices, feature/model parameters, versions and times are in model_metrics.json. Model artifact regeneration: python -m src.model_pipeline.', '']
    if synthetic:
        lines.insert(1, 'WARNING: Synthetic fixture results are for pipeline testing only, NOT FOR MODEL EVALUATION.')
    return '\n'.join(lines)

def run_modeling(root: Path = PROJECT_ROOT, experiments=EXPERIMENTS,
                 expected_rows: dict = EXPECTED_ROWS) -> dict:
    """Execute controlled full-TRAIN experiments without passing TEST labels anywhere."""
    root = Path(root)
    train, validation, test_integrity = load_development_data(root, expected_rows)
    reports = root / 'reports'
    figures = reports / 'figures' / 'modeling'
    model_dir = root / 'models' / 'candidates'
    reports.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    records, diagnostics, interpretations = [], {}, {}
    best_bundle = None
    train_text = train['clean_text'].tolist()
    validation_text = validation['clean_text'].tolist()
    train_labels = train['sentiment'].to_numpy()
    train_counts = {label: int(train['sentiment'].eq(label).sum()) for label in LABELS}
    validation_labels = validation['sentiment'].to_numpy()
    feature_order = list(dict.fromkeys(spec.feature for spec in experiments))
    for feature_name in feature_order:
        extractor = make_features(feature_name)
        start = perf_counter()
        x_train, x_validation = fit_train_transform_validation(
            extractor, train_text, validation_text)
        feature_seconds = perf_counter() - start
        feature_params = feature_parameters(extractor)
        logger.info('Features %s: %d columns, %.1fs', feature_name,
                    x_train.shape[1], feature_seconds)
        for spec in (item for item in experiments if item.feature == feature_name):
            model = make_model(spec.model)
            start = perf_counter()
            model.fit(x_train, train_labels)
            fit_seconds = perf_counter() - start
            start = perf_counter()
            predictions = model.predict(x_validation)
            predict_seconds = perf_counter() - start
            metrics = evaluate_validation(validation_labels, predictions)
            record = _flatten(spec, feature_params, model, feature_seconds,
                fit_seconds, predict_seconds, x_train.shape[1], len(train),
                len(validation), metrics)
            records.append(record)
            diagnostics[spec.id] = metrics
            interpretation = _top_linear_features(extractor, model)
            if interpretation is not None:
                interpretations[spec.id] = interpretation
            if select_winner(records)['experiment_id'] == spec.id:
                best_bundle = (spec, Pipeline([('features', extractor),
                                               ('classifier', model)]),
                               predictions.copy())
            logger.info('%s %s/%s macro-F1 %.4f accuracy %.4f fit %.1fs',
                spec.id, feature_name, spec.model, metrics['macro_f1'],
                metrics['accuracy'], fit_seconds)
        del x_train, x_validation
    winner = select_winner(records)
    spec, candidate, original_predictions = best_bundle
    assert spec.id == winner['experiment_id']
    artifact = model_dir / 'best_candidate.joblib'
    temporary = model_dir / 'best_candidate.joblib.tmp'
    joblib.dump(candidate, temporary, compress=3)
    temporary.replace(artifact)
    loaded = joblib.load(artifact)
    if not np.array_equal(original_predictions, loaded.predict(validation_text)):
        raise AssertionError('Reloaded validation predictions differ')
    metadata = {
        'status': 'VALIDATION-SELECTED CANDIDATE — NOT FINAL TEST-EVALUATED MODEL',
        'experiment_id': winner['experiment_id'],
        'feature': winner['feature'], 'model': winner['model'],
        'validation_macro_f1': winner['validation_macro_f1'],
        'validation_accuracy': winner['validation_accuracy'],
        'artifact_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
        'artifact_bytes': artifact.stat().st_size,
        'reload_predictions_match': True}
    _write_json(model_dir / 'best_candidate_metadata.json', metadata)
    comparison = _comparison_rows(records).sort_values(
        ['validation_macro_f1','validation_accuracy','experiment_id'],
        ascending=[False, False, True])
    comparison.to_csv(reports / 'model_comparison.csv', index=False)
    result = {
        'selection_metric': 'validation_macro_f1',
        'test_integrity_only': test_integrity, 'train_class_counts': train_counts,
        'experiments': records, 'diagnostics': diagnostics,
        'linear_features': interpretations, 'winner': winner,
        'candidate': metadata,
        'environment': {'python': platform.python_version(),
            'packages': {name: importlib.metadata.version(name) for name in
                ('pandas','numpy','scipy','scikit-learn','joblib','matplotlib')}}}
    _write_json(reports / 'model_metrics.json', result)
    _plot_results(records, diagnostics, figures)
    (reports / 'error_analysis.md').write_text(
        _error_analysis(validation, original_predictions,
                        diagnostics[winner['experiment_id']]), encoding='utf-8')
    (reports / 'model_summary.md').write_text(
        _summary(records, diagnostics, test_integrity, winner, train_counts,
                 synthetic=expected_rows != EXPECTED_ROWS), encoding='utf-8')
    logger.info('Selected %s, VALIDATION macro-F1 %.4f; TEST locked',
                spec.id, winner['validation_macro_f1'])
    return result

if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    logging.basicConfig(level=logging.INFO,
        format='%(levelname)s %(name)s: %(message)s')
    run_modeling()

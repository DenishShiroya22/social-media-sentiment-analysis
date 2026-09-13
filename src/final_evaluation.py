"""One-time, precommitted evaluation of the frozen sentiment procedure.

Normal invocation is read-only. The explicit unlock command must run only after
its procedure commit is pushed and GitHub CI passes.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import EXPECTED_ROWS, LABELS, PROJECT_ROOT, RANDOM_STATE
from .data_loader import load_csv, save_csv
from .evaluation import evaluate_validation
from .features import feature_parameters, make_features
from .model_pipeline import check_locked_test, load_development_data
from .models import make_model
from .preprocessing import DEFAULT_CONFIG

PROTOCOL_VERSION = '1.0'
METRICS_NAME = 'final_test_metrics.json'
MANIFEST_NAME = 'final_evaluation_manifest.json'
ATTEMPT_NAME = '.final_evaluation_attempt.json'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def canonical_features(features) -> dict:
    """Convert tuple ranges to the JSON form stored by frozen metadata."""
    return json.loads(json.dumps(feature_parameters(features), ensure_ascii=False))


def load_frozen_metadata(root: Path = PROJECT_ROOT, *, verify_artifact: bool = True,
                         strict_runtime: bool = True) -> dict:
    root = Path(root)
    path = root / 'models' / 'candidates' / 'frozen_candidate_metadata.json'
    metadata = json.loads(path.read_text(encoding='utf-8'))
    if metadata['selected_model'] != 'logistic_regression':
        raise ValueError('Frozen selection is not Logistic Regression')
    if metadata['random_state'] != RANDOM_STATE:
        raise ValueError('Frozen random state changed')
    if metadata['preprocessing'] != asdict(DEFAULT_CONFIG):
        raise ValueError('Current preprocessing disagrees with frozen metadata')
    if metadata['selection_metric'] != 'validation macro-F1 after TRAIN-only CV':
        raise ValueError('Unexpected frozen selection metric')
    if strict_runtime and importlib.metadata.version('scikit-learn') != metadata['environment']['scikit-learn']:
        raise RuntimeError('scikit-learn version differs from frozen artifact environment')
    if verify_artifact:
        artifact = root / 'models' / 'candidates' / 'frozen_candidate.joblib'
        if artifact.stat().st_size != metadata['artifact_bytes']:
            raise ValueError('Frozen candidate size mismatch')
        if sha256_file(artifact) != metadata['artifact_sha256']:
            raise ValueError('Frozen candidate checksum mismatch')
    return metadata


def make_frozen_pipeline(metadata: dict) -> Pipeline:
    """Construct a new, unfitted Pipeline from the frozen configuration."""
    if metadata['selected_model'] != 'logistic_regression':
        raise ValueError('Only frozen Logistic Regression is permitted')
    features = make_features('combined')
    feature_lookup = dict(features.transformer_list)
    if set(metadata['features']) != {'word', 'character'}:
        raise ValueError('Frozen feature branches changed')
    for name in ('word', 'character'):
        vectorizer = feature_lookup[name]
        parameters = metadata['features'][name].copy()
        parameters['ngram_range'] = tuple(parameters['ngram_range'])
        unknown = set(parameters) - set(vectorizer.get_params(deep=False))
        if unknown:
            raise ValueError(f'Unknown frozen {name} parameters: {sorted(unknown)}')
        vectorizer.set_params(**parameters)
    weights = metadata['feature_weights']
    if set(weights) != {'word', 'character'}:
        raise ValueError('Frozen feature weights changed')
    features.set_params(transformer_weights=weights.copy())
    classifier = make_model('logistic_regression')
    parameters = metadata['classifier_parameters']
    unknown = set(parameters) - set(classifier.get_params(deep=False))
    if unknown:
        raise ValueError(f'Unknown frozen classifier parameters: {sorted(unknown)}')
    classifier.set_params(**parameters)
    pipeline = Pipeline([('features', features), ('classifier', classifier)])
    if canonical_features(features) != metadata['features']:
        raise ValueError('Constructed TF-IDF differs from frozen metadata')
    if features.transformer_weights != weights:
        raise ValueError('Constructed feature weights differ from frozen metadata')
    if classifier.get_params(deep=False) != parameters:
        raise ValueError('Constructed classifier differs from frozen metadata')
    return pipeline


def verify_frozen_artifact(root: Path, metadata: dict, fresh: Pipeline) -> None:
    """Compare the original fitted candidate's configuration, never its predictions."""
    original = joblib.load(root / 'models' / 'candidates' / 'frozen_candidate.joblib')
    if not isinstance(original, Pipeline) or list(original.named_steps) != ['features', 'classifier']:
        raise ValueError('Frozen artifact is not the expected Pipeline')
    if canonical_features(original.named_steps['features']) != metadata['features']:
        raise ValueError('Frozen artifact features disagree with metadata')
    if original.named_steps['features'].transformer_weights != metadata['feature_weights']:
        raise ValueError('Frozen artifact weights disagree with metadata')
    if original.named_steps['classifier'].get_params(deep=False) != metadata['classifier_parameters']:
        raise ValueError('Frozen artifact classifier disagrees with metadata')
    old_features = dict(original.named_steps['features'].transformer_list)
    new_features = dict(fresh.named_steps['features'].transformer_list)
    for name in ('word', 'character'):
        if old_features[name].get_params(deep=False) != new_features[name].get_params(deep=False):
            raise ValueError(f'Fresh {name} vectorizer differs from frozen artifact')
    if original.named_steps['classifier'].get_params(deep=False) != fresh.named_steps['classifier'].get_params(deep=False):
        raise ValueError('Fresh classifier differs from frozen artifact')


@dataclass
class PreparedFinal:
    pipeline: Pipeline
    metadata: dict
    train_rows: int
    validation_rows: int
    development_rows: int
    test_integrity: dict
    fit_seconds: float


def prepare_final_model(root: Path = PROJECT_ROOT, *,
                        expected_rows: dict = EXPECTED_ROWS,
                        verify_artifact: bool = True,
                        strict_runtime: bool = True) -> PreparedFinal:
    """Fit only TRAIN+VALIDATION; TEST receives mechanical checks, not predictions."""
    root = Path(root)
    metadata = load_frozen_metadata(root, verify_artifact=verify_artifact,
                                    strict_runtime=strict_runtime)
    pipeline = make_frozen_pipeline(metadata)
    if verify_artifact:
        verify_frozen_artifact(root, metadata, pipeline)
    train, validation, test_integrity = load_development_data(root, expected_rows)
    if not train['split'].eq('train').all() or not validation['split'].eq('validation').all():
        raise ValueError('Official development split membership changed')
    development = pd.concat((train, validation), ignore_index=True)
    if len(development) != expected_rows['train'] + expected_rows['validation']:
        raise ValueError('Development row count changed')
    if development['clean_text'].isna().any() or development['clean_text'].eq('').any():
        raise ValueError('Blank development text')
    start = perf_counter()
    pipeline.fit(development['clean_text'].tolist(), development['sentiment'].to_numpy())
    elapsed = perf_counter() - start
    return PreparedFinal(pipeline, metadata, len(train), len(validation),
                         len(development), test_integrity, elapsed)


def assert_evaluation_available(root: Path) -> None:
    reports = Path(root) / 'reports'
    for name in (METRICS_NAME, MANIFEST_NAME, ATTEMPT_NAME):
        if (reports / name).exists():
            raise FileExistsError(f'Official evaluation already exists or was attempted: {name}')


def _test_metrics(y_true, prediction) -> dict:
    """Schema shared by synthetic dry runs and the one official evaluation."""
    result = evaluate_validation(y_true, prediction)
    for label in LABELS:
        result['per_class'][label]['support'] = int(result['classification_report'][label]['support'])
    return result


def _plot_confusion(metrics: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matrix = np.asarray(metrics['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(5.6, 5))
    image = ax.imshow(matrix, cmap='Blues')
    ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS,
           xlabel='Predicted', ylabel='Actual', title='One official TEST confusion matrix')
    for (i, j), count in np.ndenumerate(matrix):
        ax.text(j, i, str(int(count)), ha='center', va='center',
                color='white' if count > matrix.max() / 2 else 'black')
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _summary(payload: dict) -> str:
    m = payload['test_metrics']
    v = payload['frozen_validation_metrics']
    lines = [
        '# One official locked-TEST sentiment evaluation', '',
        f"Pre-TEST procedure commit: {payload['pretest_commit_sha']}. Protocol version: {PROTOCOL_VERSION}.",
        'The frozen combined TF-IDF plus Logistic Regression procedure was fitted once on TRAIN+VALIDATION, then evaluated once on untouched TEST. No competing model or TEST-driven tuning was performed.', '',
        f"Fit rows: TRAIN {payload['train_rows']:,} + VALIDATION {payload['validation_rows']:,} = {payload['development_rows']:,}. TEST rows: {payload['test_rows']:,}.",
        f"Final fitting took {payload['fit_seconds']:.2f} seconds; one official prediction took {payload['predict_seconds']:.2f} seconds.", '',
        '## Official TEST metrics', '',
        f"Accuracy {m['accuracy']:.4f}; macro precision {m['macro_precision']:.4f}; macro recall {m['macro_recall']:.4f}; **macro-F1 {m['macro_f1']:.4f}**; weighted F1 {m['weighted_f1']:.4f}.", '',
        '| Class | Precision | Recall | F1 | Support |',
        '|---|---:|---:|---:|---:|',
    ]
    for label in LABELS:
        c = m['per_class'][label]
        lines.append(f"| {label} | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1']:.4f} | {c['support']:,} |")
    lines += ['', 'Confusion rows are actual and columns predicted, ordered negative, neutral, positive:', '',
              str(m['confusion_matrix']), '', '## Descriptive VALIDATION versus TEST', '',
              '| Metric | VALIDATION | TEST | TEST − VALIDATION |', '|---|---:|---:|---:|']
    entries = [
        ('Accuracy', 'accuracy', None),
        ('Macro-F1', 'macro_f1', None),
        ('Negative F1', 'f1', 'negative'),
        ('Neutral F1', 'f1', 'neutral'),
        ('Positive F1', 'f1', 'positive'),
    ]
    for title, key, label in entries:
        val = v[key] if label is None else v['per_class'][label][key]
        test = m[key] if label is None else m['per_class'][label][key]
        lines.append(f'| {title} | {val:.4f} | {test:.4f} | {test-val:+.4f} |')
    lines += ['', 'The VALIDATION score belongs to the development candidate fitted on TRAIN only. The TEST score belongs to the same frozen configuration refitted on TRAIN+VALIDATION. This gap is descriptive, not a basis for changing the model.', '',
              '## Artifact and safeguards', '',
              f"Final fitted artifact: models/final/final_model.joblib; {payload['artifact_bytes']:,} bytes; SHA-256 {payload['artifact_sha256']}. Reloaded predictions matched the original vector exactly. The second prediction call is solely an artifact-integrity comparison; no second metric set was calculated.",
              'The label-only prediction file is kept locally under data/processed and excluded from Git. No raw TEST tweet examples appear in this report.',
              'An evaluation manifest blocks accidental reruns. The recorded pre-TEST commit was pushed and CI-passing before TEST was unlocked.', '',
              '## Limits and next work', '',
              'TweetEval contains historical English tweets. Sarcasm, missing context, annotation ambiguity, class imbalance and modern brand-domain shift limit real-world use. This benchmark result is not a guarantee for current brand comments or psychological inference. No post-TEST parameter, preprocessing or model changes were made. Next: build a reusable inference service, then evaluate responsibly on real public social-media data and design aggregate analytics.', '']
    return '\n'.join(lines)


def _model_card(payload: dict) -> str:
    m = payload['test_metrics']
    return '\n'.join([
        '# Final TweetEval sentiment model card', '',
        '## Intended use', '',
        'A research baseline for three-class sentiment on TweetEval-style English posts and a starting point for later brand-monitoring evaluation. It is not a substitute for human moderation and is not suitable for psychological inference.', '',
        '## Dataset and training', '',
        'Cardiff NLP TweetEval sentiment, pinned source revision recorded in data/raw/manifest.json. Official TRAIN (45,615) and VALIDATION (2,000) rows were combined after model selection, giving 47,615 final fitting rows. No deduplication, resplitting or resampling. One official TEST evaluation used 12,284 untouched rows.', '',
        '## Frozen pipeline', '',
        'Deterministic preprocessing with targeted escaped-punctuation normalization; word (1,2) and char_wb (3,5) TF-IDF; word weight 1.0 and character weight 1.25; balanced Logistic Regression (C=1, lbfgs, max_iter=350, seed 42). Exact parameters and environment are in final_model_metadata.json.', '',
        '## One official TEST result', '',
        f"Accuracy {m['accuracy']:.4f}; macro precision {m['macro_precision']:.4f}; macro recall {m['macro_recall']:.4f}; macro-F1 {m['macro_f1']:.4f}; weighted F1 {m['weighted_f1']:.4f}.", '',
        '| Class | Precision | Recall | F1 | Support |', '|---|---:|---:|---:|---:|',
        *[f"| {label} | {m['per_class'][label]['precision']:.4f} | {m['per_class'][label]['recall']:.4f} | {m['per_class'][label]['f1']:.4f} | {m['per_class'][label]['support']:,} |" for label in LABELS],
        '', '## Limits and ethics', '',
        'Historical tweets may not transfer to modern, brand-specific comments. Sarcasm, short or missing context, mixed sentiment, annotation ambiguity and class imbalance remain. Do not treat labels as objective psychological facts. Future collection should respect platform terms and avoid unnecessary personal information. Raw/processed dataset CSVs and row-level TEST predictions are not published. Check upstream dataset rights before redistribution.', '',
        f"Pre-TEST procedure commit: {payload['pretest_commit_sha']}. No tuning followed the TEST result.", ''
    ])


def evaluate_locked_test(prepared: PreparedFinal, root: Path, *,
                         expected_rows: dict, pretest_commit_sha: str) -> dict:
    """The only function that unlocks TEST labels and makes official predictions.

    An exclusive attempt marker remains after any failure, preventing silent retries.
    """
    root = Path(root)
    if not pretest_commit_sha:
        raise ValueError('Pre-TEST commit SHA is required')
    assert_evaluation_available(root)
    reports = root / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    attempt = reports / ATTEMPT_NAME
    with attempt.open('x', encoding='utf-8') as handle:
        json.dump({'protocol_version': PROTOCOL_VERSION,
                   'pretest_commit_sha': pretest_commit_sha,
                   'status': 'attempt started; do not retry after failure without audit'}, handle)
    # Only technical integrity checks precede the one official TEST access.
    test_path = root / 'data' / 'processed' / 'test_clean.csv'
    manifest = json.loads((root / 'data' / 'processed' / 'manifest.json').read_text(encoding='utf-8'))
    test_sha = manifest['outputs']['test']['sha256']
    check_locked_test(test_path, expected_rows['test'], test_sha)
    test = load_csv(test_path)
    if len(test) != expected_rows['test'] or not test['split'].eq('test').all():
        raise ValueError('TEST row count or membership changed')
    if not test['sentiment'].isin(LABELS).all():
        raise ValueError('Unexpected TEST label')
    if test['clean_text'].isna().any() or test['clean_text'].eq('').any():
        raise ValueError('Blank TEST clean_text')
    test_text = test['clean_text'].tolist()
    test_y = test['sentiment'].to_numpy()
    start = perf_counter()
    prediction = prepared.pipeline.predict(test_text)
    predict_seconds = perf_counter() - start
    metrics = _test_metrics(test_y, prediction)
    artifact = root / 'models' / 'final' / 'final_model.joblib'
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(prepared.pipeline, artifact, compress=3)
    artifact_sha = sha256_file(artifact)
    reloaded = joblib.load(artifact)
    # Technical persistence check only: never score this second identical vector.
    reload_match = bool(np.array_equal(reloaded.predict(test_text), prediction))
    if not reload_match:
        raise AssertionError('Persisted model changed TEST predictions')
    local_predictions = root / 'data' / 'processed' / 'final_test_predictions.csv'
    save_csv(pd.DataFrame({
        'row_index': np.arange(len(test), dtype=int),
        'actual_sentiment': test_y,
        'predicted_sentiment': prediction,
        'correct': test_y == prediction,
    }), local_predictions)
    payload = {
        'status': 'ONE OFFICIAL LOCKED-TEST EVALUATION COMPLETE',
        'protocol_version': PROTOCOL_VERSION,
        'pretest_commit_sha': pretest_commit_sha,
        'frozen_artifact_sha256': prepared.metadata['artifact_sha256'],
        'frozen_validation_metrics': prepared.metadata['validation_metrics'],
        'train_rows': prepared.train_rows, 'validation_rows': prepared.validation_rows,
        'development_rows': prepared.development_rows, 'test_rows': len(test),
        'processed_test_sha256': test_sha,
        'test_metrics': metrics,
        'fit_seconds': prepared.fit_seconds, 'predict_seconds': predict_seconds,
        'artifact_path': 'models/final/final_model.joblib',
        'artifact_bytes': artifact.stat().st_size, 'artifact_sha256': artifact_sha,
        'reload_predictions_match': reload_match,
        'prediction_file_policy': 'local ignored labels/index only; raw TEST text not published',
        'local_prediction_file_sha256': sha256_file(local_predictions),
        'environment': {key: importlib.metadata.version(key) for key in
                        ('pandas', 'numpy', 'scipy', 'scikit-learn', 'joblib')},
        'python_version': platform.python_version(),
    }
    metrics_path = reports / METRICS_NAME
    write_json(metrics_path, payload)
    summary_path = reports / 'final_evaluation_summary.md'
    summary_path.write_text(_summary(payload), encoding='utf-8')
    figure_path = reports / 'figures' / 'final_test_confusion_matrix.png'
    _plot_confusion(metrics, figure_path)
    model_metadata_path = root / 'models' / 'final' / 'final_model_metadata.json'
    write_json(model_metadata_path, {
        'pretest_commit_sha': pretest_commit_sha,
        'protocol_version': PROTOCOL_VERSION,
        'training_rows': prepared.development_rows,
        'frozen_configuration': {key: prepared.metadata[key] for key in
                                 ('preprocessing', 'features', 'feature_weights',
                                  'classifier_parameters', 'random_state')},
        'test_metrics': metrics,
        'artifact_sha256': artifact_sha, 'artifact_bytes': artifact.stat().st_size,
        'reload_predictions_match': reload_match,
        'environment': payload['environment'], 'python_version': payload['python_version'],
    })
    card_path = root / 'models' / 'final' / 'MODEL_CARD.md'
    card_path.write_text(_model_card(payload), encoding='utf-8')
    manifest_path = reports / MANIFEST_NAME
    write_json(manifest_path, {
        'evaluation_completed': True,
        'protocol_version': PROTOCOL_VERSION,
        'pretest_commit_sha': pretest_commit_sha,
        'frozen_artifact_sha256': prepared.metadata['artifact_sha256'],
        'final_model_sha256': artifact_sha,
        'final_metrics_sha256': sha256_file(metrics_path),
        'processed_test_sha256': test_sha,
        'test_rows': len(test),
        'official_prediction_vectors': 1,
        'reload_check': 'same fitted model and inputs, exact equality only; not separately scored',
        'local_prediction_file_sha256': sha256_file(local_predictions),
    })
    attempt.unlink()
    return payload


def _git(root: Path, *arguments: str) -> str:
    output = subprocess.check_output(['git', '-c', f'safe.directory={root.as_posix()}',
                                      *arguments], cwd=root)
    return output.decode().strip()


def _verify_pretest_checkout(root: Path, sha: str) -> None:
    if len(sha) != 40 or any(c not in '0123456789abcdef' for c in sha.lower()):
        raise ValueError('Full 40-character pre-TEST SHA required')
    if _git(root, 'rev-parse', 'HEAD') != sha:
        raise ValueError('Working HEAD differs from pre-TEST commit')
    if _git(root, 'rev-parse', 'origin/main') != sha:
        raise ValueError('Pre-TEST commit is not synchronized with origin/main')
    if _git(root, 'status', '--porcelain'):
        raise ValueError('Pre-TEST working tree is not clean')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate-locked-test', action='store_true',
                        help='Explicit one-time official unlock after pushed, CI-passing pre-TEST commit')
    parser.add_argument('--pretest-sha',
                        help='Full SHA of the pushed, CI-passing frozen procedure commit')
    args = parser.parse_args(argv)
    if not args.evaluate_locked_test:
        if args.pretest_sha:
            parser.error('--pretest-sha requires --evaluate-locked-test')
        manifest = PROJECT_ROOT / 'reports' / MANIFEST_NAME
        if manifest.exists():
            print(f'Official result already recorded: {manifest}')
        else:
            print('Official TEST remains locked. Explicit unlock requires a pushed, CI-passing procedure commit.')
        return 0
    if not args.pretest_sha:
        parser.error('--evaluate-locked-test requires --pretest-sha')
    assert_evaluation_available(PROJECT_ROOT)
    _verify_pretest_checkout(PROJECT_ROOT, args.pretest_sha)
    prepared = prepare_final_model(PROJECT_ROOT)
    payload = evaluate_locked_test(prepared, PROJECT_ROOT,
                                   expected_rows=EXPECTED_ROWS,
                                   pretest_commit_sha=args.pretest_sha)
    print(f"Official TEST macro-F1: {payload['test_metrics']['macro_f1']:.4f}")
    print(f"Official result: {PROJECT_ROOT / 'reports' / METRICS_NAME}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

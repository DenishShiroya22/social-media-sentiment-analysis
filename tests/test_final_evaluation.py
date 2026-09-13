"""Synthetic-only checks for the one-time final evaluation procedure."""
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import LABELS, PROJECT_ROOT, RANDOM_STATE
from src.data_loader import save_csv
from src.features import make_features
from src.models import make_model
from src.preprocessing import DEFAULT_CONFIG
from src.final_evaluation import (
    PROTOCOL_VERSION, assert_evaluation_available, canonical_features,
    evaluate_locked_test, load_frozen_metadata, make_frozen_pipeline,
    prepare_final_model, sha256_file, _test_metrics,
)


def synthetic_repository(root: Path, *, invalid_test_label=False):
    processed = root / 'data' / 'processed'
    processed.mkdir(parents=True)
    train_text = [
        'bad service today', 'awful service today', 'terrible bad day',
        'ordinary news today', 'normal update today', 'regular news update',
        'great service today', 'lovely good day', 'wonderful great day',
    ]
    validation_text = ['bad awful service', 'ordinary normal news', 'great good service']
    test_text = ['terrible awful day', 'regular update today', 'lovely wonderful day']
    labels = list(LABELS) * 3
    train = pd.DataFrame({'text': train_text, 'clean_text': train_text,
                          'sentiment': labels, 'split': ['train'] * 9})
    validation = pd.DataFrame({'text': validation_text, 'clean_text': validation_text,
                               'sentiment': list(LABELS), 'split': ['validation'] * 3})
    test = pd.DataFrame({'text': test_text, 'clean_text': test_text,
                         'sentiment': ['INVALID LABEL' if invalid_test_label else LABELS[0],
                                       LABELS[1], LABELS[2]],
                         'split': ['test'] * 3})
    for name, frame in [('train', train), ('validation', validation), ('test', test)]:
        save_csv(frame, processed / f'{name}_clean.csv')
    write_manifest = {'outputs': {'test': {'sha256': sha256_file(processed / 'test_clean.csv')}}}
    (processed / 'manifest.json').write_text(json.dumps(write_manifest), encoding='utf-8')
    features = make_features('combined')
    features.set_params(word__min_df=1, character__min_df=1,
                        transformer_weights={'word': 1.0, 'character': 1.25})
    classifier = make_model('logistic_regression')
    classifier.set_params(C=1.0, class_weight='balanced')
    metadata = {
        'selected_model': 'logistic_regression',
        'random_state': RANDOM_STATE,
        'preprocessing': asdict(DEFAULT_CONFIG),
        'selection_metric': 'validation macro-F1 after TRAIN-only CV',
        'features': canonical_features(features),
        'feature_weights': features.transformer_weights,
        'classifier_parameters': classifier.get_params(deep=False),
        'environment': {'scikit-learn': importlib.metadata.version('scikit-learn')},
        'artifact_sha256': 'synthetic-only',
        'artifact_bytes': 0,
        'validation_metrics': {'accuracy': 0.5, 'macro_f1': 0.4,
                               'per_class': {label: {'f1': 0.4} for label in LABELS}},
    }
    candidate_dir = root / 'models' / 'candidates'
    candidate_dir.mkdir(parents=True)
    (candidate_dir / 'frozen_candidate_metadata.json').write_text(
        json.dumps(metadata), encoding='utf-8')
    return metadata, {'train': 9, 'validation': 3, 'test': 3}


def test_repository_frozen_contract_and_checksum():
    metadata = load_frozen_metadata(PROJECT_ROOT, verify_artifact=True,
                                    strict_runtime=False)
    assert metadata['selected_model'] == 'logistic_regression'
    assert metadata['classifier_parameters']['C'] == 1.0
    assert metadata['classifier_parameters']['class_weight'] == 'balanced'
    assert metadata['classifier_parameters']['solver'] == 'lbfgs'
    assert metadata['features']['word']['ngram_range'] == [1, 2]
    assert metadata['features']['character']['ngram_range'] == [3, 5]
    assert metadata['feature_weights'] == {'word': 1.0, 'character': 1.25}
    assert metadata['random_state'] == RANDOM_STATE


def test_frozen_factory_exact_synthetic_metadata(tmp_path):
    metadata, _ = synthetic_repository(tmp_path)
    pipeline = make_frozen_pipeline(metadata)
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ['features', 'classifier']
    assert canonical_features(pipeline.named_steps['features']) == metadata['features']
    assert pipeline.named_steps['features'].transformer_weights == metadata['feature_weights']
    assert pipeline.named_steps['classifier'].get_params(deep=False) == metadata['classifier_parameters']
    with pytest.raises(ValueError, match='Only frozen'):
        make_frozen_pipeline(dict(metadata, selected_model='linear_svm'))


def test_synthetic_full_dry_run_and_one_time_guard(tmp_path):
    metadata, counts = synthetic_repository(tmp_path)
    prepared = prepare_final_model(tmp_path, expected_rows=counts,
                                   verify_artifact=False, strict_runtime=False)
    assert (prepared.train_rows, prepared.validation_rows, prepared.development_rows) == (9, 3, 12)
    assert prepared.test_integrity['rows'] == 3
    assert not (tmp_path / 'reports' / 'final_test_metrics.json').exists()
    result = evaluate_locked_test(prepared, tmp_path, expected_rows=counts,
                                  pretest_commit_sha='synthetic-fixture')
    assert result['protocol_version'] == PROTOCOL_VERSION
    assert result['test_rows'] == 3
    assert result['reload_predictions_match']
    assert set(result['test_metrics']) >= {
        'accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1',
        'per_class', 'confusion_matrix'
    }
    assert list(result['test_metrics']['per_class']) == list(LABELS)
    assert sum(sum(row) for row in result['test_metrics']['confusion_matrix']) == 3
    assert sum(result['test_metrics']['per_class'][x]['support'] for x in LABELS) == 3
    predictions = pd.read_csv(tmp_path / 'data' / 'processed' / 'final_test_predictions.csv')
    assert list(predictions) == ['row_index', 'actual_sentiment', 'predicted_sentiment', 'correct']
    assert predictions.row_index.tolist() == [0, 1, 2]
    artifact = tmp_path / 'models' / 'final' / 'final_model.joblib'
    assert sha256_file(artifact) == result['artifact_sha256']
    assert isinstance(joblib.load(artifact), Pipeline)
    manifest = json.loads((tmp_path / 'reports' / 'final_evaluation_manifest.json').read_text())
    assert manifest['evaluation_completed']
    assert manifest['pretest_commit_sha'] == 'synthetic-fixture'
    assert manifest['final_metrics_sha256'] == sha256_file(tmp_path / 'reports' / 'final_test_metrics.json')
    with pytest.raises(FileExistsError, match='already exists'):
        assert_evaluation_available(tmp_path)
    with pytest.raises(FileExistsError, match='already exists'):
        evaluate_locked_test(prepared, tmp_path, expected_rows=counts,
                             pretest_commit_sha='synthetic-fixture')


def test_prepare_does_not_read_test_labels(tmp_path):
    _, counts = synthetic_repository(tmp_path, invalid_test_label=True)
    prepared = prepare_final_model(tmp_path, expected_rows=counts,
                                   verify_artifact=False, strict_runtime=False)
    assert prepared.test_integrity['rows'] == 3
    assert not (tmp_path / 'reports' / 'final_test_metrics.json').exists()


def test_missing_and_changed_test_block_before_unlock(tmp_path):
    _, counts = synthetic_repository(tmp_path)
    path = tmp_path / 'data' / 'processed' / 'test_clean.csv'
    path.unlink()
    with pytest.raises(FileNotFoundError):
        prepare_final_model(tmp_path, expected_rows=counts,
                            verify_artifact=False, strict_runtime=False)
    test = pd.DataFrame({'text': ['altered'], 'clean_text': ['altered'],
                         'sentiment': ['negative'], 'split': ['test']})
    save_csv(test, path)
    with pytest.raises(ValueError, match='checksum'):
        prepare_final_model(tmp_path, expected_rows=counts,
                            verify_artifact=False, strict_runtime=False)


def test_metric_schema_and_class_order():
    truth = ['negative', 'neutral', 'positive']
    pred = ['neutral', 'neutral', 'positive']
    result = _test_metrics(truth, pred)
    assert result['confusion_matrix'] == [[0, 1, 0], [0, 1, 0], [0, 0, 1]]
    assert [result['per_class'][x]['support'] for x in LABELS] == [1, 1, 1]

def test_failed_synthetic_attempt_blocks_retry(tmp_path):
    _, counts = synthetic_repository(tmp_path)
    prepared = prepare_final_model(tmp_path, expected_rows=counts,
                                   verify_artifact=False, strict_runtime=False)
    test_path = tmp_path / 'data' / 'processed' / 'test_clean.csv'
    test_path.write_text(test_path.read_text(encoding='utf-8') + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='checksum'):
        evaluate_locked_test(prepared, tmp_path, expected_rows=counts,
                             pretest_commit_sha='synthetic-fixture')
    assert (tmp_path / 'reports' / '.final_evaluation_attempt.json').exists()
    with pytest.raises(FileExistsError, match='already exists'):
        evaluate_locked_test(prepared, tmp_path, expected_rows=counts,
                             pretest_commit_sha='synthetic-fixture')

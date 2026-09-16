"""Tests for frozen-model, aggregate-only external-domain evaluation."""
import csv
import hashlib
from pathlib import Path

import pytest

from src.config import LABELS, PROJECT_ROOT
from src.external_eval.evaluate import (
    _metric_payload, evaluate_external_dataset,
)
from src.external_eval.labeling import LABELING_COLUMNS
from src.inference import SentimentPrediction


def write_labels(path, labels=None):
    labels = labels or ['negative', 'neutral', 'positive']
    texts = [
        'This artificial app experience is terrible',
        'This artificial product arrived today',
        'I absolutely love this artificial phone',
    ]
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=LABELING_COLUMNS)
        writer.writeheader()
        for index, (text, label) in enumerate(zip(texts, labels, strict=True)):
            writer.writerow({
                'sample_id': f'sample-{index}', 'source': 'reddit',
                'post_id': f'post-{index}',
                'created_at': f'2026-09-1{index + 4}T10:00:00Z',
                'query': 'synthetic phone', 'text': text, 'label': label,
                'review_flag': '', 'notes': '',
            })
    return texts


class FakePredictor:
    artifact_sha256 = 'a' * 64
    model_version = 'synthetic-frozen-model'

    def __init__(self):
        self.calls = 0

    def predict_batch(self, texts):
        self.calls += 1
        predictions = ('negative', 'positive', 'positive')
        return [SentimentPrediction(
            text=text, clean_text=text.casefold(), sentiment=predictions[index],
            confidence=.8,
            probabilities={
                'negative': .8 if predictions[index] == 'negative' else .1,
                'neutral': .8 if predictions[index] == 'neutral' else .1,
                'positive': .8 if predictions[index] == 'positive' else .1,
            },
            model_version=self.model_version,
            model_artifact_sha256=self.artifact_sha256,
            inference_timestamp='2026-09-17T00:00:00Z')
            for index, text in enumerate(texts)]


def test_metric_schema_and_confusion_matrix_order():
    result = _metric_payload(
        ['negative', 'neutral', 'positive'],
        ['negative', 'positive', 'positive'])
    assert result['confusion_matrix_order'] == list(LABELS)
    assert result['confusion_matrix_axes'] == {
        'rows': 'actual', 'columns': 'predicted'}
    assert result['confusion_matrix'] == [[1, 0, 0], [0, 0, 1], [0, 0, 1]]
    assert set(result['per_class']) == set(LABELS)
    assert {'accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
            'weighted_f1'} <= set(result)


def test_external_evaluation_writes_aggregate_reports_only(tmp_path):
    labels = tmp_path/'labels.csv'
    raw_texts = write_labels(labels)
    predictor = FakePredictor()
    result = evaluate_external_dataset(
        labels, output_dir=tmp_path/'reports', predictor=predictor)
    assert predictor.calls == 1
    assert result['metrics']['dataset_rows'] == 3
    assert result['metrics']['training_or_tuning_performed'] is False
    assert result['metrics']['metrics']['confusion_matrix_order'] == list(LABELS)
    for path in result['paths'].values():
        assert path.is_file()
        assert path.stat().st_size > 0
    public_text = ''.join(
        path.read_text(encoding='utf-8')
        for name, path in result['paths'].items()
        if name != 'confusion_matrix')
    assert all(text not in public_text for text in raw_texts)
    comparison = result['metrics']['domain_gap_comparison']
    assert {'accuracy', 'macro_f1', 'negative_f1',
            'neutral_f1', 'positive_f1'} == set(comparison)


def test_incomplete_labels_stop_before_prediction_or_output(tmp_path):
    labels = tmp_path/'labels.csv'
    write_labels(labels, ['negative', '', 'positive'])
    predictor = FakePredictor()
    output = tmp_path/'reports'
    with pytest.raises(ValueError, match='no human label'):
        evaluate_external_dataset(labels, output_dir=output, predictor=predictor)
    assert predictor.calls == 0
    assert not output.exists()


def test_real_frozen_predictor_integration_on_synthetic_rows(tmp_path):
    labels = tmp_path/'labels.csv'
    write_labels(labels)
    result = evaluate_external_dataset(labels, output_dir=tmp_path/'reports')
    assert result['metrics']['model_artifact_sha256'] == (
        'a3e375cc0e75c68f2ac8de8a35d46369e566c3e0821d31c778018f4edc83451e')
    assert result['metrics']['dataset_rows'] == 3


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_official_benchmark_artifacts_remain_frozen():
    expected = {
        'models/final/final_model.joblib':
            'a3e375cc0e75c68f2ac8de8a35d46369e566c3e0821d31c778018f4edc83451e',
        'reports/final_test_metrics.json':
            'a9fd8daf5ee4d6cffa3f80e13c221b4597e066321c8eceefcd4e290e5846aa0c',
        'reports/final_evaluation_manifest.json':
            'c44bcb4971835a4edd2b6f6650266a4ae979af3931a55f2ea603a9e5e4274868',
    }
    for relative_path, digest in expected.items():
        content = (PROJECT_ROOT/relative_path).read_bytes()
        lf = content.replace(b'\r\n', b'\n')
        candidates = {
            hashlib.sha256(content).hexdigest(),
            hashlib.sha256(lf).hexdigest(),
            hashlib.sha256(lf.replace(b'\n', b'\r\n')).hexdigest(),
        }
        assert digest in candidates


def test_external_modules_do_not_import_official_test_runner():
    source = '\n'.join(
        path.read_text(encoding='utf-8')
        for path in (PROJECT_ROOT/'src'/'external_eval').glob('*.py'))
    assert 'src.final_evaluation' not in source
    assert 'processed_test' not in source
    assert '.fit(' not in source

"""Offline tests for the immutable application inference path."""
from dataclasses import asdict
import json

import joblib
import numpy as np
import pytest

from src.config import LABELS, PROJECT_ROOT
from src.inference import SentimentPredictor
from src.preprocessing import clean_text


@pytest.fixture(scope='module')
def predictor():
    return SentimentPredictor()


def test_final_model_loads_once_and_verifies_checksum(predictor):
    first = predictor._ensure_loaded()
    second = predictor._ensure_loaded()
    metadata = json.loads(
        (PROJECT_ROOT/'models/final/final_model_metadata.json').read_text(encoding='utf-8'))
    assert first is second
    assert predictor.artifact_sha256 == metadata['artifact_sha256']
    assert isinstance(first, type(joblib.load(PROJECT_ROOT/'models/final/final_model.joblib')))


def test_single_prediction_uses_existing_cleaner_and_schema(predictor):
    raw = '@brand I LOVE this phone!!! https://example.com'
    result = predictor.predict_one(raw)
    assert result.text == raw
    assert result.clean_text == clean_text(raw)
    assert result.sentiment in LABELS
    assert list(result.probabilities) == list(LABELS)
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert result.confidence == max(result.probabilities.values())
    assert result.model_artifact_sha256 == predictor.artifact_sha256
    assert result.inference_timestamp.endswith('Z')
    json.dumps(asdict(result))


def test_batch_is_deterministic_and_preserves_order(predictor):
    texts = [
        'I absolutely love this phone',
        'The product arrived today',
        'This app is terrible after the update',
    ]
    first = predictor.predict_batch(texts)
    second = predictor.predict_batch(texts)
    assert [item.text for item in first] == texts
    assert [item.sentiment for item in first] == [item.sentiment for item in second]
    assert [item.probabilities for item in first] == [item.probabilities for item in second]


def test_batch_calls_model_methods_once(monkeypatch, predictor):
    model = predictor._ensure_loaded()
    counts = {'predict': 0, 'predict_proba': 0}
    real_predict, real_proba = model.predict, model.predict_proba

    def predict(values):
        counts['predict'] += 1
        return real_predict(values)

    def predict_proba(values):
        counts['predict_proba'] += 1
        return real_proba(values)

    monkeypatch.setattr(model, 'predict', predict)
    monkeypatch.setattr(model, 'predict_proba', predict_proba)
    results = predictor.predict_batch(['good phone', 'ordinary update', 'awful app'])
    assert len(results) == 3
    assert counts == {'predict': 1, 'predict_proba': 1}


@pytest.mark.parametrize('value,error', [
    (None, TypeError),
    (5, TypeError),
    ('', ValueError),
    ('   ', ValueError),
    ('https://example.com', ValueError),
])
def test_invalid_single_inputs(value, error, predictor):
    with pytest.raises(error):
        predictor.predict_one(value)


def test_batch_input_rules(predictor):
    assert predictor.predict_batch([]) == []
    with pytest.raises(TypeError, match='sequence'):
        predictor.predict_batch('one string')
    with pytest.raises(TypeError, match=r'texts\[1\]'):
        predictor.predict_batch(['valid', None])
    limited = SentimentPredictor(max_text_characters=4)
    with pytest.raises(ValueError, match='exceeds'):
        limited.predict_one('12345')


def test_missing_model_and_bad_checksum(tmp_path):
    missing = SentimentPredictor(tmp_path/'missing.joblib', tmp_path/'metadata.json')
    with pytest.raises(FileNotFoundError, match='artifact'):
        missing.predict_one('valid')
    model_path = tmp_path/'model.joblib'
    model_path.write_bytes(b'not the model')
    metadata_path = tmp_path/'metadata.json'
    metadata_path.write_text(json.dumps({
        'artifact_sha256': '0'*64,
        'pretest_commit_sha': 'a'*40,
        'frozen_configuration': {'preprocessing': {
            'lowercase': True, 'strip_urls': True, 'mention_strategy': 'remove',
            'preserve_hashtag_words': True, 'emoji_strategy': 'preserve',
            'reduce_repeats': False}}}), encoding='utf-8')
    with pytest.raises(ValueError, match='SHA-256'):
        SentimentPredictor(model_path, metadata_path).predict_one('valid')


def test_probability_order_matches_pipeline_classes(predictor):
    result = predictor.predict_one('A plain product update')
    values = np.asarray([result.probabilities[label] for label in LABELS])
    assert result.sentiment == LABELS[int(np.argmax(values))]

def test_application_start_hash_record_matches_frozen_artifacts():
    import hashlib
    before = json.loads(
        (PROJECT_ROOT/'reports/application/official_hashes_before.json')
        .read_text(encoding='utf-8'))
    actual = {
        name: hashlib.sha256((PROJECT_ROOT/name).read_bytes()).hexdigest()
        for name in before}
    assert actual == before

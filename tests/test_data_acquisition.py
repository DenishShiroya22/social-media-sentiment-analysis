import json
from datasets import Dataset, Features, ClassLabel, Value
import pytest
from src.config import SPLITS
from src.data_acquisition import normalize_tweet_eval_split, ensure_dataset_available, sha256
from src.data_loader import save_csv

def source(names=None):
    return Dataset.from_dict({'text': ['NA', 'fine', 'great 😍'], 'label': [0,1,2]}, features=Features({'text': Value('string'), 'label': ClassLabel(names=names or ['negative','neutral','positive'])}))

def test_normalize():
    frame = normalize_tweet_eval_split(source(), 'validation')
    assert frame.text.tolist() == ['NA', 'fine', 'great 😍']
    assert frame.sentiment.tolist() == ['negative','neutral','positive']
    assert frame.split.eq('validation').all()

def test_semantics():
    with pytest.raises(ValueError, match='semantics'):
        normalize_tweet_eval_split(source(['positive','neutral','negative']), 'train')

def test_reuse_and_corruption(tmp_path, monkeypatch):
    import huggingface_hub
    import src.data_acquisition as acquisition
    monkeypatch.setattr(huggingface_hub.HfApi, 'dataset_info', lambda *a, **kw: pytest.fail('Reuse must be offline'))
    monkeypatch.setattr(acquisition, 'EXPECTED_ROWS', dict.fromkeys(SPLITS, 3))
    manifest = {'dataset':'cardiffnlp/tweet_eval','configuration':'sentiment','revision':'example','files':{}}
    for name in SPLITS:
        path = tmp_path / f'tweet_eval_{name}.csv'
        save_csv(normalize_tweet_eval_split(source(), name), path)
        manifest['files'][name] = {'sha256': sha256(path), 'rows':3}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    assert len(ensure_dataset_available(tmp_path, revision='example')) == 3
    with pytest.raises(ValueError, match='revision'):
        ensure_dataset_available(tmp_path)
    manifest['files']['train']['rows'] = 2
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Row-count mismatch'):
        ensure_dataset_available(tmp_path, revision='example')
    manifest['files']['train']['rows'] = 3
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    (tmp_path / 'tweet_eval_train.csv').write_text('broken')
    with pytest.raises(ValueError, match='Integrity'):
        ensure_dataset_available(tmp_path, revision='example')

def test_download_failure(tmp_path, monkeypatch):
    import huggingface_hub
    def fail(*args, **kwargs):
        raise ConnectionError('offline')
    monkeypatch.setattr(huggingface_hub.HfApi, 'dataset_info', fail)
    with pytest.raises(RuntimeError, match='No substitute dataset'):
        ensure_dataset_available(tmp_path)
    assert not list(tmp_path.glob('*.csv'))

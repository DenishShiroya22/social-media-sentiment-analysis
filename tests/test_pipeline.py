import pandas as pd
import pytest
from src.preprocessing import preprocess_dataframe
from src.data_validation import validate_dataframe, require_valid
from src.eda import analyze_train, descriptive_summary
from src.pipeline import run_phase1
from src.data_loader import save_csv, load_csv
from src.config import SPLITS

def test_contract():
    frame = pd.DataFrame({'text':["I don't like this!!!", 'Great 😍'], 'sentiment':['negative','positive'], 'split':['train']*2, 'brand':['Apple']*2})
    require_valid(validate_dataframe(frame))
    result = preprocess_dataframe(frame)
    require_valid(validate_dataframe(result, text_column='clean_text'))
    pd.testing.assert_frame_equal(result[frame.columns], frame)
    assert 'clean_text' in result and 'brand' in result

def test_clean_empty_blocks():
    frame = pd.DataFrame({'text':['@user https://example.com'], 'sentiment':['neutral'], 'split':['train']})
    with pytest.raises(ValueError, match='blank_text'):
        require_valid(validate_dataframe(preprocess_dataframe(frame), text_column='clean_text'))

def test_test_eda_locked(tmp_path):
    frame = pd.DataFrame({'text':['hello'], 'sentiment':['neutral'], 'split':['test']})
    for function in [analyze_train, descriptive_summary]:
        with pytest.raises(ValueError):
            function(frame)

def test_end_to_end_fixture(tmp_path, monkeypatch):
    # Explicit synthetic fixture; never written under the project's real data paths.
    import json
    import src.pipeline as pipeline
    paths = {}
    for name in SPLITS:
        paths[name] = tmp_path / 'data' / 'raw' / f'tweet_eval_{name}.csv'
        save_csv(pd.DataFrame({'text':['Great 😍 #Phone', "I don't like it", 'Released today'], 'sentiment':['positive','negative','neutral'], 'split':[name]*3}), paths[name])
    (tmp_path/'data'/'raw'/'manifest.json').write_text(json.dumps({'dataset':'SYNTHETIC TEST FIXTURE — NOT FOR MODEL EVALUATION','configuration':'sentiment','revision':'fixture'}))
    monkeypatch.setattr(pipeline, 'ensure_dataset_available', lambda *args: paths)
    result = run_phase1(tmp_path)
    assert result['outputs']['test']['rows'] == 3
    assert 'test_summary' not in result
    for name in SPLITS:
        assert load_csv(tmp_path / result['outputs'][name]['path']).split.eq(name).all()

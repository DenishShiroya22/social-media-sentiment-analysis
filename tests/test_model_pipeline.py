import json
import joblib
import pandas as pd
import pytest
from src.data_loader import save_csv
from src.model_pipeline import run_modeling, check_locked_test
from src.models import Experiment

def test_locked_test_mechanics_only(tmp_path):
    frame = pd.DataFrame({'text':['unread'], 'clean_text':['unread'],
        'sentiment':['INVALID TEST LABEL MUST NOT BE READ'], 'split':['test']})
    path = tmp_path / 'test_clean.csv'
    save_csv(frame, path)
    assert check_locked_test(path, 1)['rows'] == 1
    with pytest.raises(ValueError, match='row count'):
        check_locked_test(path, 2)

def test_model_pipeline_fixture(tmp_path):
    # Explicit synthetic fixture, never a benchmark result.
    root = tmp_path
    texts = ['terrible service','bad service','awful day','horrible day',
             'ordinary update','normal update','just news','regular news',
             'great service','good service','lovely day','wonderful day']
    labels = ['negative']*4 + ['neutral']*4 + ['positive']*4
    train = pd.DataFrame({'text':texts, 'clean_text':texts,
        'sentiment':labels, 'split':['train']*12})
    val_texts = ['bad service','awful day','normal update',
                 'regular news','great service','lovely day']
    validation = pd.DataFrame({'text':val_texts,'clean_text':val_texts,
        'sentiment':['negative']*2+['neutral']*2+['positive']*2,
        'split':['validation']*6})
    test = pd.DataFrame({'text':['locked'], 'clean_text':['locked'],
        'sentiment':['DO NOT INSPECT'], 'split':['test']})
    for name, frame in [('train',train),('validation',validation),('test',test)]:
        save_csv(frame, root/'data'/'processed'/f'{name}_clean.csv')
    experiments = (
        Experiment('fixture01','word_unigram','logistic_regression'),
        Experiment('fixture02','word_bigram','linear_svm'),
        Experiment('fixture03','word_bigram','naive_bayes'),
        Experiment('fixture04','word_bigram','random_forest'))
    result = run_modeling(root, experiments,
        {'train':12,'validation':6,'test':1})
    assert len(result['experiments']) == 4
    assert result['winner']['experiment_id'] in {e.id for e in experiments}
    assert result['selection_metric'] == 'validation_macro_f1'
    assert result['test_integrity_only']['rows'] == 1
    content = json.dumps(result).lower()
    assert 'test_accuracy' not in content
    assert 'test_macro_f1' not in content
    assert 'test_confusion_matrix' not in content
    assert (root/'reports'/'model_comparison.csv').is_file()
    artifact = root/'models'/'candidates'/'best_candidate.joblib'
    loaded = joblib.load(artifact)
    assert len(loaded.predict(validation.clean_text)) == 6
    assert result['candidate']['reload_predictions_match']

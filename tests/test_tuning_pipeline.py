import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from src.tuning_pipeline import (build_pipeline, cv_strategy, paired_bootstrap,
                                 C_VALUES, CLASS_WEIGHTS, FEATURE_VARIATIONS, _score_grid)
from src.config import RANDOM_STATE


def test_train_only_cv_and_pipeline_boundary():
    text = [f"train document {i}" for i in range(18)]
    labels = np.array(['negative', 'neutral', 'positive'] * 6)
    held_out = {'validation sentinel'}
    cv = cv_strategy()
    assert cv.n_splits == 3 and cv.random_state == RANDOM_STATE and cv.shuffle
    assert not held_out.intersection(text)
    for train_index, fold_index in cv.split(text, labels):
        assert not set(train_index).intersection(fold_index)
        assert len(train_index) + len(fold_index) == len(text)
        assert set(labels[fold_index]) == set(labels)
    for name in ('logistic_regression', 'linear_svm'):
        pipe = build_pipeline(name)
        assert isinstance(pipe, Pipeline)
        assert [step for step, _ in pipe.steps] == ['features', 'classifier']
        assert 'features__word__min_df' in pipe.get_params()
    assert len(C_VALUES) * len(CLASS_WEIGHTS) == 10
    assert len(FEATURE_VARIATIONS) == 4
    with pytest.raises(ValueError):
        build_pipeline('random_forest')


def test_paired_uncertainty_is_reproducible():
    truth = ['negative', 'neutral', 'positive'] * 8
    left = truth.copy()
    right = truth.copy()
    right[0] = 'neutral'
    a = paired_bootstrap(truth, left, right, draws=80, seed=42)
    b = paired_bootstrap(truth, left, right, draws=80, seed=42)
    assert a == b
    assert a['point_difference'] > 0
    assert len(a['ci_95_percentile']) == 2
    with pytest.raises(ValueError):
        paired_bootstrap([], [], [])


def test_cv_result_schema_on_tiny_train_fixture():
    text = (['bad ugly', 'bad awful', 'very bad', 'awful ugly'] +
            ['normal today', 'normal news', 'ordinary news', 'ordinary today'] +
            ['good nice', 'good great', 'very good', 'nice great'])
    labels = ['negative']*4 + ['neutral']*4 + ['positive']*4
    pipe = build_pipeline('linear_svm')
    pipe.set_params(features__word__min_df=1, features__character__min_df=1,
                    features__word__max_df=1.0, features__character__max_df=1.0)
    rows, params, best = _score_grid('linear_svm','fixture',pipe,
        {'classifier__C':[0.5,1.0]}, text, labels, cv_strategy(2))
    assert len(rows) == 2
    assert {'model','stage','params','mean_cv_macro_f1','std_cv_macro_f1',
            'rank_within_stage','mean_fit_seconds','mean_score_seconds','fold_scores'} <= rows[0].keys()
    assert params['classifier__C'] in (0.5,1.0)
    assert best['rank_within_stage'] == 1
    pipe.set_params(**params)
    pipe.fit(text, labels)
    return_prediction = pipe.predict(['bad ugly','normal news','good nice'])
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as directory:
        artifact=Path(directory)/'fixture.joblib'
        joblib.dump(pipe, artifact)
        assert np.array_equal(joblib.load(artifact).predict(['bad ugly','normal news','good nice']), return_prediction)

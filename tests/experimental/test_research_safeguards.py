"""Fast synthetic safeguards for the post-benchmark experimental track."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV

from src.config import LABELS
from src.experimental import ensemble
from src.experimental.xgboost_experiment import CANDIDATES, _new_model, require_sparse, run_xgboost
from src.tuning_pipeline import build_pipeline
from src.features import make_features


class AuditedModel(BaseEstimator, ClassifierMixin):
    fitted_sets = []
    predicted_sets = []

    def __init__(self, mode='lr'):
        self.mode = mode

    def fit(self, text, y):
        ids = frozenset(text)
        AuditedModel.fitted_sets.append(ids)
        self.train_ids_ = ids
        self.classes_ = np.asarray(LABELS)
        return self

    def predict_proba(self, text):
        AuditedModel.predicted_sets.append((self.train_ids_, tuple(text)))
        return np.tile([0.2, 0.3, 0.5], (len(text), 1))

    def decision_function(self, text):
        AuditedModel.predicted_sets.append((self.train_ids_, tuple(text)))
        return np.tile([-0.5, 0.1, 0.7], (len(text), 1))


@pytest.fixture
def toy_root(tmp_path):
    for name in ensemble.OFFICIAL_PATHS:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'frozen')
    before = {name: ensemble.sha256_file(tmp_path/name)
              for name in ensemble.OFFICIAL_PATHS}
    path = tmp_path/'reports/experimental/official_hashes_before.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(before), encoding='utf-8')
    for split, count in [('train', 18), ('validation', 6)]:
        labels = list(LABELS)*(count//3)
        rows = [{'text': f'{label} token {i}', 'clean_text': f'{label} token {i}',
                 'sentiment': label, 'split': split}
                for i, label in enumerate(labels)]
        path = tmp_path/f'data/processed/{split}_clean.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
    tuning = Path(ensemble.PROJECT_ROOT)/'reports/tuning_metrics.json'
    frozen = tmp_path/'reports/tuning_metrics.json'
    frozen.write_bytes(tuning.read_bytes())
    return tmp_path


def test_train_validation_loader_never_needs_test(toy_root):
    train, val = ensemble.load_train_validation(
        toy_root, expected_rows={'train': 18, 'validation': 6})
    assert (len(train), len(val)) == (18, 6)
    assert not (toy_root/'data/processed/test_clean.csv').exists()


def test_official_hash_guard_detects_mutation(toy_root):
    assert ensemble.assert_official_unchanged(toy_root)
    (toy_root/ensemble.OFFICIAL_PATHS[0]).write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='checksum changed'):
        ensemble.assert_official_unchanged(toy_root)


def test_oof_each_row_predicted_without_seeing_it():
    AuditedModel.fitted_sets.clear()
    AuditedModel.predicted_sets.clear()
    text = [f'id{i}' for i in range(18)]
    y = np.asarray(list(LABELS)*6)
    features = ensemble._oof_meta(text, y, AuditedModel('lr'),
                                   AuditedModel('svm'), ensemble.cv_folds())
    assert features.shape == (18, 6)
    assert np.isfinite(features).all()
    for fitted, hold in AuditedModel.predicted_sets:
        assert set(hold).isdisjoint(fitted)
    observed = [item for _, hold in AuditedModel.predicted_sets for item in hold]
    assert sorted(observed) == sorted(text*2)


def test_hard_tie_resolves_to_lr():
    lr = np.asarray(['negative', 'positive', 'neutral'])
    svm = np.asarray(['positive', 'positive', 'negative'])
    hard = ensemble.hard_vote(lr, svm)
    assert hard.tolist() == lr.tolist()
    with pytest.raises(ValueError, match='align'):
        ensemble.hard_vote(lr, svm[:2])


def test_soft_voting_outputs_valid_labels_and_probabilities():
    text = ['one', 'two']
    lr = AuditedModel('lr').fit(text, LABELS[:2])
    svm = AuditedModel('svm').fit(text, LABELS[:2])
    model = ensemble.SoftVotingModel(lr, svm, 0.6)
    assert model.predict(text).tolist() == ['positive', 'positive']
    assert np.allclose(ensemble._ordered_outputs(lr, text, 'predict_proba').sum(axis=1), 1)


def test_calibration_is_train_internal_and_produces_probabilities():
    text = [f'{label} repeated token {i}' for i in range(18) for label in LABELS]
    y = np.asarray(list(LABELS)*18)
    estimator = build_pipeline('linear_svm')
    calibrated = CalibratedClassifierCV(
        estimator=estimator, cv=ensemble.cv_folds(), method='sigmoid',
        n_jobs=1, ensemble=True).fit(text, y)
    probabilities = calibrated.predict_proba(['unseen neutral repeated token'])
    assert probabilities.shape == (1, 3)
    assert np.allclose(probabilities.sum(axis=1), 1)


def test_sparse_xgboost_fit_and_fixed_seed_predictions():
    actual_features = make_features('combined').fit_transform(['bad service', 'good service', 'bad day', 'good day'])
    assert sparse.issparse(require_sparse(actual_features))
    x = require_sparse(sparse.csr_matrix(np.asarray(
        [[i%3 == j for j in range(3)] for i in range(18)], dtype=np.float32)))
    y = np.asarray([i%3 for i in range(18)])
    params = {'max_depth': 2, 'learning_rate': 0.1, 'n_estimators': 3,
              'subsample': 0.8, 'colsample_bytree': 1.0,
              'min_child_weight': 1, 'reg_lambda': 1,
              'sample_weight_policy': 'none'}
    a, b = _new_model(params), _new_model(params)
    a.fit(x, y)
    b.fit(x, y)
    assert a.get_params()['objective'] == 'multi:softprob'
    assert a.get_params()['tree_method'] == 'hist'
    assert len(CANDIDATES) == 10
    assert np.array_equal(a.predict(x), b.predict(x))
    with pytest.raises(TypeError, match='sparse'):
        require_sparse(x.toarray())


def test_experimental_paths_are_separate_from_final_namespace():
    source = Path(ensemble.__file__).read_text(encoding='utf-8')
    xgb = Path(__import__(
        'src.experimental.xgboost_experiment', fromlist=['__file__']).__file__).read_text(encoding='utf-8')
    assert 'data/processed/test_clean.csv' not in source+xgb
    assert '.toarray(' not in xgb
    assert "models/experimental/" in xgb
    assert "models'/'experimental'" in source
    assert 'models/final/' in ensemble.OFFICIAL_PATHS[2]


def test_report_schema_if_generated():
    report_path = Path(ensemble.PROJECT_ROOT)/'reports/experimental/ensemble_metrics.json'
    if not report_path.exists():
        pytest.skip('Research report not generated yet')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    assert report['cv']['source'] == 'TRAIN only'
    assert set(('logistic_regression', 'linear_svm', 'hard_vote',
                'soft_vote', 'stacking')) <= set(report['records'])
    for record in report['records'].values():
        assert 'macro_f1' in record['validation']
        assert set(LABELS) <= set(record['validation']['per_class'])




def test_synthetic_ensemble_run_isolated_and_reports_valid(toy_root):
    report = ensemble.run_ensembles(
        toy_root, expected_rows={'train': 18, 'validation': 6})
    assert report['cv']['source'] == 'TRAIN only'
    assert report['records']['hard_vote']['validation'] == report['records']['logistic_regression']['validation']
    assert (toy_root/'reports/experimental/ensemble_comparison.csv').exists()
    assert ensemble.assert_official_unchanged(toy_root)
    tiny = tuple({**params, 'n_estimators': 3, 'max_depth': 2}
                 for params in CANDIDATES)
    xreport = run_xgboost(
        toy_root, expected_rows={'train': 18, 'validation': 6},
        candidates=tiny)
    assert xreport['cv']['candidate_count'] == 10
    assert xreport['cv']['fit_count'] == 30
    assert all(item['format'] == 'CSR' and not item['dense_conversion']
               for item in xreport['cv']['fold_sparse_matrices'])
    assert (toy_root/'reports/experimental/xgboost_results.csv').exists()
    assert not (toy_root/'data/processed/test_clean.csv').exists()
    assert ensemble.assert_official_unchanged(toy_root)

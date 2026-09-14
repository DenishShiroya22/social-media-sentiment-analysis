"""Sparse TRAIN-CV XGBoost experiment; VALIDATION is used only after selection."""
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import xgboost

from src.config import EXPECTED_ROWS, LABELS, PROJECT_ROOT, RANDOM_STATE
from src.evaluation import evaluate_validation
from src.features import make_features
from src.experimental.ensemble import (
    assert_official_unchanged, cv_folds, frozen_base_pipelines,
    load_train_validation, paired_bootstrap, sha256_file,
)

# Five deliberately chosen tree settings x two multiclass sample-weight policies.
TREE_SETTINGS = (
    {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 40,
     'subsample': 0.8, 'colsample_bytree': 0.5, 'min_child_weight': 1, 'reg_lambda': 1},
    {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 60,
     'subsample': 0.8, 'colsample_bytree': 0.75, 'min_child_weight': 1, 'reg_lambda': 5},
    {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 50,
     'subsample': 0.8, 'colsample_bytree': 0.75, 'min_child_weight': 3, 'reg_lambda': 5},
    {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 40,
     'subsample': 0.8, 'colsample_bytree': 0.5, 'min_child_weight': 3, 'reg_lambda': 1},
    {'max_depth': 7, 'learning_rate': 0.03, 'n_estimators': 80,
     'subsample': 0.8, 'colsample_bytree': 0.5, 'min_child_weight': 3, 'reg_lambda': 5},
)
CANDIDATES = tuple({**params, 'sample_weight_policy': weighting}
                   for params in TREE_SETTINGS for weighting in ('none', 'balanced'))


def require_sparse(matrix):
    """Refuse any dense TF-IDF conversion, including during the final fit."""
    if not sparse.issparse(matrix):
        raise TypeError('XGBoost experiment requires a scipy sparse matrix')
    return matrix.tocsr().astype(np.float32, copy=False)


def _new_model(params):
    clf = {key: value for key, value in params.items() if key != 'sample_weight_policy'}
    return XGBClassifier(
        **clf, objective='multi:softprob', num_class=3,
        eval_metric='mlogloss', tree_method='hist', max_bin=32,
        n_jobs=2, random_state=RANDOM_STATE)


def _weights(labels, policy):
    if policy == 'none':
        return None
    if policy == 'balanced':
        return compute_sample_weight('balanced', labels).astype(np.float32)
    raise ValueError('Unknown sample weighting policy')


def _encode(labels):
    lookup = {label: i for i, label in enumerate(LABELS)}
    return np.asarray([lookup[label] for label in labels], dtype=np.int32)


def _decode(encoded):
    return np.asarray(LABELS)[np.asarray(encoded, dtype=int)]


def _comparison_row(record):
    m = record['validation']
    return {'model': record['model'], 'cv_macro_f1': record['cv_macro_f1'],
            'cv_std': record['cv_std'], 'validation_accuracy': m['accuracy'],
            'validation_macro_precision': m['macro_precision'],
            'validation_macro_recall': m['macro_recall'],
            'validation_macro_f1': m['macro_f1'],
            'negative_f1': m['per_class']['negative']['f1'],
            'neutral_f1': m['per_class']['neutral']['f1'],
            'positive_f1': m['per_class']['positive']['f1'],
            'fit_seconds': record['fit_seconds'],
            'predict_seconds': record['predict_seconds']}


def run_xgboost(root=PROJECT_ROOT, expected_rows=EXPECTED_ROWS, candidates=CANDIDATES,
                n_splits=3):
    root = Path(root)
    assert_official_unchanged(root)
    if not 10 <= len(candidates) <= 20:
        raise ValueError('Constrained XGBoost search requires 10–20 candidates')
    train, validation = load_train_validation(root, expected_rows)
    texts = train['clean_text'].tolist()
    y_labels = train['sentiment'].to_numpy()
    y = _encode(y_labels)
    val_text = validation['clean_text'].tolist()
    val_y = validation['sentiment'].to_numpy()
    cv = cv_folds(n_splits)
    folds = list(cv.split(texts, y))
    print(f'XGBoost plan: {len(candidates)} candidates x {n_splits} TRAIN folds '
          f'= {len(candidates)*n_splits} CV fits plus one final fit; '
          'serial models, n_jobs=2 per model.', flush=True)
    rows = [{'candidate': i, 'params': dict(params), 'fold_scores': [],
             'fold_fit_seconds': [], 'fold_predict_seconds': []}
            for i, params in enumerate(candidates)]
    fold_matrix_info = []
    for fold_no, (fit_idx, hold_idx) in enumerate(folds, start=1):
        # Vocabulary and IDF are fitted only on the fold-training portion.
        features = make_features('combined')
        fit_text = [texts[i] for i in fit_idx]
        hold_text = [texts[i] for i in hold_idx]
        x_fit = require_sparse(features.fit_transform(fit_text))
        x_hold = require_sparse(features.transform(hold_text))
        y_fit, y_hold = y[fit_idx], y[hold_idx]
        fold_matrix_info.append({'fold': fold_no, 'fit_shape': list(x_fit.shape),
                                 'hold_shape': list(x_hold.shape),
                                 'fit_nnz': int(x_fit.nnz), 'hold_nnz': int(x_hold.nnz),
                                 'format': 'CSR', 'dense_conversion': False})
        for row in rows:
            params = row['params']
            model = _new_model(params)
            start = perf_counter()
            model.fit(x_fit, y_fit, sample_weight=_weights(y_fit, params['sample_weight_policy']))
            row['fold_fit_seconds'].append(perf_counter()-start)
            start = perf_counter()
            pred = model.predict(x_hold)
            row['fold_predict_seconds'].append(perf_counter()-start)
            row['fold_scores'].append(float(f1_score(y_hold, pred, average='macro', zero_division=0)))
            print(f"XGBoost fold {fold_no}/{n_splits}, candidate "
                  f"{row['candidate']+1}/{len(rows)}: macro-F1 "
                  f"{row['fold_scores'][-1]:.4f}", flush=True)
        del x_fit, x_hold, features
        assert_official_unchanged(root)
    for row in rows:
        row['mean_cv_macro_f1'] = float(np.mean(row['fold_scores']))
        row['std_cv_macro_f1'] = float(np.std(row['fold_scores']))
        row['total_cv_fit_seconds'] = float(sum(row['fold_fit_seconds']))
    best = max(rows, key=lambda r: (r['mean_cv_macro_f1'],
                                    -r['std_cv_macro_f1'], -r['candidate']))
    # No VALIDATION labels or features entered the selection above.
    features = make_features('combined')
    x_train = require_sparse(features.fit_transform(texts))
    x_val = require_sparse(features.transform(val_text))
    model = _new_model(best['params'])
    start = perf_counter()
    model.fit(x_train, y, sample_weight=_weights(y, best['params']['sample_weight_policy']))
    fit_seconds = perf_counter()-start
    start = perf_counter()
    val_pred = _decode(model.predict(x_val))
    predict_seconds = perf_counter()-start
    metrics = evaluate_validation(val_y, val_pred)
    ensemble_report = json.loads((root/'reports/experimental/ensemble_metrics.json').read_text(encoding='utf-8'))
    baseline = ensemble_report['records']['logistic_regression']
    record = {'model': 'xgboost', 'cv_macro_f1': best['mean_cv_macro_f1'],
              'cv_std': best['std_cv_macro_f1'], 'validation': metrics,
              'fit_seconds': float(fit_seconds), 'predict_seconds': float(predict_seconds),
              'details': {'params': best['params'], 'objective': 'multi:softprob',
                          'num_class': 3, 'eval_metric': 'mlogloss',
                          'tree_method': 'hist', 'random_state': RANDOM_STATE,
                          'xgboost_version': xgboost.__version__, 'max_bin': 32,
                          'sparse_input': True, 'dense_conversion': False,
                          'feature_shape': list(x_train.shape),
                          'feature_nnz': int(x_train.nnz)}}
    out = root/'reports/experimental'
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**{'candidate': r['candidate']}, **r['params'],
                   'mean_cv_macro_f1': r['mean_cv_macro_f1'],
                   'std_cv_macro_f1': r['std_cv_macro_f1'],
                   'fold_scores': json.dumps(r['fold_scores']),
                   'total_cv_fit_seconds': r['total_cv_fit_seconds']}
                  for r in rows]).to_csv(out/'xgboost_results.csv', index=False)
    xreport = {'status': 'EXPERIMENTAL DEVELOPMENT MODEL — NO NEW TEST EVALUATION',
               'cv': {'source': 'TRAIN only', 'n_splits': n_splits,
                      'shuffle': True, 'random_state': RANDOM_STATE,
                      'scoring': 'f1_macro', 'candidate_count': len(rows),
                      'fit_count': len(rows)*n_splits,
                      'feature_fitting': 'within each TRAIN fold',
                      'resource_adjustment': 'Initial 160-tree full-fold pilot exceeded two minutes; interrupted before any CV score or VALIDATION access. Final search uses 40-80 trees and 32 histogram bins.',
                      'fold_sparse_matrices': fold_matrix_info},
               'search': rows, 'selected_candidate': best['candidate'],
               'record': record}
    (out/'xgboost_metrics.json').write_text(json.dumps(xreport, indent=2)+'\n', encoding='utf-8')
    table = pd.read_csv(out/'ensemble_comparison.csv')
    table = table[table['model'] != 'xgboost']
    table = pd.concat([table, pd.DataFrame([_comparison_row(record)])], ignore_index=True)
    table.to_csv(out/'ensemble_comparison.csv', index=False)
    if metrics['macro_f1'] > baseline['validation']['macro_f1']:
        path = root/'models/experimental/best_xgboost.joblib'
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'features': features, 'classifier': model, 'labels': LABELS},
                    path, compress=3)
        loaded = joblib.load(path)
        if not np.array_equal(
                _decode(loaded['classifier'].predict(require_sparse(loaded['features'].transform(val_text)))),
                val_pred):
            raise AssertionError('Experimental XGBoost reload differs')
        (path.parent/'best_xgboost_metadata.json').write_text(json.dumps({
            'status': 'EXPERIMENTAL DEVELOPMENT MODEL — NOT OFFICIAL',
            'model': 'xgboost', 'validation_macro_f1': metrics['macro_f1'],
            'artifact_sha256': sha256_file(path), 'artifact_bytes': path.stat().st_size,
            'test_policy': 'TEST not loaded or predicted'}, indent=2)+'\n', encoding='utf-8')
    best_ensemble = max(('soft_vote', 'stacking'),
                        key=lambda name: ensemble_report['records'][name]['validation']['macro_f1'])
    if metrics['macro_f1'] > ensemble_report['records'][best_ensemble]['validation']['macro_f1']:
        lr = clone(frozen_base_pipelines(root)['logistic_regression']).fit(texts, y_labels)
        lr_pred = lr.predict(val_text)
        bootstrap = paired_bootstrap(val_y, val_pred, lr_pred)
        (out/'bootstrap_comparison.json').write_text(
            json.dumps({'candidate': 'xgboost', 'baseline': 'logistic_regression',
                        **bootstrap}, indent=2)+'\n', encoding='utf-8')
        ci = bootstrap['ci_95_percentile']
        (out/'bootstrap_comparison.md').write_text(
            '# Paired VALIDATION comparison\n\nBest experimental model: XGBoost. '
            f"Macro-F1 minus frozen LR: {bootstrap['difference_candidate_minus_lr']:+.4f}. "
            f'Seed-{RANDOM_STATE} paired-bootstrap 95% percentile interval: '
            f'[{ci[0]:+.4f}, {ci[1]:+.4f}] over {bootstrap["draws"]} draws. '
            'This measures holdout sampling sensitivity, not independent evidence.\n',
            encoding='utf-8')
    assert_official_unchanged(root)
    print(f"XGBoost winner CV {best['mean_cv_macro_f1']:.4f} "
          f"VALIDATION {metrics['macro_f1']:.4f}", flush=True)
    return xreport


if __name__ == '__main__':
    run_xgboost()

"""Leakage-safe two-model voting, calibration and explicit OOF stacking."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from src.config import EXPECTED_ROWS, LABELS, PROJECT_ROOT, RANDOM_STATE
from src.data_loader import load_csv
from src.data_validation import require_valid, validate_dataframe
from src.evaluation import evaluate_validation
from src.tuning_pipeline import build_pipeline

OFFICIAL_PATHS = (
    'reports/final_test_metrics.json',
    'reports/final_evaluation_manifest.json',
    'models/final/final_model.joblib',
)
SOFT_WEIGHTS = (0.30, 0.40, 0.50, 0.60, 0.70)  # LR share; SVM gets 1-share
META_CONFIGS = (
    {'C': 0.1, 'class_weight': None},
    {'C': 0.1, 'class_weight': 'balanced'},
    {'C': 1.0, 'class_weight': None},
    {'C': 1.0, 'class_weight': 'balanced'},
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def assert_official_unchanged(root=PROJECT_ROOT):
    root = Path(root)
    before = json.loads((root/'reports/experimental/official_hashes_before.json').read_text(encoding='utf-8'))
    if set(before) != set(OFFICIAL_PATHS):
        raise ValueError('Official benchmark protection list changed')
    after = {name: sha256_file(root/name) for name in OFFICIAL_PATHS}
    if after != before:
        raise RuntimeError('Official benchmark checksum changed; stop experimental work')
    return after


def load_train_validation(root=PROJECT_ROOT, expected_rows=EXPECTED_ROWS):
    root = Path(root)
    frames = {}
    for name in ('train', 'validation'):
        frame = load_csv(root/'data'/'processed'/f'{name}_clean.csv')
        require_valid(validate_dataframe(frame, name))
        require_valid(validate_dataframe(frame, name, text_column='clean_text'))
        if len(frame) != expected_rows[name]:
            raise ValueError(f'{name} row count changed')
        frames[name] = frame
    return frames['train'], frames['validation']


def frozen_base_pipelines(root=PROJECT_ROOT):
    """Use the already-selected TRAIN-only development configurations."""
    tuning = json.loads((Path(root)/'reports/tuning_metrics.json').read_text(encoding='utf-8'))
    result = {}
    for name in ('logistic_regression', 'linear_svm'):
        params = tuning['finalists'][name]['params']
        pipeline = build_pipeline(name)
        pipeline.set_params(**params)
        result[name] = pipeline
    if result['logistic_regression'].named_steps['classifier'].get_params()['class_weight'] != 'balanced':
        raise ValueError('Frozen LR configuration changed')
    return result


def cv_folds(n_splits=3):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def _ordered_outputs(model, text, method):
    raw = getattr(model, method)(text)
    order = [list(model.classes_).index(label) for label in LABELS]
    return np.asarray(raw)[:, order]


def _meta_features(lr, svm, text):
    """Three LR probabilities plus three SVM margins; no TF-IDF passthrough."""
    lr_prob = _ordered_outputs(lr, text, 'predict_proba')
    svm_margin = _ordered_outputs(svm, text, 'decision_function')
    result = np.hstack((lr_prob, svm_margin))
    if result.shape != (len(text), 6):
        raise ValueError('Stack meta-feature dimension changed')
    return result


def _oof_meta(text, labels, lr_proto, svm_proto, cv):
    """Fit full text Pipelines inside each fold and return OOF meta-features."""
    text = np.asarray(text, dtype=object)
    labels = np.asarray(labels)
    result = np.empty((len(text), 6), dtype=np.float64)
    touched = np.zeros(len(text), dtype=int)
    for train_idx, hold_idx in cv.split(text, labels):
        lr = clone(lr_proto).fit(text[train_idx].tolist(), labels[train_idx])
        svm = clone(svm_proto).fit(text[train_idx].tolist(), labels[train_idx])
        result[hold_idx] = _meta_features(lr, svm, text[hold_idx].tolist())
        touched[hold_idx] += 1
    if not np.all(touched == 1):
        raise AssertionError('OOF row assignment is not exactly once')
    return result


def _meta_model(params):
    return LogisticRegression(C=params['C'], class_weight=params['class_weight'],
                              solver='lbfgs', max_iter=350, random_state=RANDOM_STATE)


def hard_vote(lr_pred, svm_pred):
    """With two voters, resolve every disagreement in favor of frozen LR."""
    lr_pred, svm_pred = np.asarray(lr_pred), np.asarray(svm_pred)
    if lr_pred.shape != svm_pred.shape:
        raise ValueError('Hard-vote predictions must align')
    return lr_pred.copy()

def _labels_from_proba(prob):
    return np.asarray(LABELS)[np.argmax(prob, axis=1)]


@dataclass
class SoftVotingModel:
    lr: object
    calibrated_svm: object
    lr_weight: float

    def predict(self, text):
        p_lr = _ordered_outputs(self.lr, text, 'predict_proba')
        p_svm = _ordered_outputs(self.calibrated_svm, text, 'predict_proba')
        return _labels_from_proba(self.lr_weight*p_lr + (1-self.lr_weight)*p_svm)


@dataclass
class ExplicitStackingModel:
    lr: object
    svm: object
    meta: object

    def predict(self, text):
        return self.meta.predict(_meta_features(self.lr, self.svm, text))


def paired_bootstrap(truth, candidate, baseline, draws=2000, seed=RANDOM_STATE):
    truth, candidate, baseline = map(np.asarray, (truth, candidate, baseline))
    if not (len(truth) == len(candidate) == len(baseline)):
        raise ValueError('Paired arrays must align')
    rng = np.random.default_rng(seed)
    delta = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, len(truth), len(truth))
        delta[i] = (f1_score(truth[idx], candidate[idx], labels=LABELS, average='macro', zero_division=0)
                    - f1_score(truth[idx], baseline[idx], labels=LABELS, average='macro', zero_division=0))
    point = f1_score(truth, candidate, labels=LABELS, average='macro', zero_division=0) - \
            f1_score(truth, baseline, labels=LABELS, average='macro', zero_division=0)
    return {'difference_candidate_minus_lr': float(point),
            'ci_95_percentile': [float(x) for x in np.quantile(delta, [0.025, 0.975])],
            'draws': draws, 'seed': seed}


def _record(name, fold_scores, metrics, fit_seconds, predict_seconds, details):
    return {'model': name, 'cv_macro_f1': float(np.mean(fold_scores)),
            'cv_std': float(np.std(fold_scores)),
            'validation': metrics, 'fit_seconds': float(fit_seconds),
            'predict_seconds': float(predict_seconds), 'details': details}


def run_ensembles(root=PROJECT_ROOT, expected_rows=EXPECTED_ROWS, n_splits=3):
    root = Path(root)
    official_hashes = assert_official_unchanged(root)
    train, validation = load_train_validation(root, expected_rows)
    text = np.asarray(train['clean_text'].tolist(), dtype=object)
    y = train['sentiment'].to_numpy()
    val_text = validation['clean_text'].tolist()
    val_y = validation['sentiment'].to_numpy()
    base = frozen_base_pipelines(root)
    lr_proto, svm_proto = base['logistic_regression'], base['linear_svm']
    outer = cv_folds(n_splits)
    fold_indices = list(outer.split(text, y))
    print(f'Ensemble plan: {n_splits} outer folds; approximately 44 base fits including '
          f'{n_splits}x{n_splits} inner OOF fits per base; {len(SOFT_WEIGHTS)} soft weights '
          f'and {len(META_CONFIGS)} meta configurations; serial execution.', flush=True)
    fold_scores = {name: [] for name in ('logistic_regression','linear_svm','hard_vote')}
    soft_scores = {weight: [] for weight in SOFT_WEIGHTS}
    meta_scores = {i: [] for i in range(len(META_CONFIGS))}
    outer_predictions = {'logistic_regression': np.empty(len(text),dtype=object),
                         'linear_svm': np.empty(len(text),dtype=object)}
    stage_start = perf_counter()
    for fold_number, (train_idx, hold_idx) in enumerate(fold_indices, start=1):
        x_fit = text[train_idx].tolist()
        y_fit = y[train_idx]
        x_hold = text[hold_idx].tolist()
        y_hold = y[hold_idx]
        # Every base feature extractor is fitted on this outer TRAIN partition.
        lr = clone(lr_proto).fit(x_fit, y_fit)
        svm = clone(svm_proto).fit(x_fit, y_fit)
        lr_pred, svm_pred = lr.predict(x_hold), svm.predict(x_hold)
        outer_predictions['logistic_regression'][hold_idx] = lr_pred
        outer_predictions['linear_svm'][hold_idx] = svm_pred
        fold_scores['logistic_regression'].append(f1_score(y_hold,lr_pred,labels=LABELS,average='macro'))
        fold_scores['linear_svm'].append(f1_score(y_hold,svm_pred,labels=LABELS,average='macro'))
        # Two-voter tie goes to LR; this makes hard voting exactly the LR output.
        hard_pred = hard_vote(lr_pred, svm_pred)
        fold_scores['hard_vote'].append(f1_score(y_hold,hard_pred,labels=LABELS,average='macro'))
        calibrated = CalibratedClassifierCV(
            estimator=clone(svm_proto), method='sigmoid',
            cv=cv_folds(n_splits), n_jobs=1, ensemble=True)
        calibrated.fit(x_fit, y_fit)
        lr_prob = _ordered_outputs(lr, x_hold, 'predict_proba')
        svm_prob = _ordered_outputs(calibrated, x_hold, 'predict_proba')
        for weight in SOFT_WEIGHTS:
            pred = _labels_from_proba(weight*lr_prob+(1-weight)*svm_prob)
            soft_scores[weight].append(f1_score(y_hold,pred,labels=LABELS,average='macro'))
        inner_oof = _oof_meta(x_fit,y_fit,lr_proto,svm_proto,cv_folds(n_splits))
        hold_meta = _meta_features(lr,svm,x_hold)
        for i,params in enumerate(META_CONFIGS):
            meta = _meta_model(params).fit(inner_oof,y_fit)
            meta_pred = meta.predict(hold_meta)
            meta_scores[i].append(f1_score(y_hold,meta_pred,labels=LABELS,average='macro'))
        print(f'Ensemble outer fold {fold_number}/{n_splits} complete',flush=True)
        assert_official_unchanged(root)
    cv_seconds = perf_counter()-stage_start
    best_weight = max(SOFT_WEIGHTS,key=lambda w:(np.mean(soft_scores[w]),-np.std(soft_scores[w]),-abs(w-.5)))
    best_meta = max(range(len(META_CONFIGS)),key=lambda i:(np.mean(meta_scores[i]),-np.std(meta_scores[i]),-i))
    print(f'Soft CV winner LR weight {best_weight}; stack meta {META_CONFIGS[best_meta]}',flush=True)
    records, prediction_map = {}, {}
    full_start = perf_counter()
    lr = clone(lr_proto).fit(text.tolist(),y)
    lr_fit_seconds = perf_counter()-full_start
    start=perf_counter()
    lr_pred = lr.predict(val_text)
    lr_pred_seconds=perf_counter()-start
    prediction_map['logistic_regression']=lr_pred
    records['logistic_regression']=_record('logistic_regression',fold_scores['logistic_regression'],
        evaluate_validation(val_y,lr_pred),lr_fit_seconds,lr_pred_seconds,
        {'configuration':'frozen TRAIN-only LR development baseline'})
    full_start=perf_counter()
    svm=clone(svm_proto).fit(text.tolist(),y)
    svm_fit_seconds=perf_counter()-full_start
    start=perf_counter()
    svm_pred=svm.predict(val_text)
    svm_pred_seconds=perf_counter()-start
    prediction_map['linear_svm']=svm_pred
    records['linear_svm']=_record('linear_svm',fold_scores['linear_svm'],
        evaluate_validation(val_y,svm_pred),svm_fit_seconds,svm_pred_seconds,
        {'configuration':'frozen TRAIN-only SVM development baseline'})
    hard_pred=hard_vote(lr_pred,svm_pred)
    prediction_map['hard_vote']=hard_pred
    records['hard_vote']=_record('hard_vote',fold_scores['hard_vote'],
        evaluate_validation(val_y,hard_pred),lr_fit_seconds+svm_fit_seconds,
        lr_pred_seconds+svm_pred_seconds,{'tie_rule':'choose LR on disagreement',
        'consequence':'with exactly two voters, hard vote equals LR predictions'})
    full_start=perf_counter()
    calibrated=CalibratedClassifierCV(estimator=clone(svm_proto),method='sigmoid',
        cv=cv_folds(n_splits),n_jobs=1,ensemble=True).fit(text.tolist(),y)
    calibration_seconds=perf_counter()-full_start
    soft_model=SoftVotingModel(lr,calibrated,best_weight)
    start=perf_counter()
    soft_pred=soft_model.predict(val_text)
    soft_pred_seconds=perf_counter()-start
    prediction_map['soft_vote']=soft_pred
    records['soft_vote']=_record('soft_vote',soft_scores[best_weight],
        evaluate_validation(val_y,soft_pred),lr_fit_seconds+calibration_seconds,
        soft_pred_seconds,{'lr_weight':best_weight,'svm_weight':1-best_weight,
        'calibration':'sigmoid','calibration_cv_folds':n_splits,
        'calibration_scope':'TRAIN only, nested within each outer fold during weight CV'})
    full_start=perf_counter()
    full_oof=_oof_meta(text,y,lr_proto,svm_proto,cv_folds(n_splits))
    meta=_meta_model(META_CONFIGS[best_meta]).fit(full_oof,y)
    stack_fit_seconds=perf_counter()-full_start+lr_fit_seconds+svm_fit_seconds
    stack_model=ExplicitStackingModel(lr,svm,meta)
    start=perf_counter()
    stack_pred=stack_model.predict(val_text)
    stack_pred_seconds=perf_counter()-start
    prediction_map['stacking']=stack_pred
    records['stacking']=_record('stacking',meta_scores[best_meta],
        evaluate_validation(val_y,stack_pred),stack_fit_seconds,stack_pred_seconds,
        {'base_estimators':['logistic_regression','linear_svm'],
         'outer_cv_folds':n_splits,'inner_oof_folds':n_splits,
         'stack_method':['predict_proba','decision_function'],
         'meta_feature_dimensions':6,'meta_classifier':'LogisticRegression',
         'meta_params':META_CONFIGS[best_meta],'passthrough':False})
    both_correct=(lr_pred==val_y)&(svm_pred==val_y)
    both_wrong=(lr_pred!=val_y)&(svm_pred!=val_y)
    lr_only=(lr_pred==val_y)&(svm_pred!=val_y)
    svm_only=(lr_pred!=val_y)&(svm_pred==val_y)
    disagreements=lr_pred!=svm_pred
    diversity={'both_correct':int(both_correct.sum()),'both_wrong':int(both_wrong.sum()),
               'lr_only_correct':int(lr_only.sum()),'svm_only_correct':int(svm_only.sum()),
               'prediction_disagreements':int(disagreements.sum()),
               'disagreement_rate':float(disagreements.mean()),
               'actual_class_distribution_on_disagreement':
               {label:int(((val_y==label)&disagreements).sum()) for label in LABELS}}
    report={'status':'EXPERIMENTAL DEVELOPMENT MODELS — NO NEW TEST EVALUATION',
            'cv':{'outer_folds':n_splits,'inner_oof_folds':n_splits,
                  'shuffle':True,'random_state':RANDOM_STATE,'scoring':'f1_macro',
                  'source':'TRAIN only','estimated_base_fits':44,
                  'cv_seconds':cv_seconds},
            'soft_weight_cv':{str(w):{'mean':float(np.mean(scores)),'std':float(np.std(scores)),
                                      'fold_scores':[float(x) for x in scores]} for w,scores in soft_scores.items()},
            'stack_meta_cv':{str(i):{'params':META_CONFIGS[i],
                                      'mean':float(np.mean(scores)),'std':float(np.std(scores)),
                                      'fold_scores':[float(x) for x in scores]} for i,scores in meta_scores.items()},
            'records':records,'diversity':diversity,
            'official_hashes_verified':official_hashes}
    out=root/'reports'/'experimental'
    out.mkdir(parents=True,exist_ok=True)
    (out/'ensemble_metrics.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    pd.DataFrame([{'model':name,'cv_macro_f1':r['cv_macro_f1'],'cv_std':r['cv_std'],
                   'validation_accuracy':r['validation']['accuracy'],
                   'validation_macro_precision':r['validation']['macro_precision'],
                   'validation_macro_recall':r['validation']['macro_recall'],
                   'validation_macro_f1':r['validation']['macro_f1'],
                   'negative_f1':r['validation']['per_class']['negative']['f1'],
                   'neutral_f1':r['validation']['per_class']['neutral']['f1'],
                   'positive_f1':r['validation']['per_class']['positive']['f1'],
                   'fit_seconds':r['fit_seconds'],'predict_seconds':r['predict_seconds']}
                 for name,r in records.items()]).to_csv(out/'ensemble_comparison.csv',index=False)
    pd.DataFrame([{'lr_weight':w,'svm_weight':1-w,
                   'mean_cv_macro_f1':np.mean(s),'std_cv_macro_f1':np.std(s),
                   'fold_scores':json.dumps(s)}
                 for w,s in soft_scores.items()]).to_csv(out/'soft_voting_weights.csv',index=False)
    pd.DataFrame([{'params':json.dumps(META_CONFIGS[i]),'mean_cv_macro_f1':np.mean(s),
                   'std_cv_macro_f1':np.std(s),'fold_scores':json.dumps(s)}
                 for i,s in meta_scores.items()]).to_csv(out/'stacking_meta_search.csv',index=False)
    (out/'model_disagreement.md').write_text(
        '# TRAIN-fitted baseline disagreement on VALIDATION\n\n'
        f"Both correct: {diversity['both_correct']}; both wrong: {diversity['both_wrong']}; "
        f"LR only correct: {diversity['lr_only_correct']}; SVM only correct: {diversity['svm_only_correct']}.\n\n"
        f"Prediction disagreements: {diversity['prediction_disagreements']} of {len(val_y)} "
        f"({diversity['disagreement_rate']:.3%}). Actual classes on disagreement: "
        f"{diversity['actual_class_distribution_on_disagreement']}.\n\n"
        'Counts are aggregate; no post text or TEST information is included.\n',encoding='utf-8')
    best_ensemble=max(('soft_vote','stacking'),key=lambda name:records[name]['validation']['macro_f1'])
    bootstrap=paired_bootstrap(val_y,prediction_map[best_ensemble],lr_pred)
    (out/'bootstrap_comparison.json').write_text(json.dumps({'candidate':best_ensemble,
        'baseline':'logistic_regression',**bootstrap},indent=2)+'\n',encoding='utf-8')
    ci=bootstrap['ci_95_percentile']
    (out/'bootstrap_comparison.md').write_text(
        f"# Paired VALIDATION comparison\n\nBest ensemble: {best_ensemble}. "
        f"Macro-F1 minus frozen LR: {bootstrap['difference_candidate_minus_lr']:+.4f}. "
        f"Seeded paired-bootstrap 95% percentile interval: [{ci[0]:+.4f}, {ci[1]:+.4f}] "
        f"over {bootstrap['draws']} draws. This describes sampling sensitivity on this "
        'development holdout and is not external evidence or formal proof.\n',encoding='utf-8')
    if (bootstrap['difference_candidate_minus_lr'] >= 0.01 and ci[0] > 0):
        model=soft_model if best_ensemble=='soft_vote' else stack_model
        path=root/'models'/'experimental'/'best_ensemble.joblib'
        path.parent.mkdir(parents=True,exist_ok=True)
        joblib.dump(model,path,compress=3)
        if not np.array_equal(joblib.load(path).predict(val_text),prediction_map[best_ensemble]):
            raise AssertionError('Experimental ensemble reload differs')
        (path.parent/'best_ensemble_metadata.json').write_text(json.dumps({
            'status':'EXPERIMENTAL DEVELOPMENT MODEL — NOT OFFICIAL',
            'model':best_ensemble,'validation_macro_f1':records[best_ensemble]['validation']['macro_f1'],
            'artifact_sha256':sha256_file(path),'artifact_bytes':path.stat().st_size,
            'test_policy':'TEST not loaded or predicted'},indent=2)+'\n',encoding='utf-8')
    assert_official_unchanged(root)
    print('Ensemble validation macro-F1:',
          {name:round(r['validation']['macro_f1'],4) for name,r in records.items()},flush=True)
    return report

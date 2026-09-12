"""TRAIN-only stratified CV and validation-only finalist checkpoint."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
import importlib.metadata

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from .config import PROJECT_ROOT, RANDOM_STATE, LABELS
from .model_pipeline import load_development_data
from .preprocessing import DEFAULT_CONFIG
from .features import make_features, feature_parameters
from .models import make_model
from .evaluation import evaluate_validation

C_VALUES = (0.25, 0.5, 1.0, 2.0, 4.0)
CLASS_WEIGHTS = (None, 'balanced')
FEATURE_VARIATIONS = (
    {'features__transformer_weights': {'word': 1.0, 'character': 0.75}},
    {'features__transformer_weights': {'word': 1.0, 'character': 1.25}},
    {'features__word__min_df': 3},
    {'features__character__ngram_range': (3, 6)},
)
MODEL_NAMES = ('logistic_regression', 'linear_svm')


def cv_strategy(n_splits=3):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def build_pipeline(model_name, memory=None):
    if model_name not in MODEL_NAMES:
        raise ValueError('Only the two leading linear models may be tuned')
    return Pipeline([('features', make_features('combined')),
                     ('classifier', make_model(model_name))], memory=memory)


def paired_bootstrap(y_true, lr_pred, svm_pred, draws=2000, seed=RANDOM_STATE):
    """Paired row resampling; returns LR minus SVM validation macro-F1."""
    y_true, lr_pred, svm_pred = map(np.asarray, (y_true, lr_pred, svm_pred))
    if not (len(y_true) == len(lr_pred) == len(svm_pred)) or len(y_true) == 0:
        raise ValueError('Paired nonempty validation arrays required')
    rng = np.random.default_rng(seed)
    deltas = np.empty(draws)
    for i in range(draws):
        ix = rng.integers(0, len(y_true), len(y_true))
        deltas[i] = (f1_score(y_true[ix], lr_pred[ix], labels=LABELS, average='macro', zero_division=0)
                     - f1_score(y_true[ix], svm_pred[ix], labels=LABELS, average='macro', zero_division=0))
    return {'point_difference': float(f1_score(y_true, lr_pred, labels=LABELS, average='macro', zero_division=0)
                                      - f1_score(y_true, svm_pred, labels=LABELS, average='macro', zero_division=0)),
            'ci_95_percentile': [float(x) for x in np.quantile(deltas, [.025, .975])],
            'bootstrap_probability_lr_higher': float((deltas > 0).mean()),
            'draws': draws, 'seed': seed}


def _score_grid(name, stage, pipeline, grid, text, labels, cv):
    search = GridSearchCV(pipeline, grid, scoring='f1_macro', cv=cv, n_jobs=1,
                          refit=False, return_train_score=False, error_score='raise')
    search.fit(text, labels)
    frame = pd.DataFrame(search.cv_results_)
    rows = []
    for _, row in frame.iterrows():
        rows.append({'model': name, 'stage': stage, 'params': json.dumps(row['params'], sort_keys=True),
                     'mean_cv_macro_f1': float(row['mean_test_score']),
                     'std_cv_macro_f1': float(row['std_test_score']),
                     'rank_within_stage': int(row['rank_test_score']),
                     'mean_fit_seconds': float(row['mean_fit_time']),
                     'mean_score_seconds': float(row['mean_score_time']),
                     'fold_scores': json.dumps([float(row[f'split{i}_test_score']) for i in range(cv.n_splits)])})
    best = max(rows, key=lambda row: (row['mean_cv_macro_f1'], -row['std_cv_macro_f1']))
    return rows, json.loads(best['params']), best


def _cue_counts(text, y_true, prediction):
    import re
    values = np.asarray(text, dtype=str)
    truth, predicted = np.asarray(y_true), np.asarray(prediction)
    error = truth != predicted
    cues = {
        'negation': [bool(re.search(r"\b(?:not|no|never|don't|can't|won't|isn't)\b", x)) for x in values],
        'contraction': [bool(re.search(r"\b\w+'\w+\b", x)) for x in values],
        'emoticon': [bool(re.search(r"(?::|;|=)[-']?[)(DPp/]", x)) for x in values],
        'punctuation': [('!' in x or '?' in x) for x in values],
        'short_5_words': [len(x.split()) <= 5 for x in values],
    }
    result = {}
    for cue, present in cues.items():
        mask = np.asarray(present)
        result[cue] = {'rows': int(mask.sum()), 'errors': int((mask & error).sum()),
                       'error_rate': float((mask & error).sum()/mask.sum()) if mask.sum() else None}
    return result


def _margin_summary(pipeline, text, prediction, truth, name):
    classifier = pipeline.named_steps['classifier']
    features = pipeline.named_steps['features'].transform(text)
    if name == 'logistic_regression':
        values = classifier.predict_proba(features).max(axis=1)
        label = 'maximum predicted probability'
    else:
        scores = classifier.decision_function(features)
        top = np.sort(scores, axis=1)
        values = top[:, -1] - top[:, -2]
        label = 'top-versus-second decision margin'
    correct = np.asarray(prediction) == np.asarray(truth)
    threshold = float(np.median(values))
    return {'measure': label, 'median_correct': float(np.median(values[correct])),
            'median_incorrect': float(np.median(values[~correct])),
            'median_all': threshold,
            'error_rate_below_median': float((~correct[values < threshold]).mean()),
            'error_rate_at_or_above_median': float((~correct[values >= threshold]).mean()),
            'incorrect_above_median': int((~correct & (values >= threshold)).sum())}


def run_tuning(root=PROJECT_ROOT, n_splits=3):
    root = Path(root)
    train, validation, test_integrity = load_development_data(root)
    train_text = train.clean_text.tolist()
    train_y = train.sentiment.to_numpy()
    validation_text = validation.clean_text.tolist()
    validation_y = validation.sentiment.to_numpy()
    cv = cv_strategy(n_splits)
    reports = root/'reports'
    model_dir = root/'models'/'candidates'
    cache_dir = root/'data'/'processed'/'.cv_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_rows, chosen, cv_best = [], {}, {}
    for name in MODEL_NAMES:
        print(f'Tuning {name}: 10 classifier configs x {n_splits} TRAIN folds', flush=True)
        pipeline = build_pipeline(name, memory=str(cache_dir))
        model_grid = {'classifier__C': C_VALUES, 'classifier__class_weight': CLASS_WEIGHTS}
        rows, params, best = _score_grid(name, 'classifier', pipeline, model_grid,
                                         train_text, train_y, cv)
        all_rows.extend(rows)
        base_params = params.copy()
        print(f'{name} classifier best {best["mean_cv_macro_f1"]:.4f} {params}', flush=True)
        # A small, controlled feature stage around each classifier's best TRAIN-CV setting.
        feature_grid = [{key: [value] for key, value in dict(base_params, **variation).items()} for variation in ({}, *FEATURE_VARIATIONS)]
        rows, feature_params, feature_best = _score_grid(name, 'features', pipeline,
                                                          feature_grid, train_text, train_y, cv)
        all_rows.extend(rows)
        winner = max((best, feature_best), key=lambda row: (row['mean_cv_macro_f1'], -row['std_cv_macro_f1']))
        chosen[name] = json.loads(winner['params'])
        cv_best[name] = winner
        print(f'{name} overall best {winner["mean_cv_macro_f1"]:.4f} {chosen[name]}', flush=True)
        pd.DataFrame(all_rows).to_csv(reports/'tuning_results.csv', index=False)
    # VALIDATION is first used here, after all TRAIN CV decisions are fixed.
    finalists = {}
    for name in MODEL_NAMES:
        pipe = build_pipeline(name)
        pipe.set_params(**chosen[name])
        start = perf_counter()
        pipe.fit(train_text, train_y)
        fit_seconds = perf_counter()-start
        start = perf_counter()
        predictions = pipe.predict(validation_text)
        prediction_seconds = perf_counter()-start
        metrics = evaluate_validation(validation_y, predictions)
        finalists[name] = {'pipeline': pipe, 'predictions': predictions, 'metrics': metrics,
                           'fit_seconds': fit_seconds, 'prediction_seconds': prediction_seconds,
                           'cv': cv_best[name], 'params': chosen[name],
                           'cues': _cue_counts(validation_text, validation_y, predictions),
                           'margin': _margin_summary(pipe, validation_text, predictions, validation_y, name)}
        print(f'VALIDATION {name} macro-F1 {metrics["macro_f1"]:.4f}', flush=True)
    uncertainty = paired_bootstrap(validation_y, finalists['logistic_regression']['predictions'],
                                   finalists['linear_svm']['predictions'])
    selection = max(MODEL_NAMES, key=lambda name: (finalists[name]['metrics']['macro_f1'],
                        finalists[name]['cv']['mean_cv_macro_f1'],
                        finalists[name]['metrics']['per_class']['negative']['f1'],
                        -finalists[name]['fit_seconds']))
    winner = finalists[selection]
    artifact = model_dir/'frozen_candidate.joblib'
    joblib.dump(winner['pipeline'], artifact, compress=3)
    sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    reloaded = joblib.load(artifact)
    reload_match = bool(np.array_equal(reloaded.predict(validation_text), winner['predictions']))
    if not reload_match:
        raise AssertionError('Frozen candidate reload predictions differ')
    def public_record(name):
        v=finalists[name]
        return {'cv': v['cv'], 'params': v['params'], 'validation': v['metrics'],
                'fit_seconds': v['fit_seconds'], 'prediction_seconds': v['prediction_seconds'],
                'error_cues': v['cues'], 'confidence_or_margin': v['margin']}
    result = {'status': 'FROZEN DEVELOPMENT CANDIDATE — TEST NOT EVALUATED',
              'selected_model': selection, 'finalists': {name:public_record(name) for name in MODEL_NAMES},
              'paired_bootstrap_lr_minus_svm': uncertainty,
              'test_integrity_only': test_integrity,
              'cv': {'type':'StratifiedKFold','n_splits':n_splits,'shuffle':True,
                     'random_state':RANDOM_STATE,'scoring':'f1_macro','rows':'TRAIN only',
                     'feature_fitting':'inside each Pipeline fold','configurations':len(all_rows)},
              'environment': {key:importlib.metadata.version(key) for key in ('pandas','numpy','scipy','scikit-learn','joblib')},
              'python_version':platform.python_version()}
    (reports/'tuning_metrics.json').write_text(json.dumps(result,indent=2,default=str)+'\n',encoding='utf-8')
    metadata = {'status':result['status'],'selected_model':selection,
                'preprocessing':asdict(DEFAULT_CONFIG),
                'features':feature_parameters(winner['pipeline'].named_steps['features']),
                'feature_weights':winner['pipeline'].named_steps['features'].transformer_weights or {'word':1.0,'character':1.0},
                'classifier_parameters':winner['pipeline'].named_steps['classifier'].get_params(deep=False),
                'selection_metric':'validation macro-F1 after TRAIN-only CV',
                'cv':result['cv'],'random_state':RANDOM_STATE,
                'validation_metrics':winner['metrics'],
                'artifact_sha256':sha,'artifact_bytes':artifact.stat().st_size,
                'reload_predictions_match':reload_match,'environment':result['environment'],
                'python_version':result['python_version']}
    (model_dir/'frozen_candidate_metadata.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n',encoding='utf-8')
    lines=['# Focused TRAIN-only tuning','',
           f"{n_splits}-fold stratified CV (shuffle, seed 42), macro-F1 scoring. Three folds limit runtime and memory for 216,107-dimensional combined sparse features. TF-IDF and IDF are fitted inside each training fold by the sklearn Pipeline. VALIDATION is excluded from the search; TEST has no predictive use.",
           '',f"{len(all_rows)} configurations: 10 classifier settings and 5 focused feature settings per model, each run on {n_splits} folds. CV executes serially to limit memory duplication.",
           '','## TRAIN CV winners','']
    for name in MODEL_NAMES:
        v=finalists[name]
        lines += [f"- {name}: mean macro-F1 {v['cv']['mean_cv_macro_f1']:.4f} ± {v['cv']['std_cv_macro_f1']:.4f}; parameters {v['params']}."]
    lines += ['','## VALIDATION checkpoint','',
              '| Model | Accuracy | Macro precision | Macro recall | Macro-F1 | Negative F1 | Neutral F1 | Positive F1 | Fit s | Predict s |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in MODEL_NAMES:
        v=finalists[name]; m=v['metrics']
        lines.append(f"| {name} | {m['accuracy']:.4f} | {m['macro_precision']:.4f} | {m['macro_recall']:.4f} | {m['macro_f1']:.4f} | {m['per_class']['negative']['f1']:.4f} | {m['per_class']['neutral']['f1']:.4f} | {m['per_class']['positive']['f1']:.4f} | {v['fit_seconds']:.1f} | {v['prediction_seconds']:.3f} |")
    lines += ['',f"Selected: **{selection}** by VALIDATION macro-F1, considering CV stability, negative F1 and runtime as secondary evidence.",
              'This is a frozen development candidate; TEST has not been evaluated. The paired bootstrap interval in model_uncertainty.md describes remaining selection uncertainty.',
              'Full parameters, scores, per-class metrics and versions are in tuning_metrics.json and frozen_candidate_metadata.json.','']
    (reports/'tuning_summary.md').write_text('\n'.join(lines),encoding='utf-8')
    ci=uncertainty['ci_95_percentile']
    (reports/'model_uncertainty.md').write_text(
        '# Paired VALIDATION uncertainty\n\n'
        f"LR minus LinearSVC macro-F1: {uncertainty['point_difference']:+.4f}. "
        f"Paired row-bootstrap percentile 95% interval: [{ci[0]:+.4f}, {ci[1]:+.4f}]. "
        f"Fraction of {uncertainty['draws']} seeded draws favoring LR: {uncertainty['bootstrap_probability_lr_higher']:.3f}.\n\n"
        'Each resample draws the same VALIDATION row indices for both fixed finalist predictions. '
        'This describes sampling sensitivity on this holdout, not formal proof, independent external validation or a correction for development selection. '
        'An interval crossing zero means the difference is too small to distinguish confidently by this diagnostic.\n',
        encoding='utf-8')
    error_lines=['# Finalist VALIDATION errors','',
                 'Rows are true labels and columns predictions in negative, neutral, positive order. Cue associations are descriptive, not causal. TEST is locked.','']
    for name in MODEL_NAMES:
        v=finalists[name]; m=v['metrics']; cm=m['confusion_matrix']
        error_lines += [f'## {name}','',f"Total errors: {int((v['predictions'] != validation_y).sum())}.",
                        f"Confusion matrix: {cm}.",'',
                        '| Cue | Rows | Errors | Error rate |','|---|---:|---:|---:|']
        for cue, item in v['cues'].items():
            rate='n/a' if item['error_rate'] is None else f"{item['error_rate']:.3f}"
            error_lines.append(f"| {cue} | {item['rows']} | {item['errors']} | {rate} |")
        error_lines.append('')
    error_lines += ['## Off-diagonal comparison','',
                    '| True → predicted | Logistic Regression | LinearSVC |',
                    '|---|---:|---:|']
    lr_cm = finalists['logistic_regression']['metrics']['confusion_matrix']
    svm_cm = finalists['linear_svm']['metrics']['confusion_matrix']
    for i, actual in enumerate(LABELS):
        for j, predicted in enumerate(LABELS):
            if i != j:
                error_lines.append(f'| {actual} → {predicted} | {lr_cm[i][j]} | {svm_cm[i][j]} |')
    error_lines += ['', 'Cue groups overlap and have different denominators. Compare the cue-specific error rates above; short context, negation, contractions, emoticons and punctuation do not establish causes of error.','']
    (reports/'error_analysis.md').write_text('\n'.join(error_lines),encoding='utf-8')
    margin_lines=['# VALIDATION confidence and decision margins','',
                  'Logistic Regression uses maximum predicted probability. LinearSVC uses the gap between the highest and second-highest decision scores. Their scales are different; SVM margins are not probabilities.','']
    for name in MODEL_NAMES:
        s=finalists[name]['margin']
        margin_lines += [f"## {name}",'',f"Measure: {s['measure']}.",
                         f"Median correct: {s['median_correct']:.4f}; median incorrect: {s['median_incorrect']:.4f}.",
                         f"Error rate below model-specific median: {s['error_rate_below_median']:.4f}; at/above median: {s['error_rate_at_or_above_median']:.4f}.",
                         f"Incorrect predictions at/above median: {s['incorrect_above_median']}.",'']
    margin_lines += ['These are diagnostics on VALIDATION, not calibrated reliability guarantees. High-confidence errors warrant manual review before application use.','']
    (reports/'model_confidence.md').write_text('\n'.join(margin_lines),encoding='utf-8')
    print(f'Frozen {selection}, sha256 {sha}, reload exact: {reload_match}', flush=True)
    return result


if __name__ == '__main__':
    run_tuning()

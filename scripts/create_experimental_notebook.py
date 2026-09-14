"""Generate and execute only the read-only experimental results notebook."""
import json
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT
from src.experimental.ensemble import assert_official_unchanged


def create_and_execute(root=PROJECT_ROOT):
    root = Path(root)
    assert_official_unchanged(root)
    reports = root/'reports/experimental'
    ensemble = json.loads((reports/'ensemble_metrics.json').read_text(encoding='utf-8'))
    xgb = json.loads((reports/'xgboost_metrics.json').read_text(encoding='utf-8'))
    bootstrap = json.loads((reports/'bootstrap_comparison.json').read_text(encoding='utf-8'))
    diversity = ensemble['diversity']
    soft = ensemble['records']['soft_vote']
    stack = ensemble['records']['stacking']
    best = xgb['search'][xgb['selected_candidate']]
    cells = [
        nbformat.v4.new_markdown_cell(
            '# Post-benchmark ensemble and XGBoost research\n\n'
            'The official TweetEval TEST benchmark was evaluated once and is frozen. '
            'This separate experiment uses TRAIN for all fitting and cross-validation '
            'and VALIDATION only for development comparison. Reusing TEST would turn '
            'the completed benchmark into a tuning set; this notebook contains no '
            'TEST prediction or evaluation code.'),
        nbformat.v4.new_markdown_cell(
            '## Motivation and design\n\n'
            f"Frozen LR and LinearSVC disagree on {diversity['prediction_disagreements']} "
            f"of 2,000 VALIDATION rows ({diversity['disagreement_rate']:.2%}). "
            f"LR alone is correct on {diversity['lr_only_correct']}; SVM alone on "
            f"{diversity['svm_only_correct']}. Their different errors motivate "
            'combination, but a two-model hard vote ties whenever they disagree. '
            'We explicitly resolve ties to LR, making hard voting identical to LR.\n\n'
            'For soft voting, LinearSVC margins are not probabilities. Sigmoid '
            'CalibratedClassifierCV is fitted within TRAIN folds before mixing '
            'its probabilities with LR probabilities. Five weights are selected '
            'using three-fold stratified TRAIN CV. Calibration adds fitting cost '
            'and does not by itself guarantee better predictions.\n\n'
            'For stacking, each meta-training row receives LR probabilities and '
            'SVM decision scores from base models that did not fit that row. '
            'Inner OOF fitting is nested inside three outer TRAIN folds for '
            'meta-parameter selection. The six meta-features feed a simple '
            'LogisticRegression classifier; full TF-IDF does not pass through.'),
        nbformat.v4.new_markdown_cell(
            '## Sparse XGBoost design\n\n'
            'XGBoost uses the same combined word (1,2) and character (3,5) '
            'TF-IDF representation. Vocabulary and IDF are fitted separately '
            'inside each TRAIN fold. All matrices stay scipy CSR sparse; '
            'the full feature matrix is never densified. Multiclass '
            'multi:softprob with mlogloss evaluates ten selected tree and '
            'sample-weight settings (30 fold fits), with macro-F1 as the '
            'selection score. Balanced weights, when used, are calculated '
            'from each fold-training label distribution.'),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "from IPython.display import display\n"
            "root = Path.cwd()\n"
            "reports = root / 'reports' / 'experimental'\n"
            "table = pd.read_csv(reports / 'ensemble_comparison.csv')\n"
            "display(table.round(4))"),
        nbformat.v4.new_markdown_cell(
            '## Selected configurations and diagnostics\n\n'
            f"Soft voting selected LR weight {soft['details']['lr_weight']:.2f} "
            f"and SVM weight {soft['details']['svm_weight']:.2f}. Its TRAIN-CV "
            f"macro-F1 was {soft['cv_macro_f1']:.4f}; VALIDATION macro-F1 "
            f"was {soft['validation']['macro_f1']:.4f}.\n\n"
            f"Stacking selected {stack['details']['meta_params']}; "
            f"TRAIN-CV macro-F1 {stack['cv_macro_f1']:.4f}; VALIDATION "
            f"macro-F1 {stack['validation']['macro_f1']:.4f}.\n\n"
            f"XGBoost candidate {xgb['selected_candidate']+1}/10 selected "
            f"{best['params']}; TRAIN-CV macro-F1 {best['mean_cv_macro_f1']:.4f}; "
            f"VALIDATION macro-F1 {xgb['record']['validation']['macro_f1']:.4f}. "
            'Tree feature importance is not interpreted causally.'),
        nbformat.v4.new_code_cell(
            "weights = pd.read_csv(reports / 'soft_voting_weights.csv')\n"
            "stacking = pd.read_csv(reports / 'stacking_meta_search.csv')\n"
            "trees = pd.read_csv(reports / 'xgboost_results.csv')\n"
            "display(weights.round(4))\n"
            "display(stacking.round(4))\n"
            "display(trees.sort_values('mean_cv_macro_f1', ascending=False).head(5).round(4))"),
        nbformat.v4.new_markdown_cell(
            '## Holdout uncertainty and next decision\n\n'
            f"The best experimental model for this comparison is "
            f"{bootstrap['candidate']}. Its paired VALIDATION macro-F1 "
            f"difference from frozen LR is "
            f"{bootstrap['difference_candidate_minus_lr']:+.4f}, with a "
            f"seeded {bootstrap['draws']}-draw 95% percentile interval "
            f"[{bootstrap['ci_95_percentile'][0]:+.4f}, "
            f"{bootstrap['ci_95_percentile'][1]:+.4f}]. This describes "
            'sampling sensitivity on the same development holdout, not '
            'independent external evidence. Any promising model should '
            'be tested next on a newly collected, manually labeled '
            'brand-comment dataset. The official TweetEval TEST split '
            'remains closed.'),
    ]
    nb = nbformat.v4.new_notebook(cells=cells,
        metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                 'name': 'python3'}})
    path = root/'notebooks/05_ensemble_experiments.ipynb'
    nbformat.validate(nb)
    nbformat.write(nb, path)
    local_tmp = root/'.pytest_tmp'
    local_tmp.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=local_tmp) as temporary:
        os.environ['IPYTHONDIR'] = str(Path(temporary)/'ipython')
        kernel_dir = Path(temporary)/'experimental'
        kernel_dir.mkdir()
        (kernel_dir/'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f',
                     '{connection_file}'],
            'display_name': 'Experimental results', 'language': 'python'}),
            encoding='utf-8')
        manager = KernelManager(
            kernel_name='experimental',
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary]))
        try:
            NotebookClient(nb, km=manager, timeout=180,
                           resources={'metadata': {'path': str(root)}}).execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
    nbformat.validate(nb)
    nbformat.write(nb, path)
    record = {'notebook': path.name, 'status': 'passed',
              'executed_code_cells': sum(c.cell_type == 'code' for c in nb.cells),
              'data_scope': 'persisted TRAIN-CV and VALIDATION aggregate reports only'}
    (reports/'notebook_execution.json').write_text(
        json.dumps(record, indent=2)+'\n', encoding='utf-8')
    assert_official_unchanged(root)
    return record


if __name__ == '__main__':
    print(create_and_execute())

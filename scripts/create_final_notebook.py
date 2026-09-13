"""Generate a read-only presentation of the persisted official TEST result."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []

def md(value):
    cells.append(nbf.v4.new_markdown_cell(value))

def code(value):
    cells.append(nbf.v4.new_code_cell(value))

md("""# One official locked-TEST evaluation

The final procedure was frozen, committed at 7922527d9599567887de2a5836e84a4e7e89458e, pushed, and CI-passing before TEST was unlocked. The exact frozen combined TF-IDF plus balanced Logistic Regression pipeline was freshly fitted on all TRAIN+VALIDATION rows (47,615). One official metric set was produced on the untouched 12,284-row TEST split. This notebook reads persisted aggregate results only: it never loads TEST CSVs, a model artifact, or row-level predictions, and it does not run the evaluation pipeline.

Macro-F1 was fixed as the primary metric before TEST. The observed result is accepted without tuning or comparing a second TEST model.""")

code("""from pathlib import Path
import json
import hashlib
import pandas as pd
from IPython.display import display, Image, Markdown

root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'reports' / 'final_test_metrics.json').is_file())
metrics_path = root / 'reports' / 'final_test_metrics.json'
payload = json.loads(metrics_path.read_text(encoding='utf-8'))
manifest = json.loads((root / 'reports' / 'final_evaluation_manifest.json').read_text(encoding='utf-8'))
assert hashlib.sha256(metrics_path.read_bytes()).hexdigest() == manifest['final_metrics_sha256']
assert payload['pretest_commit_sha'] == manifest['pretest_commit_sha']
assert manifest['evaluation_completed']
display({'pretest_commit': payload['pretest_commit_sha'],
         'train_rows': payload['train_rows'],
         'validation_rows': payload['validation_rows'],
         'development_rows': payload['development_rows'],
         'test_rows': payload['test_rows'],
         'official_prediction_vectors': manifest['official_prediction_vectors']})""")

md("""## Official TEST metrics

The three classes are ordered negative, neutral, positive. Per-class support and the confusion matrix cells sum to 12,284. These are benchmark results for historical tweets, not an estimate for current brand-specific comments.""")

code("""m = payload['test_metrics']
display(pd.DataFrame([{'metric': key, 'value': m[key]} for key in
                      ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1')]))
display(pd.DataFrame([{'sentiment': label, **m['per_class'][label]} for label in
                      ('negative', 'neutral', 'positive')]))
display(Image(filename=str(root / 'reports' / 'figures' / 'final_test_confusion_matrix.png')))
display({'confusion_matrix': m['confusion_matrix']})""")

md("""## VALIDATION versus TEST

VALIDATION measured the TRAIN-only development candidate. TEST measured the identical frozen configuration after the authorized TRAIN+VALIDATION refit. Differences are descriptive. The lower TEST macro-F1 and positive-class F1 do not trigger a model change.""")

code("""v = payload['frozen_validation_metrics']
rows = []
for title, key, label in [('Accuracy', 'accuracy', None), ('Macro-F1', 'macro_f1', None),
                          ('Negative F1', 'f1', 'negative'), ('Neutral F1', 'f1', 'neutral'),
                          ('Positive F1', 'f1', 'positive')]:
    before = v[key] if label is None else v['per_class'][label][key]
    after = m[key] if label is None else m['per_class'][label][key]
    rows.append({'metric': title, 'validation': before, 'test': after,
                 'test_minus_validation': after - before})
display(pd.DataFrame(rows).round(4))""")

md("""## Procedure, safeguards and limitations

The manifest records the pre-TEST SHA, the final model and metrics checksums, and the processed TEST checksum. The final saved estimator was reloaded and its predictions matched the original vector; that technical check was not scored a second time. The default evaluation command only points to the persisted result, and the manifest blocks another official evaluation. No raw TEST examples or row-level predictions are published.

Historical English tweets, sarcasm, missing conversational context, ambiguous annotation, class imbalance and modern brand-domain shift remain. The model is not a substitute for human moderation or psychological inference. Next work concerns a reusable inference service and separate real-world validation, not retuning on this TEST split.""")

code("""display(Markdown((root / 'reports' / 'final_evaluation_summary.md').read_text(encoding='utf-8')))
display(Markdown((root / 'models' / 'final' / 'MODEL_CARD.md').read_text(encoding='utf-8')))""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
path = ROOT / 'notebooks' / '04_final_evaluation.ipynb'
nbf.write(notebook, path)
print(path)

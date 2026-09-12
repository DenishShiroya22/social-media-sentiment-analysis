"""Generate the modeling notebook from reusable code and executed reports."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(value):
    cells.append(nbf.v4.new_markdown_cell(value))
def code(value):
    cells.append(nbf.v4.new_code_cell(value))

md("""# Classical sentiment model development

Objective: compare TRAIN-fitted TF-IDF representations and classical classifiers on the official TweetEval VALIDATION split. Macro-F1 is primary because TRAIN is imbalanced (negative 7,093; neutral 20,673; positive 17,849). TEST remains locked; this notebook never loads TEST labels or predictions. Dataset provenance and data-quality evidence are in the data-foundation reports.

TF-IDF is term frequency times inverse document frequency: it weights a term within a post and downweights terms common across many posts. For *not good*, unigrams are *not* and *good*, while the bigram *not good* retains the phrase. Character n-grams can capture contraction fragments, repeated letters and punctuation. The word token pattern keeps contractions such as *don't* together. No vocabulary or IDF was fitted on VALIDATION.""")
code("""from pathlib import Path
import sys
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'src' / 'config.py').is_file())
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
import json
import pandas as pd
from IPython.display import display, Image, Markdown
from src.features import make_features, WORD_TOKEN_PATTERN
from src.evaluation import select_winner
from src.config import LABELS
comparison = pd.read_csv(root / 'reports' / 'model_comparison.csv')
metrics = json.loads((root / 'reports' / 'model_metrics.json').read_text(encoding='utf-8'))
display(comparison[['experiment_id','feature','model','validation_accuracy','validation_macro_f1','fit_seconds']])""")
md("""## Controlled experiments

Nine full-TRAIN runs cover word unigrams, word unigrams+bigrams, character 3–5 grams and combined word+character TF-IDF. The algorithms are Logistic Regression, LinearSVC, MultinomialNB and a bounded Random Forest. The same official TRAIN and VALIDATION rows are used throughout. Word trigrams and exhaustive search are deferred to focused tuning because the current matrix already tests the main representation differences with manageable memory.

TRAIN fits vectorizers and classifiers; VALIDATION is only transformed and scored. The ranked table and JSON below were produced by the executable pipeline, not recomputed from TEST.""")
code("""from src.models import EXPERIMENTS
display(pd.DataFrame([vars(e) for e in EXPERIMENTS]))
display({'winner': metrics['winner']['experiment_id'],
         'macro_f1': metrics['winner']['validation_macro_f1'],
         'accuracy': metrics['winner']['validation_accuracy']})
display({'word_pattern': WORD_TOKEN_PATTERN,
         'tokenization_demo': __import__('re').findall(WORD_TOKEN_PATTERN, "don't can't isn't won't not good")})""")
md("""## Validation figures

Confusion rows are true labels, columns are predictions, ordered negative, neutral, positive. Only distinct leading candidates get confusion figures.""")
code("""figures = root / 'reports' / 'figures' / 'modeling'
display(Image(filename=str(figures / 'validation_macro_f1.png')))
display(Image(filename=str(figures / 'validation_per_class_f1.png')))
winner_id = metrics['winner']['experiment_id']
display(Image(filename=str(figures / f'validation_confusion_{winner_id}.png')))
display({'classes': LABELS,
         'winner_confusion': metrics['diagnostics'][winner_id]['confusion_matrix']})""")
md("""## Validation errors and TRAIN-learned features

Small, ordered validation error samples indicate hypotheses, not generalizable causes. Linear coefficient rankings use TRAIN-fitted weights; they do not prove causation. No TEST examples are inspected.""")
code("""display(Markdown((root / 'reports' / 'error_analysis.md').read_text(encoding='utf-8')))
for label, features in metrics['linear_features'][winner_id].items():
    display({label: [item['feature'] for item in features[:10]]})""")
md("""## Baseline candidate before focused tuning

The winning artifact is a VALIDATION-SELECTED CANDIDATE, not a final test-evaluated model. It is persisted as a fitted sklearn Pipeline and its reloaded VALIDATION predictions were checked for exact equality. The bounded Random Forest remains a full-TRAIN baseline; the report records its runtime and weakness on the minority negative class. The focused TRAIN-CV tuning and frozen development candidate appear below. Final TEST evaluation is a separate future task. Historical English tweets, sarcasm, context loss and brand-domain shift remain limitations.""")
code("""metadata = json.loads((root / 'models' / 'candidates' / 'best_candidate_metadata.json').read_text(encoding='utf-8'))
display(metadata)
assert metadata['reload_predictions_match']
assert 'test_accuracy' not in json.dumps(metrics).lower()""")

md("""## Corrected escaped punctuation

The source contains literal escaped punctuation in TRAIN and VALIDATION, notably the strings for a comma and curly apostrophe. The preprocessing step now maps only known punctuation escapes before NFC normalization. Existing Unicode, emoji, and unrelated backslashes are preserved. The full before/after baseline comparison is below; TEST content was not explored.""")
code("""impact = json.loads((root / 'reports' / 'preprocessing_impact.json').read_text(encoding='utf-8'))
display({'affected_rows': impact['affected_raw_rows'],
         'learned_features': impact['learned_features']})
display(pd.DataFrame(impact['after'])[['experiment_id','old_macro_f1','new_macro_f1','delta_macro_f1','new_accuracy']])""")
md("""## Focused TRAIN-only cross-validation

The corrected baseline retains combined TF-IDF Logistic Regression and LinearSVC as the two leading linear candidates. Each uses three stratified TRAIN folds, seed 42, and macro-F1 scoring. A sklearn Pipeline includes both FeatureUnion and classifier, so vocabulary and IDF are refitted within each fold. VALIDATION is excluded from the grid search; TEST is locked. Three folds and serial execution contain memory and runtime for sparse matrices with over 200,000 columns. The staged search covers five C values and two class weights per model, then five controlled feature settings near the best classifier configuration.""")
code("""tuning = json.loads((root / 'reports' / 'tuning_metrics.json').read_text(encoding='utf-8'))
search = pd.read_csv(root / 'reports' / 'tuning_results.csv')
display(tuning['cv'])
display(search.sort_values(['model','stage','rank_within_stage']).groupby(['model','stage']).head(3)[['model','stage','params','mean_cv_macro_f1','std_cv_macro_f1','mean_fit_seconds']])""")
md("""## Final VALIDATION checkpoint and uncertainty

Both finalists were fitted on all TRAIN after their CV configurations were fixed. They were then evaluated once on VALIDATION. A paired bootstrap resamples the same validation rows for both predictions and measures the macro-F1 difference. It describes sensitivity, not formal proof or external generalization.""")
code("""display(Markdown((root / 'reports' / 'tuning_summary.md').read_text(encoding='utf-8')))
display(Markdown((root / 'reports' / 'model_uncertainty.md').read_text(encoding='utf-8')))""")
md("""## Errors, confidence, and frozen development candidate

Error cues are descriptive. Logistic Regression's maximum probability and LinearSVC's top-versus-second decision margin have different scales. The selected candidate is frozen with its preprocessing and feature configuration, validation scores, artifact checksum and exact reload-prediction check. TEST has not been evaluated.""")
code("""display(Markdown((root / 'reports' / 'error_analysis.md').read_text(encoding='utf-8')))
display(Markdown((root / 'reports' / 'model_confidence.md').read_text(encoding='utf-8')))
frozen = json.loads((root / 'models' / 'candidates' / 'frozen_candidate_metadata.json').read_text(encoding='utf-8'))
display({'status':frozen['status'], 'selected_model':frozen['selected_model'],
         'artifact_sha256':frozen['artifact_sha256'],
         'reload_predictions_match':frozen['reload_predictions_match']})
assert frozen['reload_predictions_match']
assert 'TEST NOT EVALUATED' in frozen['status']""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = {'display_name':'Python 3', 'language':'python', 'name':'python3'}
path = ROOT / 'notebooks' / '03_model_development.ipynb'
nbf.write(notebook, path)
print(path)

"""Rebuild the two teaching notebooks from compact, reusable-function examples."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SETUP = '''from pathlib import Path
import sys
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'src' / 'config.py').is_file())
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from src.config import RAW_DATA_DIR, FIGURES_DIR
from src.data_loader import load_csv
from IPython.display import display, Image
train = load_csv(RAW_DATA_DIR / 'tweet_eval_train.csv')'''

def write(name, cells):
    notebook = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(text) if kind == 'md' else nbf.v4.new_code_cell(text) for kind, text in cells])
    notebook.metadata.kernelspec = {'display_name':'Python 3', 'language':'python', 'name':'python3'}
    (ROOT / 'notebooks').mkdir(exist_ok=True)
    nbf.write(notebook, ROOT / 'notebooks' / name)

write('01_data_understanding_eda.ipynb', [
    ('md', '# Data understanding and TRAIN EDA\n\nThe future project monitors brand/product sentiment. Phase 1 establishes a reproducible foundation without model training. TweetEval provides a public English tweet benchmark with official train, validation and test splits.\n\nSource: [Cardiff NLP TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval). Citation: Barbieri et al. (2020), [TweetEval](https://aclanthology.org/2020.findings-emnlp.148/). TRAIN supports development; VALIDATION supports future selection; TEST remains locked. These cells load only TRAIN.'),
    ('code', SETUP),
    ('code', "display({'shape': train.shape, 'columns': list(train.columns)})\ndisplay(train.groupby('sentiment', sort=True).head(2))"),
    ('md', '## Missing values, duplicates and balance\nMechanical checks preserve official rows. Empty strings are checked separately because literal strings such as NA must not become missing data.'),
    ('code', "from src.data_validation import validate_dataframe\nfrom src.eda import analyze_train\ndisplay(validate_dataframe(train, 'train'))\nanalysis = analyze_train(train)\ndisplay({k: analysis[k] for k in ['class_counts', 'class_percentages', 'characters', 'words', 'artifacts']})"),
    ('code', "for filename in ['train_class_distribution.png', 'train_text_lengths.png', 'train_artifacts.png', 'train_top_tokens.png']:\n    display(Image(filename=str(FIGURES_DIR / filename)))"),
    ('md', '## Frequent words and bigrams\nDescriptive counts include stopwords and use a simple regex. These are not TF-IDF features. Artifact counts measure posts containing each artifact, not total occurrences.'),
    ('code', "display(analysis['top_tokens'])\ndisplay(analysis['top_bigrams'])\ndisplay(analysis['tokens_by_sentiment'])\ndisplay({'short': analysis['short_examples'], 'long': analysis['long_examples']})"),
    ('md', '## Observations and Phase 2 risks\nThe executed class counts show imbalance, so future evaluation should include macro-F1 and per-class recall. Mentions and hashtag words occur frequently; raw token counts include user placeholders and common function words. No Unicode emojis were detected in this training snapshot; emoji-preservation behavior is still covered by explicit test examples for future inputs. Short historical tweets lack conversational context. Negation, sarcasm and brand-specific vocabulary remain challenging. No test distribution or examples inform these decisions. See reports/phase1_summary.md for all measured findings.')])

write('02_text_preprocessing.ipynb', [
    ('md', '# Conservative sentiment preprocessing\n\nPreserve original text and sentiment cues; apply one fixed stateless function to each official split. No learned features, stopword filtering, stemming, lemmatization or resplitting. Defaults retain emoji, punctuation, repeated characters and contractions; optional transformations are demonstrated on illustrative examples only.'),
    ('code', SETUP),
    ('code', '''import pandas as pd
from src.preprocessing import (clean_text, preprocess_dataframe, PreprocessingConfig,
    remove_urls, handle_mentions, handle_hashtags, handle_emojis, handle_html,
    normalize_whitespace, normalize_unicode, normalize_repeated_characters)
examples = ["@Apple WTF 😡😡 battery dies in 2 hrs!!! https://abc.com #iPhone",
            "I DON'T like this phone!!!", "Great phone 😍 #AmazingCamera",
            "  Hello\\t world  ", "I can’t recommend this", "sooooo good!!!"]
display(pd.DataFrame({'text': examples, 'clean_text': [clean_text(t) for t in examples]}))'''),
    ('code', '''display({
    'URL': remove_urls('See https://example.com now!'),
    'mention': handle_mentions('Thanks @Apple'),
    'hashtag': handle_hashtags('#AmazingCamera'),
    'emoji_preserve': handle_emojis('😍 😡 ❤️'),
    'emoji_text_optional': handle_emojis('😍', 'demojize'),
    'HTML': handle_html('<b>Good</b> &amp; nice'),
    'whitespace': normalize_whitespace('  hello   world  '),
    'unicode': normalize_unicode('cafe\\u0301'),
    'repeat_optional': normalize_repeated_characters('sooooo good!!!'),
    'negation': clean_text("No, I don't like it. Never again!")})
assert "don't" in clean_text("I DON'T like this phone!!!")
assert '😡' in clean_text(examples[0])'''),
    ('md', '## TRAIN transformation and post-cleaning validation\nThe original frame and optional metadata are preserved. Missing input becomes empty clean text and is flagged by validation; the pipeline blocks unusable rows rather than dropping benchmark records.'),
    ('code', "from src.data_validation import validate_dataframe, require_valid\nfrom src.eda import plot_cleaning_lengths\nprocessed = preprocess_dataframe(train)\nquality = validate_dataframe(processed, 'train', text_column='clean_text')\nrequire_valid(quality)\ndisplay(quality)\ndisplay(processed.head(5))\nplot_cleaning_lengths(processed)\ndisplay(Image(filename=str(FIGURES_DIR / 'train_cleaning_lengths.png')))"),
    ('md', '## Generate all processed splits\nThis uses the same deterministic configuration for every split, verifies original text and label preservation, and reloads outputs. Test integrity is checked mechanically; test examples and class statistics are not displayed.'),
    ('code', "from src.pipeline import run_phase1\nresult = run_phase1(root)\ndisplay({name: {'rows': item['rows'], 'removed_rows': item['removed_rows']} for name, item in result['outputs'].items()})"),
    ('md', '## Phase 2 contract\nLoad train_clean.csv for fitting features/models and validation_clean.csv for selection. Keep test_clean.csv locked until final evaluation. Default TF-IDF tokenization may discard emoji/punctuation and split contractions; design that future tokenizer with TRAIN and validate on VALIDATION. No modeling is implemented here.')])

"""TRAIN-focused descriptive analysis; no learned features or test-label EDA."""
from collections import Counter
from pathlib import Path
import re
import os
from .config import REPORTS_DIR
os.environ.setdefault('MPLCONFIGDIR', str(REPORTS_DIR / '.mpl_cache'))
import emoji
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from .config import LABELS, FIGURES_DIR
from .preprocessing import URL_PATTERN, MENTION_PATTERN, HASHTAG_PATTERN

def descriptive_summary(frame: pd.DataFrame) -> dict:
    """Only training/validation summaries are permitted through this API."""
    if 'split' not in frame or not frame['split'].isin(['train', 'validation']).all():
        raise ValueError('EDA accepts only train/validation data; test is locked')
    text = frame['text'].fillna('')
    distribution = frame['sentiment'].value_counts().reindex(LABELS, fill_value=0)
    return {'rows': len(frame), 'columns': list(frame.columns),
            'missing_values': frame.isna().sum().to_dict(),
            'duplicate_rows': int(frame.duplicated().sum()),
            'duplicate_text': int(text.duplicated().sum()),
            'class_counts': distribution.to_dict(),
            'class_percentages': (100 * distribution / len(frame)).round(3).to_dict(),
            'split_counts': frame['split'].value_counts().to_dict(),
            'sentiment_by_split': pd.crosstab(frame['split'], frame['sentiment']).to_dict(),
            'characters': text.str.len().describe().to_dict(),
            'words': text.str.split().str.len().describe().to_dict()}

def frequent_terms(texts: pd.Series, n: int = 1, limit: int = 20) -> list:
    counts = Counter()
    for text in texts:
        tokens = re.findall(r"\b\w+(?:['’]\w+)?\b", text.lower())
        counts.update(' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    return counts.most_common(limit)

def analyze_train(frame: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> dict:
    if not frame['split'].eq('train').all():
        raise ValueError('Exploratory analysis requires TRAIN exclusively')
    result = descriptive_summary(frame)
    texts = frame['text']
    masks = {'url': texts.map(lambda t: bool(URL_PATTERN.search(t))),
             'mention': texts.map(lambda t: bool(MENTION_PATTERN.search(t))),
             'hashtag': texts.map(lambda t: bool(HASHTAG_PATTERN.search(t))),
             'emoji': texts.map(lambda t: bool(emoji.emoji_count(t)))}
    result['artifacts'] = {k: {'rows': int(v.sum()), 'percent': round(float(v.mean()*100), 3)} for k, v in masks.items()}
    result['top_tokens'] = frequent_terms(texts)
    result['top_bigrams'] = frequent_terms(texts, 2)
    result['tokens_by_sentiment'] = {label: frequent_terms(frame.loc[frame.sentiment.eq(label), 'text']) for label in LABELS}
    lengths = texts.str.len()
    result['short_examples'] = frame.loc[lengths.lt(3), ['text', 'sentiment']].head(5).to_dict('records')
    result['long_examples'] = frame.loc[lengths.gt(500), ['text', 'sentiment']].head(5).to_dict('records')
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(result['class_counts'].keys(), result['class_counts'].values(), color=['#b44c4c', '#708090', '#368f68'])
    ax.set(title='TweetEval TRAIN sentiment distribution', xlabel='Sentiment', ylabel='Posts')
    _save(fig, figures_dir / 'train_class_distribution.png')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, series, label in [(axes[0], lengths, 'Characters'), (axes[1], texts.str.split().str.len(), 'Words')]:
        ax.hist(series, bins=35, color='#3979a8')
        ax.set(xlabel=label, ylabel='Posts', title=f'TRAIN {label.lower()} per post')
    _save(fig, figures_dir / 'train_text_lengths.png')
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(masks.keys(), [v.mean()*100 for v in masks.values()], color='#3979a8')
    ax.set(title='Social-media artifacts in TRAIN', xlabel='Artifact', ylabel='Posts containing artifact (%)')
    _save(fig, figures_dir / 'train_artifacts.png')
    fig, ax = plt.subplots(figsize=(8, 5))
    terms, values = zip(*result['top_tokens'][:15])
    ax.barh(terms[::-1], values[::-1], color='#3979a8')
    ax.set(title='Most frequent TRAIN tokens (including stopwords)', xlabel='Occurrences', ylabel='Token')
    _save(fig, figures_dir / 'train_top_tokens.png')
    return result

def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_cleaning_lengths(frame: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> None:
    if not frame['split'].eq('train').all():
        raise ValueError('Length comparison requires TRAIN')
    fig, ax = plt.subplots(figsize=(8, 4))
    for column, label in [('text', 'Raw'), ('clean_text', 'Clean')]:
        ax.hist(frame[column].str.len(), bins=35, alpha=.5, label=label)
    ax.set(title='TRAIN text length before and after cleaning', xlabel='Characters', ylabel='Posts')
    ax.legend()
    _save(fig, Path(figures_dir) / 'train_cleaning_lengths.png')

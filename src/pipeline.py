"""Run the complete Phase 1 pipeline with reproducible reports."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import pandas as pd
from .config import PROJECT_ROOT, SPLITS
from .data_acquisition import ensure_dataset_available, sha256
from .data_loader import load_csv, save_csv
from .data_validation import validate_dataframe, require_valid, cross_split_duplicates
from .eda import analyze_train, descriptive_summary, plot_cleaning_lengths
from .preprocessing import DEFAULT_CONFIG, preprocess_dataframe

def run_phase1(root: Path = PROJECT_ROOT) -> dict:
    """Validate before publishing outputs; no row filtering or fitting is performed."""
    root = Path(root)
    reports = root / 'reports'
    figures = reports / 'figures'
    reports.mkdir(parents=True, exist_ok=True)
    paths = ensure_dataset_available(root / 'data' / 'raw')
    frames = {name: load_csv(path) for name, path in paths.items()}
    quality = {name: validate_dataframe(frame, name) for name, frame in frames.items()}
    for result in quality.values():
        require_valid(result)
    train_eda = analyze_train(frames['train'], figures)
    processed = {name: preprocess_dataframe(frame) for name, frame in frames.items()}
    clean_quality = {name: validate_dataframe(frame, name, text_column='clean_text') for name, frame in processed.items()}
    # Persist diagnostics even if cleaning exposes unusable rows. Never silently drop them.
    audit = {'raw': quality, 'processed': clean_quality,
             'cross_split_exact_text': cross_split_duplicates(frames),
             'cross_split_clean_text': cross_split_duplicates(processed, 'clean_text')}
    (reports / 'data_quality.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    for result in clean_quality.values():
        require_valid(result)
    files = {}
    for name, frame in processed.items():
        pd.testing.assert_frame_equal(frame[frames[name].columns], frames[name])
        path = root / 'data' / 'processed' / f'{name}_clean.csv'
        save_csv(frame, path)
        pd.testing.assert_frame_equal(load_csv(path), frame)
        files[name] = {'path': path.relative_to(root).as_posix(), 'rows': len(frame),
                       'sha256': sha256(path), 'removed_rows': 0,
                       'changed_rows': int(frame.text.ne(frame.clean_text).sum())}
    plot_cleaning_lengths(processed['train'], figures)
    manifest = json.loads((root / 'data' / 'raw' / 'manifest.json').read_text(encoding='utf-8'))
    result = {'dataset': manifest['dataset'], 'revision': manifest['revision'],
              'configuration': manifest['configuration'], 'preprocessing': asdict(DEFAULT_CONFIG),
              'train_eda': train_eda, 'validation_summary': descriptive_summary(frames['validation']),
              'quality': audit, 'outputs': files,
              'environment': {'python': platform.python_version(), 'packages': {
                  name: importlib.metadata.version(name) for name in ['pandas', 'matplotlib', 'datasets', 'emoji', 'pytest', 'nbformat', 'nbclient', 'ipykernel']}}}
    (reports / 'phase1_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (root / 'data' / 'processed' / 'manifest.json').write_text(json.dumps({k: result[k] for k in ['dataset', 'revision', 'preprocessing', 'outputs', 'environment']}, indent=2), encoding='utf-8')
    (reports / 'phase1_summary.md').write_text(render_summary(result), encoding='utf-8')
    logging.getLogger(__name__).info('Phase 1 passed: %s', {k: v['rows'] for k, v in files.items()})
    return result

def render_summary(result: dict) -> str:
    """Render only measured values; suppress test class distributions/examples."""
    eda, quality = result['train_eda'], result['quality']
    lines = ['# Phase 1 executed summary', '',
             f"Dataset: `{result['dataset']}`, configuration `sentiment`.",
             'Source: https://huggingface.co/datasets/cardiffnlp/tweet_eval',
             f"Immutable revision: `{result['revision']}`.", '',
             '## Split integrity and outputs', '',
             '| Split | Raw rows | Processed rows | Changed text | Removed |',
             '|---|---:|---:|---:|---:|']
    for name, output in result['outputs'].items():
        lines.append(f"| {name} | {quality['raw'][name]['counts']['rows']} | {output['rows']} | {output['changed_rows']} | {output['removed_rows']} |")
    lines += ['', 'Raw columns: text, sentiment, split. Processed columns add clean_text.',
              'All saved CSVs were reloaded and compared with in-memory data. Original text, order, labels and split membership are unchanged.',
              '', '## Quality findings', '',
              'Counts below are mechanical integrity checks, not exploratory test-label analysis. Duplicate counts exclude the first occurrence; overlap counts are distinct nonblank strings.', '']
    for stage in ['raw', 'processed']:
        for name, audit in quality[stage].items():
            lines.append(f"- {stage}/{name}: errors `{audit['critical_errors']}`; warnings `{audit['warnings']}`; missing cells `{sum(audit['counts']['missing_values'].values())}`; blank text `{audit['counts']['blank_text']}`.")
    lines += ['', f"Exact raw text overlap: `{quality['cross_split_exact_text']}`.",
              f"Cleaned text overlap: `{quality['cross_split_clean_text']}`.",
              'Official duplicates are preserved. No deduplication or rebalancing was performed.',
              '', '## TRAIN findings', '', '| Sentiment | Rows | Percent |', '|---|---:|---:|']
    for label, count in eda['class_counts'].items():
        lines.append(f"| {label} | {count} | {eda['class_percentages'][label]} |")
    lines += ['', f"Characters per training post: mean {eda['characters']['mean']:.2f}, median {eda['characters']['50%']:.0f}, range {eda['characters']['min']:.0f}–{eda['characters']['max']:.0f}.",
              f"Words per training post: mean {eda['words']['mean']:.2f}, median {eda['words']['50%']:.0f}.",
              f"Artifact prevalence: `{eda['artifacts']}`.",
              f"Top training tokens (stopwords included): `{eda['top_tokens'][:10]}`.",
              f"Top training bigrams: `{eda['top_bigrams'][:10]}`.",
              f"Suspiciously short TRAIN examples (<3 characters): `{eda['short_examples']}`.",
              f"Suspiciously long TRAIN examples (>500 characters): `{eda['long_examples']}`.",
              'Class imbalance means Phase 2 should report macro-F1 and per-class precision/recall alongside accuracy. Frequent tokens are descriptive counts, not fitted modeling features.',
              'See phase1_metrics.json for validation summary, full training statistics and tokens by sentiment; figures/ contains five training charts.',
              '', '## Fixed preprocessing decisions', '',
              'NFC Unicode and curly-apostrophe normalization; HTML tags removed and entities decoded; lowercase; URLs and mentions removed; hashtag words retained. Unicode emojis, contractions, negation, punctuation and repeated characters retained. Whitespace collapsed. No stopword removal, stemming or lemmatization. Optional mention tokens, demojizing and repeat reduction are available but unused in primary outputs.',
              'No rows removed. Missing or cleaning-empty text is reported as a critical error and blocks publication instead of silently filtering benchmark rows.',
              '', '## Risks and Phase 2 handoff', '',
              'Historical English tweets are not representative of all customers or contemporary brand discourse. Sarcasm, context, annotation ambiguity and platform/demographic bias remain. Exact duplicate auditing does not detect paraphrases. Lowercasing loses capitalization intensity, but raw text is retained.',
              'Default word-based TF-IDF tokenization may discard emoji and punctuation and fragment contractions: Phase 2 must explicitly design its tokenizer using TRAIN only and validate choices on VALIDATION. No tokenizer or feature extractor was fit here.',
              'Consume data/processed/{train,validation,test}_clean.csv with text, clean_text, sentiment, split. Fit features and models only on TRAIN; tune/select on VALIDATION; use TEST only for final evaluation. Test class distributions/examples are intentionally absent.',
              '', '## Reproducibility', '',
              'Run `python -m src.pipeline`. Raw and processed manifests record SHA-256 checksums, source revision, configuration and runtime versions. Run `python -m pytest -q` and `python scripts/execute_notebooks.py` for verification.', '']
    if result['dataset'] != 'cardiffnlp/tweet_eval':
        lines.insert(1, '\nWARNING: Results below are pipeline-validation results using synthetic/sample data and MUST NOT be interpreted as real-world findings. NOT FOR MODEL EVALUATION.\n')
    return '\n'.join(lines)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    run_phase1(args.root)

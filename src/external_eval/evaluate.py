"""One-way frozen-model evaluation on completed human external labels."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from typing import Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support,
)

from src.config import LABELS, PROJECT_ROOT
from src.inference import SentimentPredictor

from .labeling import read_labeling_csv, validate_labeled_file

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'reports' / 'external_eval'
OFFICIAL_METRICS_PATH = PROJECT_ROOT / 'reports' / 'final_test_metrics.json'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _metric_payload(actual: list[str], predicted: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=list(LABELS), average=None, zero_division=0)
    macro = precision_recall_fscore_support(
        actual, predicted, labels=list(LABELS), average='macro', zero_division=0)
    weighted = precision_recall_fscore_support(
        actual, predicted, labels=list(LABELS), average='weighted', zero_division=0)
    matrix = confusion_matrix(actual, predicted, labels=list(LABELS))
    return {
        'accuracy': float(accuracy_score(actual, predicted)),
        'macro_precision': float(macro[0]),
        'macro_recall': float(macro[1]),
        'macro_f1': float(macro[2]),
        'weighted_f1': float(weighted[2]),
        'per_class': {
            label: {
                'precision': float(precision[index]),
                'recall': float(recall[index]),
                'f1': float(f1[index]),
                'support': int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        'confusion_matrix': matrix.astype(int).tolist(),
        'confusion_matrix_order': list(LABELS),
        'confusion_matrix_axes': {'rows': 'actual', 'columns': 'predicted'},
    }


def _dataset_summary(rows: list[dict[str, str]]) -> dict:
    lengths = [len(row['text']) for row in rows]
    dates = sorted(row['created_at'].strip() for row in rows
                   if row.get('created_at', '').strip())
    normalized_texts = [' '.join(row['text'].casefold().split()) for row in rows]
    return {
        'total_rows': len(rows),
        'rows_per_source': dict(sorted(Counter(
            row['source'] for row in rows).items())),
        'rows_per_query': dict(sorted(Counter(
            row['query'] for row in rows).items())),
        'label_distribution': dict(sorted(Counter(
            row['label'] for row in rows).items())),
        'date_range': {
            'minimum': dates[0] if dates else None,
            'maximum': dates[-1] if dates else None,
        },
        'duplicate_sample_id_count': len(rows) - len({
            row['sample_id'] for row in rows}),
        'exact_text_duplicate_count': len(rows) - len(set(normalized_texts)),
        'average_text_length': mean(lengths) if lengths else 0.0,
        'median_text_length': median(lengths) if lengths else 0.0,
    }


def _benchmark_comparison(external: dict) -> dict:
    official = json.loads(OFFICIAL_METRICS_PATH.read_text(encoding='utf-8'))[
        'test_metrics']
    fields = {
        'accuracy': (official['accuracy'], external['accuracy']),
        'macro_f1': (official['macro_f1'], external['macro_f1']),
        **{
            f'{label}_f1': (
                official['per_class'][label]['f1'],
                external['per_class'][label]['f1'])
            for label in LABELS
        },
    }
    return {
        name: {
            'tweeteval_test': float(values[0]),
            'external_brand_domain': float(values[1]),
            'difference': float(values[1] - values[0]),
        }
        for name, values in fields.items()
    }


def _save_confusion_matrix(matrix: list[list[int]], path: Path) -> None:
    values = np.asarray(matrix, dtype=int)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(values, cmap='Blues')
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(LABELS)), yticks=range(len(LABELS)),
        xticklabels=LABELS, yticklabels=LABELS,
        xlabel='Predicted label', ylabel='Actual label',
        title='External brand-domain confusion matrix',
    )
    threshold = values.max() / 2 if values.size else 0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, str(values[row, column]),
                      ha='center', va='center',
                      color='white' if values[row, column] > threshold else 'black')
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(figure)


def _summary_markdown(metrics: dict, comparison: dict, row_count: int) -> str:
    lines = [
        '# External brand-domain evaluation', '',
        f'Rows evaluated: {row_count}', '',
        'The frozen TweetEval-trained Logistic Regression model was evaluated '
        'without retraining, threshold changes, recalibration, or model selection.', '',
        '## External metrics', '',
        '| Metric | Value |', '|---|---:|',
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| Macro precision | {metrics['macro_precision']:.4f} |",
        f"| Macro recall | {metrics['macro_recall']:.4f} |",
        f"| Macro-F1 | {metrics['macro_f1']:.4f} |",
        f"| Weighted F1 | {metrics['weighted_f1']:.4f} |", '',
        '## Per-class metrics', '',
        '| Label | Precision | Recall | F1 | Support |',
        '|---|---:|---:|---:|---:|',
    ]
    for label in LABELS:
        values = metrics['per_class'][label]
        lines.append(
            f"| {label} | {values['precision']:.4f} | {values['recall']:.4f} | "
            f"{values['f1']:.4f} | {values['support']} |")
    lines.extend([
        '', '## Domain-gap comparison', '',
        '| Metric | TweetEval TEST | External domain | Difference |',
        '|---|---:|---:|---:|',
    ])
    for name, values in comparison.items():
        lines.append(
            f"| {name} | {values['tweeteval_test']:.4f} | "
            f"{values['external_brand_domain']:.4f} | "
            f"{values['difference']:+.4f} |")
    lines.extend([
        '', 'TweetEval TEST and the external brand/product sample come from '
        'different distributions and are not directly interchangeable. '
        'These results describe external-domain transfer only.', '',
        'The gap can reflect a platform shift from historical tweets to YouTube '
        'comments, newer language and slang, brand/product-specific vocabulary, '
        'sarcasm, missing video or conversation context, and the external '
        'sample\'s different class and query composition. In particular, a '
        'large class-specific change should be interpreted as evidence about '
        'this sample rather than a reason to alter the frozen model.', '',
    ])
    return '\n'.join(lines)


def evaluate_external_dataset(
    labels_path: Path | str,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    predictor: SentimentPredictor | None = None,
) -> dict:
    """Validate human labels, predict once, and save aggregate-only reports."""

    labels_file = Path(labels_path)
    validation = validate_labeled_file(labels_file)
    rows = read_labeling_csv(labels_file)
    predictor = predictor or SentimentPredictor()
    predictions = predictor.predict_batch([row['text'] for row in rows])
    actual = [row['label'] for row in rows]
    predicted = [prediction.sentiment for prediction in predictions]
    metrics = _metric_payload(actual, predicted)
    comparison = _benchmark_comparison(metrics)
    generated_at = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    payload = {
        'status': 'external domain evaluation complete',
        'generated_at': generated_at,
        'dataset_rows': validation['row_count'],
        'labels_file_sha256': _sha256(labels_file),
        'model_artifact_sha256': predictor.artifact_sha256,
        'model_version': predictor.model_version,
        'training_or_tuning_performed': False,
        'metrics': metrics,
        'domain_gap_comparison': comparison,
        'distribution_caveat': (
            'TweetEval TEST and external brand-domain results are different '
            'distributions and are not directly interchangeable.'),
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / 'external_metrics.json'
    summary_path = destination / 'external_summary.md'
    matrix_path = destination / 'external_confusion_matrix.png'
    dataset_path = destination / 'dataset_summary.json'
    metrics_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    dataset_path.write_text(
        json.dumps(_dataset_summary(rows), indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    summary_path.write_text(
        _summary_markdown(metrics, comparison, len(rows)), encoding='utf-8')
    _save_confusion_matrix(metrics['confusion_matrix'], matrix_path)
    return {
        'metrics': payload,
        'paths': {
            'metrics': metrics_path,
            'summary': summary_path,
            'confusion_matrix': matrix_path,
            'dataset_summary': dataset_path,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Evaluate completed human labels with the frozen model.')
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    try:
        result = evaluate_external_dataset(
            args.labels, output_dir=args.output_dir)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Rows evaluated: {result['metrics']['dataset_rows']}")
    print(f"Macro-F1: {result['metrics']['metrics']['macro_f1']:.4f}")
    print(f"Reports: {args.output_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

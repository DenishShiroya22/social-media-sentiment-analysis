"""VALIDATION-only metrics with explicit class order."""
from sklearn.metrics import (accuracy_score, classification_report,
    confusion_matrix, precision_recall_fscore_support)
from .config import LABELS

def evaluate_validation(y_true, y_pred):
    """The caller passes VALIDATION labels and predictions only."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average='macro', zero_division=0)
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average='weighted', zero_division=0)
    return {'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_precision': float(macro[0]), 'macro_recall': float(macro[1]),
        'macro_f1': float(macro[2]), 'weighted_f1': float(weighted[2]),
        'per_class': {label: {'precision': float(precision[i]),
            'recall': float(recall[i]), 'f1': float(f1[i])}
            for i, label in enumerate(LABELS)},
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        'classification_report': classification_report(y_true, y_pred,
            labels=LABELS, output_dict=True, zero_division=0)}

def select_winner(records):
    """Rank by VALIDATION macro-F1, accuracy, then stable experiment ID."""
    if not records:
        raise ValueError('No completed experiments')
    return sorted(records, key=lambda r: (-r['validation_macro_f1'],
        -r['validation_accuracy'], r['experiment_id']))[0]

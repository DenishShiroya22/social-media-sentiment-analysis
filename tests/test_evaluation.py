import pytest
from src.evaluation import evaluate_validation, select_winner
from src.config import LABELS

def test_metrics_class_order_and_zero_division():
    result = evaluate_validation(
        ['negative', 'neutral', 'positive'],
        ['neutral', 'neutral', 'positive'])
    assert tuple(result['per_class']) == LABELS
    assert result['confusion_matrix'] == [[0,1,0],[0,1,0],[0,0,1]]
    assert result['per_class']['negative']['precision'] == 0
    assert result['per_class']['negative']['recall'] == 0
    assert result['per_class']['neutral']['f1'] == pytest.approx(2/3)
    assert result['macro_f1'] == pytest.approx((0+2/3+1)/3)
    assert result['accuracy'] == pytest.approx(2/3)

def test_winner_uses_macro_f1():
    records = [
        {'experiment_id':'a','validation_macro_f1':.6,'validation_accuracy':.9},
        {'experiment_id':'b','validation_macro_f1':.7,'validation_accuracy':.7}]
    assert select_winner(records)['experiment_id'] == 'b'
    with pytest.raises(ValueError):
        select_winner([])

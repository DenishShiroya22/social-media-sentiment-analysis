import pandas as pd
import pytest
from src.data_validation import normalize_labels, validate_dataframe, cross_split_duplicates

def example():
    return pd.DataFrame({'text': ['great', 'great', 'no thanks'], 'sentiment': ['positive', 'positive', 'negative'], 'split': ['train']*3})

@pytest.mark.parametrize('column', ['text', 'sentiment', 'split'])
def test_missing_column(column):
    assert column in validate_dataframe(example().drop(columns=column))['critical_errors']['missing_columns']

@pytest.mark.parametrize('column,value', [('sentiment', 'bad'), ('sentiment', None), ('split', 'dev'), ('split', ''), ('text', None), ('text', ''), ('text', '   '), ('text', 42)])
def test_invalid_values(column, value):
    frame = example().astype(object)
    frame.loc[0, column] = value
    assert validate_dataframe(frame)['critical_errors']

def test_duplicates():
    result = validate_dataframe(example())
    assert result['warnings']['duplicate_rows'] == 1
    assert result['warnings']['duplicate_text'] == 1

def test_overlap():
    a = example()
    b = a.iloc[:1].assign(split='test', sentiment='negative')
    assert cross_split_duplicates({'train': a, 'test': b}) == {'train__test': 1}

def test_empty_and_dates_lengths():
    assert validate_dataframe(example().iloc[:0])['critical_errors']['empty_dataframe']
    frame = example().assign(text=['a', 'x'*501, 'normal'], created_at=['bad', '2020-01-01', None])
    warnings = validate_dataframe(frame)['warnings']
    assert warnings == {'short_text': 1, 'long_text': 1, 'invalid_dates': 1}

@pytest.mark.parametrize('values,mapping,expected', [([0, 1, 2], None, ['negative', 'neutral', 'positive']), ([-1, 0, 1], {-1:'negative', 0:'neutral', 1:'positive'}, ['negative', 'neutral', 'positive']), (['POSITIVE', 'Neutral', ' negative '], None, ['positive', 'neutral', 'negative'])])
def test_label_conventions(values, mapping, expected):
    assert normalize_labels(pd.Series(values), mapping).tolist() == expected

@pytest.mark.parametrize('value', [None, 'unknown', 5])
def test_reject_bad_label(value):
    with pytest.raises(ValueError):
        normalize_labels(pd.Series([value]))

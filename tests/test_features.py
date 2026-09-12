import numpy as np
import pytest
from scipy import sparse
from src.features import make_features, fit_train_transform_validation, WORD_TOKEN_PATTERN
import re

@pytest.mark.parametrize('name', ['word_unigram', 'word_bigram', 'character', 'combined'])
def test_sparse_train_fit_validation_transform(name):
    train = ["i don't like this", "i don't like that", "great phone", "great battery"]
    validation = ["exclusivevalidationword great"]
    extractor = make_features(name)
    x_train, x_validation = fit_train_transform_validation(extractor, train, validation)
    assert sparse.issparse(x_train) and sparse.issparse(x_validation)
    assert x_train.shape[0] == 4 and x_validation.shape[0] == 1
    assert x_train.shape[1] == x_validation.shape[1]
    assert not any('exclusivevalidationword' in term for term in extractor.get_feature_names_out())

def test_contraction_pattern():
    assert "don't" in re.findall(WORD_TOKEN_PATTERN, "don't can't isn't won't not good")
    assert "not" in re.findall(WORD_TOKEN_PATTERN, "not good")

def test_invalid_feature():
    with pytest.raises(ValueError, match='Unknown feature'):
        make_features('invalid')

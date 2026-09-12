import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from src.models import make_model

@pytest.mark.parametrize('name,kind', [
    ('logistic_regression', LogisticRegression),
    ('linear_svm', LinearSVC),
    ('random_forest', RandomForestClassifier),
    ('naive_bayes', MultinomialNB),
])
def test_model_factory(name, kind):
    assert isinstance(make_model(name), kind)

def test_invalid_model():
    with pytest.raises(ValueError, match='Unknown model'):
        make_model('invalid')

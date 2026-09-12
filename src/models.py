"""Small, reproducible classical-classifier registry."""
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from .config import RANDOM_STATE

@dataclass(frozen=True)
class Experiment:
    id: str
    feature: str
    model: str

EXPERIMENTS = (
    Experiment('exp01', 'word_unigram', 'logistic_regression'),
    Experiment('exp02', 'word_bigram', 'logistic_regression'),
    Experiment('exp03', 'character', 'logistic_regression'),
    Experiment('exp04', 'combined', 'logistic_regression'),
    Experiment('exp05', 'word_bigram', 'linear_svm'),
    Experiment('exp06', 'character', 'linear_svm'),
    Experiment('exp07', 'combined', 'linear_svm'),
    Experiment('exp08', 'word_bigram', 'naive_bayes'),
    Experiment('exp09', 'word_bigram', 'random_forest'),
)

def make_model(name):
    """Return an unfitted model; preserve official class proportions."""
    if name == 'logistic_regression':
        return LogisticRegression(C=1.0, max_iter=350, solver='lbfgs', random_state=RANDOM_STATE)
    if name == 'linear_svm':
        return LinearSVC(C=1.0, max_iter=3000, random_state=RANDOM_STATE)
    if name == 'naive_bayes':
        return MultinomialNB(alpha=1.0)
    if name == 'random_forest':
        return RandomForestClassifier(n_estimators=80, max_depth=24,
            min_samples_leaf=2, max_features='sqrt', n_jobs=2,
            random_state=RANDOM_STATE)
    raise ValueError(f'Unknown model: {name}')

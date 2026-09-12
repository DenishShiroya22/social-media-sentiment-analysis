"""Controlled TF-IDF representations fitted exclusively on TRAIN."""
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

WORD_TOKEN_PATTERN = r"(?u)\b\w+(?:['’]\w+)*\b"
FEATURE_NAMES = ('word_unigram', 'word_bigram', 'character', 'combined')

def _word(ngram_range):
    return TfidfVectorizer(analyzer='word', ngram_range=ngram_range,
        token_pattern=WORD_TOKEN_PATTERN, min_df=2, max_df=.95,
        max_features=100_000, sublinear_tf=True, norm='l2',
        lowercase=False, dtype=np.float32)

def _character():
    return TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
        min_df=2, max_df=.95, max_features=120_000, sublinear_tf=True,
        norm='l2', lowercase=False, dtype=np.float32)

def make_features(name):
    """Build an unfitted extractor; clean_text is already lowercased."""
    if name == 'word_unigram':
        return _word((1, 1))
    if name == 'word_bigram':
        return _word((1, 2))
    if name == 'character':
        return _character()
    if name == 'combined':
        return FeatureUnion([('word', _word((1, 2))), ('character', _character())])
    raise ValueError(f'Unknown feature representation: {name}')

def fit_train_transform_validation(extractor, train_text, validation_text):
    """Fit vocabulary and IDF on TRAIN only; transform VALIDATION."""
    x_train = extractor.fit_transform(train_text)
    x_validation = extractor.transform(validation_text)
    if not sparse.issparse(x_train) or not sparse.issparse(x_validation):
        raise TypeError('TF-IDF must produce sparse matrices')
    return x_train, x_validation

def feature_parameters(extractor):
    def params(vectorizer):
        p = vectorizer.get_params(deep=False)
        return {key: p[key] for key in ('analyzer', 'ngram_range', 'min_df',
            'max_df', 'max_features', 'sublinear_tf', 'norm', 'lowercase',
            'token_pattern')}
    if isinstance(extractor, FeatureUnion):
        return {name: params(vectorizer) for name, vectorizer in extractor.transformer_list}
    return params(extractor)

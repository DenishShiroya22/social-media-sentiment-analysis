"""Deterministic, stateless cleaning that retains sentiment cues."""
from dataclasses import dataclass
import html
import re
import unicodedata
import emoji
import pandas as pd

URL_PATTERN = re.compile(r'\b(?:https?://|www\.)[^\s<>]+', re.IGNORECASE)
MENTION_PATTERN = re.compile(r'(?<![\w@])@[\w]+')
HASHTAG_PATTERN = re.compile(r'#(\w+)')

@dataclass(frozen=True)
class PreprocessingConfig:
    lowercase: bool = True
    strip_urls: bool = True
    mention_strategy: str = 'remove'
    preserve_hashtag_words: bool = True
    emoji_strategy: str = 'preserve'
    reduce_repeats: bool = False

DEFAULT_CONFIG = PreprocessingConfig()

ESCAPED_PUNCTUATION = {
    r'\u002c': ',', r'\u0027': "'",
    r'\u2018': "'", r'\u2019': "'",
    r'\u201c': '"', r'\u201d': '"',
}
ESCAPED_PUNCTUATION_PATTERN = re.compile(
    r'\\u(?:002c|0027|2018|2019|201c|201d)', re.IGNORECASE
)

def normalize_unicode(text: str) -> str:
    """Decode only known escaped punctuation, then normalize actual Unicode."""
    text = ESCAPED_PUNCTUATION_PATTERN.sub(
        lambda match: ESCAPED_PUNCTUATION[match.group().lower()], text)
    return (unicodedata.normalize('NFC', text)
            .replace('\u2019', "'").replace('\u2018', "'")
            .replace('\u201c', '"').replace('\u201d', '"'))

def handle_html(text: str) -> str:
    # Strip actual tags before unescaping so encoded literal comparisons survive.
    return html.unescape(re.sub(r'</?[A-Za-z][^>]*>', ' ', text))

def remove_urls(text: str) -> str:
    return URL_PATTERN.sub(lambda m: ' ' + (re.search(r'[.!?,;:]+$', m[0])[0] if re.search(r'[.!?,;:]+$', m[0]) else ''), text)

def handle_mentions(text: str, strategy: str = 'remove') -> str:
    if strategy not in ('remove', 'token', 'preserve'):
        raise ValueError('mention_strategy must be remove, token or preserve')
    return text if strategy == 'preserve' else MENTION_PATTERN.sub(' USER ' if strategy == 'token' else ' ', text)

def handle_hashtags(text: str, preserve_words: bool = True) -> str:
    return HASHTAG_PATTERN.sub(r'\1' if preserve_words else ' ', text)

def normalize_whitespace(text: str) -> str:
    return ' '.join(text.split())

def normalize_repeated_characters(text: str) -> str:
    return re.sub(r'([A-Za-z])\1{2,}', r'\1\1', text)

def handle_emojis(text: str, strategy: str = 'preserve') -> str:
    if strategy == 'preserve':
        return text
    if strategy == 'demojize':
        return emoji.demojize(text, delimiters=(' ', ' '))
    raise ValueError('emoji_strategy must be preserve or demojize')

def clean_text(text: object, config: PreprocessingConfig = DEFAULT_CONFIG) -> str:
    """Missing values become empty strings for explicit downstream validation."""
    if text is None or (not isinstance(text, str) and pd.isna(text)):
        return ''
    if not isinstance(text, str):
        raise TypeError('Text must be a string or a missing scalar')
    text = normalize_unicode(handle_html(text))
    if config.strip_urls:
        text = remove_urls(text)
    text = handle_mentions(text, config.mention_strategy)
    text = handle_hashtags(text, config.preserve_hashtag_words)
    text = handle_emojis(text, config.emoji_strategy)
    if config.reduce_repeats:
        text = normalize_repeated_characters(text)
    if config.lowercase:
        text = text.lower()
    return normalize_whitespace(text)

def preprocess_dataframe(frame: pd.DataFrame, config: PreprocessingConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Keep original text, metadata, labels, order and row count unchanged."""
    if 'text' not in frame:
        raise ValueError('Required column text is missing')
    result = frame.copy()
    result['clean_text'] = result['text'].map(lambda value: clean_text(value, config))
    return result

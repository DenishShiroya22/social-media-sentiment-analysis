import pandas as pd
import pytest
from src.preprocessing import clean_text, PreprocessingConfig

@pytest.mark.parametrize('raw,expected', [('Visit https://example.com NOW', 'visit now'), ('   hello \t world\n ', 'hello world'), ('Thanks @SamsungIndia', 'thanks'), ('Great phone 😍 #AmazingCamera', 'great phone 😍 amazingcamera'), ("I DON'T like this!!!", "i don't like this!!!"), ('no never nor not', 'no never nor not'), (None, ''), (float('nan'), ''), (pd.NA, ''), ('', ''), ('Soooo GOOD!!!', 'soooo good!!!'), ('<b>Good</b> &amp; nice', 'good & nice'), ('I can’t', "i can't"), ('cafe\u0301', 'café')])
def test_cleaning(raw, expected):
    assert clean_text(raw) == expected

def test_options():
    config = PreprocessingConfig(lowercase=False, strip_urls=False, mention_strategy='token', emoji_strategy='demojize', reduce_repeats=True)
    result = clean_text('@person Soooo 😍 https://example.com', config)
    assert 'USER' in result and 'Soo' in result and 'smiling_face_with_heart' in result and 'https://' in result

def test_preserve_options():
    assert clean_text('@brand #Good', PreprocessingConfig(mention_strategy='preserve', preserve_hashtag_words=False)) == '@brand'

def test_email_and_url_punctuation():
    assert clean_text('me@example.com https://example.com!!!') == 'me@example.com !!!'

def test_bad_type():
    with pytest.raises(TypeError):
        clean_text(5)

@pytest.mark.parametrize('config', [PreprocessingConfig(emoji_strategy='delete'), PreprocessingConfig(mention_strategy='bad')])
def test_bad_configuration(config):
    with pytest.raises(ValueError):
        clean_text('hello', config)

@pytest.mark.parametrize('raw,expected', [
    (r"I don\u2019t like this", "i don't like this"),
    (r"She said \u201cgood\u201d", 'she said "good"'),
    (r"I can\u2018t", "i can't"),
    (r"I don\u0027t", "i don't"),
    (r"red\u002c blue", "red, blue"),
    (r"I CAN\u2019T", "i can't"),
    ("I can’t 😡 café", "i can't 😡 café"),
    ("I don't like this", "i don't like this"),
    (r"C:\users\docs", r"c:\users\docs"),
    (r"keep \u1234 untouched", r"keep \u1234 untouched"),
])
def test_targeted_unicode_escape_normalization(raw, expected):
    assert clean_text(raw) == expected

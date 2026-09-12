import pandas as pd
import pytest
from src.data_loader import load_csv, save_csv

def test_csv_roundtrip(tmp_path):
    frame = pd.DataFrame({'text': ['NA', 'null', 'quoted, "text"\nnext 😍'], 'sentiment': ['neutral']*3})
    path = tmp_path / 'nested' / 'data.csv'
    save_csv(frame, path)
    pd.testing.assert_frame_equal(load_csv(path), frame)

def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match='Dataset CSV not found'):
        load_csv(tmp_path / 'absent.csv')

def test_invalid_encoding(tmp_path):
    path = tmp_path / 'bad.csv'
    path.write_bytes(b'text\n\xff')
    with pytest.raises(ValueError, match='Cannot read CSV'):
        load_csv(path)

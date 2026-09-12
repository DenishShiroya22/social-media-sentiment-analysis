"""CSV I/O that preserves literal text such as 'NA' and 'null'."""
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

def load_csv(path: Path | str, encoding: str = 'utf-8-sig') -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Dataset CSV not found: {path}')
    try:
        frame = pd.read_csv(path, encoding=encoding, keep_default_na=False)
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f'Cannot read CSV {path}: {exc}') from exc
    logger.info('Loaded %s: %d rows, %d columns: %s', path, *frame.shape, list(frame.columns))
    return frame

def save_csv(frame: pd.DataFrame, path: Path | str) -> None:
    """Write atomically without an index; preserve optional metadata columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.csv.tmp')
    frame.to_csv(temporary, index=False, encoding='utf-8', lineterminator='\n')
    temporary.replace(path)

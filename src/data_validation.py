"""Structured validation; benchmark rows are reported, never silently removed."""
from itertools import combinations
import pandas as pd
from .config import LABEL_MAP, LABELS, SPLITS

def normalize_labels(values: pd.Series, mapping: dict | None = None) -> pd.Series:
    """Normalize strings case-insensitively; numeric conventions require a mapping."""
    lookup = {str(k).strip().lower(): v for k, v in (LABEL_MAP if mapping is None else mapping).items()}
    lookup.update({label: label for label in LABELS})
    def convert(value):
        if pd.isna(value):
            raise ValueError('Null sentiment label')
        key = str(int(value)) if isinstance(value, float) and value.is_integer() else str(value).strip().lower()
        if key not in lookup or lookup[key] not in LABELS:
            raise ValueError(f'Unsupported sentiment label: {value!r}')
        return lookup[key]
    return values.map(convert)

def validate_dataframe(frame: pd.DataFrame, expected_split: str | None = None,
                       text_column: str = 'text', short_threshold: int = 3,
                       long_threshold: int = 500) -> dict:
    """Return critical errors, warnings and counts without exposing examples/labels."""
    errors, warnings, counts = {}, {}, {}
    missing = sorted({'text', 'sentiment', 'split', text_column} - set(frame.columns))
    if missing:
        errors['missing_columns'] = missing
    if frame.empty:
        errors['empty_dataframe'] = True
    counts['rows'] = len(frame)
    counts['missing_values'] = {k: int(v) for k, v in frame.isna().sum().items()}
    counts['duplicate_rows'] = int(frame.duplicated().sum())
    if frame.empty:
        return {'critical_errors': errors, 'warnings': warnings, 'counts': counts}
    if counts['duplicate_rows']:
        warnings['duplicate_rows'] = counts['duplicate_rows']
    if text_column in frame:
        text = frame[text_column]
        bad_type = ~text.map(lambda x: isinstance(x, str)) & text.notna()
        empty = text.map(lambda x: isinstance(x, str) and not x.strip())
        for key, mask in [('null_text', text.isna()), ('non_string_text', bad_type), ('blank_text', empty)]:
            counts[key] = int(mask.sum())
            if counts[key]:
                errors[key] = counts[key]
        valid = text[text.map(lambda x: isinstance(x, str) and bool(x.strip()))]
        counts['duplicate_text'] = int(valid.duplicated().sum())
        counts['short_text'] = int((valid.str.len() < short_threshold).sum())
        counts['long_text'] = int((valid.str.len() > long_threshold).sum())
        warnings.update({k: counts[k] for k in ('duplicate_text', 'short_text', 'long_text') if counts[k]})
    for column, allowed in [('sentiment', LABELS), ('split', SPLITS)]:
        if column in frame:
            invalid = int((~frame[column].isin(allowed)).sum())
            if invalid:
                errors[f'invalid_{column}'] = invalid
    if expected_split is not None and 'split' in frame and not frame['split'].eq(expected_split).all():
        errors['split_mismatch'] = expected_split
    if 'created_at' in frame:
        dates = frame['created_at']
        invalid = int((dates.notna() & dates.ne('') & pd.to_datetime(dates, errors='coerce', utc=True, format='mixed').isna()).sum())
        if invalid:
            warnings['invalid_dates'] = invalid
    return {'critical_errors': errors, 'warnings': warnings, 'counts': counts}

def require_valid(result: dict) -> None:
    if result['critical_errors']:
        raise ValueError(f"Dataset contract failed: {result['critical_errors']}")

def cross_split_duplicates(frames: dict[str, pd.DataFrame], column: str = 'text') -> dict:
    """Label-blind exact overlap counts; no frames/labels are concatenated."""
    sets = {name: set(df[column].dropna().loc[lambda s: s.map(lambda x: isinstance(x, str) and bool(x.strip()))]) for name, df in frames.items()}
    return {f'{a}__{b}': len(sets[a] & sets[b]) for a, b in combinations(sets, 2)}

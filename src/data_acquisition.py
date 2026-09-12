"""Acquire official TweetEval splits and verify reuse with checksums."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import pandas as pd
from .config import DATASET_ID, DATASET_CONFIG, DATASET_REVISION, SPLITS, LABELS, EXPECTED_ROWS, RAW_DATA_DIR
from .data_loader import load_csv, save_csv
from .data_validation import normalize_labels, validate_dataframe, require_valid

logger = logging.getLogger(__name__)

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def normalize_tweet_eval_split(split, name: str) -> pd.DataFrame:
    if name not in SPLITS:
        raise ValueError(f'Unexpected split {name}')
    if not {'text', 'label'} <= set(split.column_names):
        raise ValueError('TweetEval requires text and label fields')
    if list(getattr(split.features['label'], 'names', [])) != list(LABELS):
        raise ValueError('TweetEval label semantics do not match negative/neutral/positive')
    frame = split.to_pandas().rename(columns={'label': 'sentiment'})
    frame['sentiment'] = normalize_labels(frame['sentiment'])
    frame['split'] = name
    require_valid(validate_dataframe(frame, name))
    return frame

def ensure_dataset_available(raw_dir: Path = RAW_DATA_DIR, force: bool = False,
                             revision: str | None = None) -> dict[str, Path]:
    """Reuse verified local files; otherwise fetch one immutable upstream revision."""
    raw_dir = Path(raw_dir)
    paths = {name: raw_dir / f'tweet_eval_{name}.csv' for name in SPLITS}
    manifest_path = raw_dir / 'manifest.json'
    if not force and manifest_path.is_file() and all(p.is_file() for p in paths.values()):
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['dataset'] != DATASET_ID or manifest['configuration'] != DATASET_CONFIG:
            raise ValueError('Local provenance mismatch; use --force to reacquire')
        expected_revision = revision or DATASET_REVISION
        if expected_revision != manifest['revision']:
            raise ValueError('Configured/requested revision differs from local cache; use --force')
        for name, path in paths.items():
            if sha256(path) != manifest['files'][name]['sha256']:
                raise ValueError(f'Integrity failure: {path}; use --force to reacquire')
            frame = load_csv(path)
            require_valid(validate_dataframe(frame, name))
            if len(frame) != manifest['files'][name]['rows'] or len(frame) != EXPECTED_ROWS[name]:
                raise ValueError(f'Row-count mismatch for {name}')
        logger.info('Reusing verified official raw splits')
        return paths
    raw_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('HF_HOME', str(raw_dir.parent / '.hf_cache'))
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
        resolved = HfApi().dataset_info(DATASET_ID, revision=revision or DATASET_REVISION).sha
        dataset = load_dataset(DATASET_ID, DATASET_CONFIG, revision=resolved,
                               cache_dir=str(raw_dir.parent / '.hf_cache' / 'datasets'))
    except Exception as exc:
        raise RuntimeError('TweetEval acquisition failed. Check network access and run '
                           'python -m src.data_acquisition again. No substitute dataset was used.') from exc
    if set(dataset) != set(SPLITS):
        raise ValueError(f'Unexpected official splits: {list(dataset)}')
    frames = {name: normalize_tweet_eval_split(dataset[name], name) for name in SPLITS}
    manifest = {'dataset': DATASET_ID, 'configuration': DATASET_CONFIG, 'revision': resolved,
                'label_mapping': {str(i): label for i, label in enumerate(LABELS)}, 'files': {}}
    for name, frame in frames.items():
        if len(frame) != EXPECTED_ROWS[name]:
            raise ValueError(f'Unexpected benchmark size for {name}: {len(frame)}')
        save_csv(frame, paths[name])
        manifest['files'][name] = {'rows': len(frame), 'sha256': sha256(paths[name]),
                                  'fingerprint': dataset[name]._fingerprint}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--revision')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    ensure_dataset_available(force=args.force, revision=args.revision)

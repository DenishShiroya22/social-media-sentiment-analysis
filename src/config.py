"""Central paths and benchmark contracts."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
SAMPLE_DATA_DIR = DATA_DIR / 'sample'
REPORTS_DIR = PROJECT_ROOT / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'
DATASET_ID = 'cardiffnlp/tweet_eval'
DATASET_CONFIG = 'sentiment'
DATASET_REVISION = 'b3a375baf0f409c77e6bc7aa35102b7b3534f8be'
SPLITS = ('train', 'validation', 'test')
LABEL_MAP = {0: 'negative', 1: 'neutral', 2: 'positive'}
LABELS = tuple(LABEL_MAP.values())
EXPECTED_ROWS = {'train': 45615, 'validation': 2000, 'test': 12284}

"""Reusable inference over the immutable final sentiment pipeline."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from .config import LABELS, PROJECT_ROOT
from .preprocessing import DEFAULT_CONFIG, clean_text

DEFAULT_MODEL_PATH = PROJECT_ROOT / 'models' / 'final' / 'final_model.joblib'
DEFAULT_METADATA_PATH = PROJECT_ROOT / 'models' / 'final' / 'final_model_metadata.json'
MAX_TEXT_CHARACTERS = 100_000


@dataclass(frozen=True)
class SentimentPrediction:
    """Serializable prediction for one raw application text."""

    text: str
    clean_text: str
    sentiment: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str
    model_artifact_sha256: str
    inference_timestamp: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class SentimentPredictor:
    """Load the frozen model once and predict validated raw text in batches."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        metadata_path: Path | str = DEFAULT_METADATA_PATH,
        *,
        verify_checksum: bool = True,
        max_text_characters: int = MAX_TEXT_CHARACTERS,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.verify_checksum = verify_checksum
        if max_text_characters < 1:
            raise ValueError('max_text_characters must be positive')
        self.max_text_characters = max_text_characters
        self._model: Pipeline | None = None
        self._metadata: dict | None = None
        self._artifact_sha256: str | None = None

    @property
    def artifact_sha256(self) -> str:
        self._ensure_loaded()
        return str(self._artifact_sha256)

    @property
    def model_version(self) -> str:
        self._ensure_loaded()
        return f"final-lr-{self._metadata['pretest_commit_sha'][:12]}"

    def _ensure_loaded(self) -> Pipeline:
        if self._model is not None:
            return self._model
        if not self.model_path.is_file():
            raise FileNotFoundError(f'Final model artifact not found: {self.model_path}')
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f'Final model metadata not found: {self.metadata_path}')
        metadata = json.loads(self.metadata_path.read_text(encoding='utf-8'))
        expected = metadata.get('artifact_sha256')
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError('Final model metadata has no valid artifact SHA-256')
        actual = sha256_file(self.model_path)
        if self.verify_checksum and actual != expected:
            raise ValueError('Final model artifact SHA-256 does not match metadata')
        if metadata.get('frozen_configuration', {}).get('preprocessing') != asdict(DEFAULT_CONFIG):
            raise ValueError('Application preprocessing differs from final model metadata')
        model = joblib.load(self.model_path)
        if not isinstance(model, Pipeline):
            raise TypeError('Final model must be an sklearn Pipeline')
        if list(model.named_steps) != ['features', 'classifier']:
            raise ValueError('Final model pipeline contract changed')
        if not hasattr(model, 'predict_proba'):
            raise TypeError('Final classifier does not provide probabilities')
        if tuple(model.classes_) != LABELS:
            raise ValueError('Final model class order changed')
        self._metadata = metadata
        self._artifact_sha256 = actual
        self._model = model
        return model

    def _validate_and_clean(self, texts: Sequence[str]) -> tuple[list[str], list[str]]:
        raw, cleaned = [], []
        for index, value in enumerate(texts):
            if value is None:
                raise TypeError(f'texts[{index}] must be a string, received None')
            if not isinstance(value, str):
                raise TypeError(f'texts[{index}] must be a string')
            if not value.strip():
                raise ValueError(f'texts[{index}] must not be empty or whitespace-only')
            if len(value) > self.max_text_characters:
                raise ValueError(
                    f'texts[{index}] exceeds {self.max_text_characters} characters')
            normalized = clean_text(value, DEFAULT_CONFIG)
            if not normalized:
                raise ValueError(
                    f'texts[{index}] is empty after deterministic preprocessing')
            raw.append(value)
            cleaned.append(normalized)
        return raw, cleaned

    def predict_batch(self, texts: Sequence[str]) -> list[SentimentPrediction]:
        """Predict a sequence with one batched predict and predict_proba call."""

        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError('predict_batch expects a sequence of strings')
        if len(texts) == 0:
            return []
        raw, cleaned = self._validate_and_clean(texts)
        model = self._ensure_loaded()
        labels = np.asarray(model.predict(cleaned))
        probabilities = np.asarray(model.predict_proba(cleaned), dtype=float)
        if labels.shape != (len(raw),) or probabilities.shape != (len(raw), len(LABELS)):
            raise RuntimeError('Final model returned an unexpected batch shape')
        timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        model_version = self.model_version
        artifact_sha256 = self.artifact_sha256
        results = []
        for index, label in enumerate(labels):
            distribution = {
                class_name: float(probabilities[index, class_index])
                for class_index, class_name in enumerate(LABELS)
            }
            if str(label) not in LABELS:
                raise RuntimeError(f'Final model returned unknown label: {label}')
            results.append(SentimentPrediction(
                text=raw[index],
                clean_text=cleaned[index],
                sentiment=str(label),
                confidence=max(distribution.values()),
                probabilities=distribution,
                model_version=model_version,
                model_artifact_sha256=artifact_sha256,
                inference_timestamp=timestamp,
            ))
        return results

    def predict_one(self, text: str) -> SentimentPrediction:
        """Predict one raw text using the same batch implementation."""

        return self.predict_batch([text])[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Predict sentiment with the frozen final model.')
    parser.add_argument('--text', required=True, help='Raw text to analyze.')
    args = parser.parse_args(argv)
    result = SentimentPredictor().predict_one(args.text)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

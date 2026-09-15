"""Local JSONL storage with exclusive run-specific writes."""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
from uuid import uuid4

from src.config import PROJECT_ROOT

RUN_ID_PATTERN = re.compile(r'^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}$')


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def create_run_id(now: datetime | None = None) -> str:
    moment = (now or utc_now()).astimezone(timezone.utc)
    return f"{moment.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"


def safe_segment(value: str, *, maximum: int = 48) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError('filename segment must be a non-empty string')
    segment = re.sub(r'[^a-z0-9]+', '-', value.lower()).strip('-')[:maximum]
    if not segment:
        raise ValueError('filename segment contains no safe characters')
    return segment


def _record_dict(record: Any) -> dict[str, Any]:
    if hasattr(record, 'to_dict'):
        value = record.to_dict()
    elif is_dataclass(record):
        value = asdict(record)
    elif isinstance(record, Mapping):
        value = dict(record)
    else:
        raise TypeError('records must be mappings or serializable dataclasses')
    if not isinstance(value, dict):
        raise TypeError('record serialization must return a dictionary')
    return value


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    source: str
    query: str
    retrieved_count: int
    analyzed_count: int
    created_at: str
    model_sha256: str
    model_version: str
    connector_version: str
    collected_file: str
    output_file: str

    def __post_init__(self) -> None:
        if not RUN_ID_PATTERN.fullmatch(self.run_id):
            raise ValueError('invalid run_id')
        if self.retrieved_count < 0 or self.analyzed_count < 0:
            raise ValueError('manifest counts cannot be negative')
        if self.analyzed_count > self.retrieved_count:
            raise ValueError('analyzed_count cannot exceed retrieved_count')
        if len(self.model_sha256) != 64:
            raise ValueError('model_sha256 must be a full SHA-256')


class JsonlRepository:
    """Write collected, analyzed, and manifest files below a local data root."""

    def __init__(self, root: Path | str = PROJECT_ROOT/'data') -> None:
        self.root = Path(root)

    def _record_path(self, category: str, source: str,
                     query: str, run_id: str) -> Path:
        if category not in {'collected', 'analyzed'}:
            raise ValueError('category must be collected or analyzed')
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError('invalid run_id')
        return (self.root/category/
                f'{safe_segment(source)}_{safe_segment(query)}_{run_id}.jsonl')

    def write_records(
        self,
        category: str,
        source: str,
        query: str,
        run_id: str,
        records: Iterable[Any],
    ) -> Path:
        """Write a UTF-8 JSONL file exclusively; an existing run is never replaced."""

        path = self._record_path(category, source, query, run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8', newline='\n') as handle:
            for record in records:
                handle.write(json.dumps(
                    _record_dict(record), ensure_ascii=False,
                    sort_keys=True, separators=(',', ':')) + '\n')
        return path

    @staticmethod
    def read_records(path: Path | str) -> list[dict[str, Any]]:
        values = []
        with Path(path).open('r', encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f'JSONL line {line_number} is not an object')
                values.append(value)
        return values

    def write_manifest(self, manifest: RunManifest) -> Path:
        path = self.root/'manifests'/f'{manifest.run_id}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(json.dumps(
                asdict(manifest), ensure_ascii=False,
                indent=2, sort_keys=True) + '\n')
        return path

    def relative_path(self, path: Path) -> str:
        return path.resolve().relative_to(self.root.resolve().parent).as_posix()

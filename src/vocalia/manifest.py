"""JSONL manifest helpers.

Every stage that produces data writes a manifest describing what it produced and
what it came from. Manifests are tracked in git while the media is not, so the
provenance of the corpus survives even though the audio is never redistributed.

JSONL rather than JSON: stages append incrementally and may be interrupted, and a
half-written array is unrecoverable while a half-written line file loses only the
last record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read all rows. Missing file yields an empty list, so callers need no guard."""
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
    return rows


def iter_manifest(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream rows, for manifests too large to hold in memory."""
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_row(path: str | Path, row: dict[str, Any]) -> None:
    """Append one record. ensure_ascii=False keeps Devanagari readable in diffs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Rewrite the whole manifest. Used when a row is replaced under --force."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def upsert_row(path: str | Path, row: dict[str, Any], key: str) -> None:
    """Insert or replace the row whose `key` matches, preserving file order."""
    rows = read_manifest(path)
    for i, existing in enumerate(rows):
        if existing.get(key) == row.get(key):
            rows[i] = row
            write_manifest(path, rows)
            return
    append_row(path, row)


def has_id(path: str | Path, key: str, value: str) -> bool:
    """True when a row with this key already exists — the idempotency check."""
    return any(row.get(key) == value for row in iter_manifest(path))

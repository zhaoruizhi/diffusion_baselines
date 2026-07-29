"""Dependency-free helpers for reproducible JSON assets."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def atomic_json_write(path: Path, value: object) -> None:
    """Atomically write JSON to *path*, creating its parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary_file:
        temporary_file.write(payload)
        temporary_file.flush()
        os.fsync(temporary_file.fileno())
        temporary_path = Path(temporary_file.name)
    os.replace(temporary_path, path)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

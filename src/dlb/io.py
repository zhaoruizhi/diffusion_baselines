"""Dependency-free helpers for reproducible JSON assets."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Iterable, Iterator, Mapping, Sequence

from dlb.schema import ConditionalSampleRecord, SampleRecord


class SampleValidationError(ValueError):
    """A JSONL sample artifact violates the public sample contract."""


class SampleCountError(SampleValidationError):
    """A JSONL sample artifact has a different number of records than requested."""


def expected_conditional_schedule(
    prompt_count: int = 1024,
    diversity_prompt_count: int = 256,
    completions: int = 5,
) -> list[tuple[int, int]]:
    """Return the canonical quality-then-diversity conditional sample schedule."""

    return [(prompt, 0) for prompt in range(prompt_count)] + [
        (prompt, completion)
        for completion in range(1, completions)
        for prompt in range(diversity_prompt_count)
    ]


def _ensure_safe_directory(path: Path) -> None:
    """Create *path* while refusing symlinked or non-directory ancestors."""

    missing: list[Path] = []
    current = path
    while True:
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            missing.append(current)
            parent = current.parent
            if parent == current:
                raise SampleValidationError(f"directory does not have an existing root: {path}")
            current = parent
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise SampleValidationError(f"directory is a symlink: {current}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise SampleValidationError(f"directory is not a directory: {current}")
        break
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            metadata = directory.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise SampleValidationError(f"directory is unsafe: {directory}")


def _require_regular_or_missing(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode):
        raise SampleValidationError(f"path is a symlink: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise SampleValidationError(f"path is not a regular file: {path}")


def ensure_safe_directory(path: Path) -> None:
    """Public directory guard for artifacts that need descriptor-backed writes."""

    _ensure_safe_directory(path)


def open_safe_output(path: Path):
    """Open a regular output file without following a leaf symlink."""

    _ensure_safe_directory(path.parent)
    _require_regular_or_missing(path)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    return os.fdopen(descriptor, "wb")


def remove_safe_file(path: Path) -> None:
    """Remove one regular artifact and durably record the directory update."""

    _ensure_safe_directory(path.parent)
    _require_regular_or_missing(path)
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(path.parent)


def _fsync_directory(directory: Path) -> None:
    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_temporary(path: Path) -> tuple[Path, int]:
    for _ in range(100):
        temporary = path.parent / f".{path.name}.{secrets.token_hex(16)}.partial"
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except FileExistsError:
            continue
        else:
            return temporary, descriptor
    raise RuntimeError(f"could not allocate a unique temporary file for {path}")


def _atomic_bytes_write(path: Path, payload: bytes) -> None:
    _ensure_safe_directory(path.parent)
    _require_regular_or_missing(path)
    temporary, descriptor = _open_temporary(path)
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        _require_regular_or_missing(path)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_json_write(path: Path, value: object) -> None:
    """Atomically write JSON to *path*, creating its parent directories."""

    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    _atomic_bytes_write(path, payload.encode("utf-8"))


def write_compact_jsonl_atomic(path: Path, records: Iterable[Mapping[str, object]]) -> int:
    """Atomically publish compact, deterministic UTF-8 JSONL records.

    Callers retain ownership of record-specific validation.  This helper owns the
    descriptor-safe temporary-file, fsync, replacement, and directory-durability
    protocol shared by JSONL artifact types.
    """

    _ensure_safe_directory(path.parent)
    _require_regular_or_missing(path)
    temporary, descriptor = _open_temporary(path)
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as temporary_file:
            for count, record in enumerate(records, start=1):
                temporary_file.write(
                    json.dumps(
                        dict(record),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        _require_regular_or_missing(path)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        return count
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _json_object_with_unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SampleValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _sample_from_value(value: object, index: int) -> SampleRecord:
    try:
        return SampleRecord.model_validate(value)
    except Exception as error:
        raise SampleValidationError(f"record {index}: {error}") from error


def _conditional_sample_from_value(value: object, index: int) -> ConditionalSampleRecord:
    try:
        return ConditionalSampleRecord.model_validate(value)
    except Exception as error:
        raise SampleValidationError(f"record {index}: {error}") from error


def _conditional_contract(
    *,
    expected: int | None,
    schedule: Sequence[tuple[int, int]] | None,
    sequence_length: int,
    vocab_size: int,
) -> list[tuple[int, int]]:
    if expected is not None and (type(expected) is not int or expected < 0):
        raise ValueError("expected sample count must be a non-negative integer or None")
    if type(sequence_length) is not int or sequence_length not in {128, 1024}:
        raise ValueError("conditional sequence_length must be 128 for LM1B or 1024 for OWT")
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("conditional vocab_size must be a positive integer")
    schedule_records = list(expected_conditional_schedule() if schedule is None else schedule)
    if expected is not None and len(schedule_records) != expected:
        raise ValueError(
            f"conditional schedule has {len(schedule_records)} entries, expected {expected}"
        )
    if not schedule_records:
        raise ValueError("conditional schedule must contain at least one completion-0 prompt")
    for index, entry in enumerate(schedule_records):
        if (
            not isinstance(entry, tuple)
            or len(entry) != 2
            or type(entry[0]) is not int
            or type(entry[1]) is not int
        ):
            raise ValueError(f"conditional schedule entry {index} must be an (int, int) tuple")
    quality_prompt_count = 0
    while (
        quality_prompt_count < len(schedule_records)
        and schedule_records[quality_prompt_count][1] == 0
    ):
        if schedule_records[quality_prompt_count] != (quality_prompt_count, 0):
            raise ValueError("conditional schedule quality prompts must be contiguous and unique")
        quality_prompt_count += 1
    if quality_prompt_count == 0 or quality_prompt_count > 1024:
        raise ValueError("conditional schedule prompt count must be between 1 and 1024")
    diversity_records = schedule_records[quality_prompt_count:]
    if not diversity_records:
        return schedule_records
    diversity_prompt_count = 0
    while (
        diversity_prompt_count < len(diversity_records)
        and diversity_records[diversity_prompt_count] == (diversity_prompt_count, 1)
    ):
        diversity_prompt_count += 1
    if diversity_prompt_count == 0 or diversity_prompt_count > quality_prompt_count:
        raise ValueError("conditional schedule diversity prompts must be a quality-prompt prefix")
    if len(diversity_records) % diversity_prompt_count != 0:
        raise ValueError("conditional schedule has an incomplete diversity completion group")
    completions = len(diversity_records) // diversity_prompt_count + 1
    if completions > 5:
        raise ValueError("conditional schedule completions must be between 1 and 5")
    if schedule_records != expected_conditional_schedule(
        quality_prompt_count, diversity_prompt_count, completions
    ):
        raise ValueError("conditional schedule is noncanonical")
    return schedule_records


def _validate_conditional_record(
    record: ConditionalSampleRecord,
    index: int,
    *,
    schedule: Sequence[tuple[int, int]],
    sequence_length: int,
    vocab_size: int,
) -> None:
    if record.sample_id != index:
        raise SampleValidationError(
            f"record {index}: expected sample_id {index}, found {record.sample_id}"
        )
    if index >= len(schedule):
        raise SampleValidationError(f"record {index}: no expected prompt/completion schedule entry")
    expected_prompt_completion = schedule[index]
    observed_prompt_completion = (record.prompt_id, record.completion_id)
    if observed_prompt_completion != expected_prompt_completion:
        raise SampleValidationError(
            f"record {index}: expected prompt/completion {expected_prompt_completion}, "
            f"found {observed_prompt_completion}"
        )
    if len(record.full_token_ids) != sequence_length:
        raise SampleValidationError(
            f"record {index}: full_token_ids length {len(record.full_token_ids)} does not match "
            f"sequence_length {sequence_length}"
        )
    for field in (
        "prefix_token_ids",
        "continuation_token_ids",
        "reference_token_ids",
        "full_token_ids",
    ):
        if any(token >= vocab_size for token in getattr(record, field)):
            raise SampleValidationError(
                f"record {index}: {field} contains a token outside vocabulary [0, {vocab_size - 1}]"
            )


def read_conditional_samples(path: Path) -> Iterator[ConditionalSampleRecord]:
    """Stream strict conditional sample records from a JSONL artifact."""

    _ensure_safe_directory(path.parent)
    _require_regular_or_missing(path)
    if not path.exists():
        raise SampleValidationError(f"conditional sample file is missing: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as sample_file:
            for index, raw_line in enumerate(sample_file):
                line_number = index + 1
                if not raw_line.strip():
                    raise SampleValidationError(f"record {index} (line {line_number}) is blank")
                try:
                    value = json.loads(raw_line, object_pairs_hook=_json_object_with_unique_keys)
                except (json.JSONDecodeError, UnicodeDecodeError, SampleValidationError) as error:
                    raise SampleValidationError(
                        f"record {index} (line {line_number}) has malformed JSON: {error}"
                    ) from error
                yield _conditional_sample_from_value(value, index)
    except UnicodeDecodeError as error:
        raise SampleValidationError(f"conditional sample file is not UTF-8: {path}") from error


def validate_conditional_samples(
    path: Path,
    *,
    expected: int | None = 2048,
    schedule: Sequence[tuple[int, int]] | None = None,
    sequence_length: int,
    vocab_size: int,
) -> int:
    """Stream and validate a fixed-prefix conditional sample JSONL artifact."""

    schedule_records = _conditional_contract(
        expected=expected,
        schedule=schedule,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
    )
    count = 0
    for index, record in enumerate(read_conditional_samples(path)):
        _validate_conditional_record(
            record,
            index,
            schedule=schedule_records,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
        )
        count = index + 1
    if count != len(schedule_records):
        raise SampleCountError(f"expected {len(schedule_records)} records, found {count}")
    return count


def write_conditional_samples_atomic(
    path: Path,
    records: Iterable[ConditionalSampleRecord | Mapping[str, object]],
    *,
    expected: int | None = 2048,
    schedule: Sequence[tuple[int, int]] | None = None,
    sequence_length: int,
    vocab_size: int,
) -> int:
    """Validate and atomically publish a fixed-prefix conditional JSONL artifact."""

    schedule_records = _conditional_contract(
        expected=expected,
        schedule=schedule,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
    )
    serialized_records: list[dict[str, object]] = []
    for index, value in enumerate(records):
        record = (
            value
            if isinstance(value, ConditionalSampleRecord)
            else _conditional_sample_from_value(value, index)
        )
        _validate_conditional_record(
            record,
            index,
            schedule=schedule_records,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
        )
        serialized_records.append(record.model_dump(mode="json"))
    count = len(serialized_records)
    if count != len(schedule_records):
        raise SampleCountError(f"expected {len(schedule_records)} records, found {count}")
    return write_compact_jsonl_atomic(path, serialized_records)


def validate_samples(path: Path, expected: int | None = 1024) -> int:
    """Stream and validate a sample JSONL artifact, returning its record count."""

    if expected is not None and expected < 0:
        raise ValueError("expected sample count must be non-negative")
    _ensure_safe_directory(path.parent)
    _require_regular_or_missing(path)
    if not path.exists():
        raise SampleValidationError(f"sample file is missing: {path}")

    count = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as sample_file:
            for line_number, raw_line in enumerate(sample_file, start=1):
                if not raw_line.strip():
                    raise SampleValidationError(f"record {count} (line {line_number}) is blank")
                try:
                    value = json.loads(raw_line, object_pairs_hook=_json_object_with_unique_keys)
                except (json.JSONDecodeError, UnicodeDecodeError, SampleValidationError) as error:
                    raise SampleValidationError(
                        f"record {count} (line {line_number}) has malformed JSON: {error}"
                    ) from error
                record = _sample_from_value(value, count)
                if record.sample_id != count:
                    raise SampleValidationError(
                        f"record {count} (line {line_number}): expected sample_id {count}, "
                        f"found {record.sample_id}"
                    )
                count += 1
    except UnicodeDecodeError as error:
        raise SampleValidationError(f"sample file is not UTF-8: {path}") from error
    if expected is not None and count != expected:
        raise SampleCountError(f"expected {expected} records, found {count}")
    return count


def write_samples_atomic(
    path: Path,
    records: Iterable[SampleRecord | Mapping[str, object]],
    *,
    expected: int | None = None,
) -> int:
    """Validate and atomically publish deterministic UTF-8 sample JSONL."""

    if expected is not None and expected < 0:
        raise ValueError("expected sample count must be non-negative")
    _ensure_safe_directory(path.parent)
    _require_regular_or_missing(path)
    temporary, descriptor = _open_temporary(path)
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as temporary_file:
            for index, value in enumerate(records):
                count = index + 1
                record = value if isinstance(value, SampleRecord) else _sample_from_value(value, index)
                if record.sample_id != index:
                    raise SampleValidationError(
                        f"record {index}: expected sample_id {index}, found {record.sample_id}"
                    )
                temporary_file.write(
                    json.dumps(
                        record.model_dump(mode="json"),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        validate_samples(temporary, expected=count if expected is None else expected)
        _require_regular_or_missing(path)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        return count
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

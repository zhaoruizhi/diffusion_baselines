"""Deterministic data splitting, packing, manifests, and verification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import random
from typing import Iterable, Iterator, Mapping, Sequence

from dlb.io import sha256_file


OWT_REQUIRED_FREE_BYTES = 55 * 1024**3


def check_owt_disk_space(free_bytes: int, *, allow_low_disk: bool) -> bool:
    """Enforce the OWT preflight and return whether an override was used."""

    below_threshold = free_bytes < OWT_REQUIRED_FREE_BYTES
    if below_threshold and not allow_low_disk:
        raise RuntimeError(
            f"OpenWebText requires at least 55 GiB free; found {free_bytes} bytes. "
            "Pass --allow-low-disk to explicitly bypass."
        )
    return below_threshold


@dataclass(frozen=True)
class SplitExpressions:
    """Hugging Face split expressions used to construct train and validation."""

    train: str
    validation: str


def build_owt_split(total_documents: int) -> SplitExpressions:
    """Return the fixed original-order OpenWebText split."""

    if total_documents <= 100_000:
        raise ValueError("OpenWebText must contain more than 100,000 documents")
    return SplitExpressions(
        train="train[:-100000]",
        validation="train[-100000:]",
    )


def pack_tokens(
    documents: Iterable[Sequence[int]],
    *,
    length: int,
    bos_id: int,
    eos_id: int,
) -> Iterator[list[int]]:
    """Pack a document stream into full blocks with BOS/EOS boundaries.

    An EOS is inserted after every document. The resulting continuous stream is
    divided into ``length - 2`` token chunks, then every full chunk is wrapped in
    BOS/EOS. The final incomplete chunk is intentionally dropped.
    """

    if length < 3:
        raise ValueError("packed sequence length must be at least 3")
    content_length = length - 2
    pending: list[int] = []
    for document in documents:
        pending.extend(int(token_id) for token_id in document)
        pending.append(eos_id)
        while len(pending) >= content_length:
            content = pending[:content_length]
            del pending[:content_length]
            yield [bos_id, *content, eos_id]


def preprocess_split(
    source: object,
    *,
    tokenizer: object,
    length: int,
    bos_id: int,
    eos_id: int,
    cache_dir: Path,
    batch_documents: int = 1_000,
    detokenizer: object | None = None,
) -> object:
    """Tokenize and continuously pack a complete local source split."""

    if batch_documents < 1:
        raise ValueError("batch_documents must be positive")
    from datasets import Dataset, Features, Sequence, Value

    def generate_rows() -> Iterator[dict[str, list[int]]]:
        def document_tokens() -> Iterator[Sequence[int]]:
            for start in range(0, len(source), batch_documents):  # type: ignore[arg-type]
                texts = source[start : start + batch_documents]["text"]  # type: ignore[index]
                if detokenizer is not None:
                    texts = [detokenizer(text) for text in texts]  # type: ignore[operator]
                encoded = tokenizer(  # type: ignore[operator]
                    texts,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )["input_ids"]
                yield from encoded

        for input_ids in pack_tokens(
            document_tokens(), length=length, bos_id=bos_id, eos_id=eos_id
        ):
            yield {"input_ids": input_ids}

    cache_dir.mkdir(parents=True, exist_ok=True)
    features = Features({"input_ids": Sequence(Value("uint16"), length=length)})
    return Dataset.from_generator(
        generate_rows,
        features=features,
        cache_dir=str(cache_dir),
    )


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def inventory_files(output_dir: Path) -> list[dict[str, object]]:
    """Return deterministic size and SHA-256 records for every regular file."""

    return [
        {
            "path": path.relative_to(output_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(candidate for candidate in output_dir.rglob("*") if candidate.is_file())
    ]


def build_data_manifest(
    *,
    dataset: str,
    dataset_id: str,
    source_revision: str,
    tokenizer_id: str,
    tokenizer_revision: str,
    sequence_length: int,
    split_expression: Mapping[str, str],
    document_counts: Mapping[str, int],
    packed_sequence_counts: Mapping[str, int],
    vocab_size: int,
    min_token_id: int,
    max_token_id: int,
    output_dir: Path,
    root: Path,
) -> dict[str, object]:
    """Build the reproducibility and integrity manifest for processed data."""

    return {
        "schema_version": 1,
        "dataset": dataset,
        "dataset_id": dataset_id,
        "source_revision": source_revision,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "sequence_length": sequence_length,
        "split_expression": dict(split_expression),
        "document_counts": dict(document_counts),
        "packed_sequence_counts": dict(packed_sequence_counts),
        "vocabulary": {
            "size": vocab_size,
            "min_token_id": min_token_id,
            "max_token_id": max_token_id,
        },
        "processed_path": _relative_to_root(output_dir, root),
        "files": inventory_files(output_dir),
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "verified": False,
    }


def _python_sequence_stats(split: object, expected_length: int) -> tuple[int, int, int]:
    minimum: int | None = None
    maximum: int | None = None
    count = len(split)  # type: ignore[arg-type]
    for index in range(count):
        token_ids = split[index]["input_ids"]  # type: ignore[index]
        if len(token_ids) != expected_length:
            raise ValueError(
                f"sequence {index} has length {len(token_ids)}, expected {expected_length}"
            )
        row_minimum = min(token_ids)
        row_maximum = max(token_ids)
        minimum = row_minimum if minimum is None else min(minimum, row_minimum)
        maximum = row_maximum if maximum is None else max(maximum, row_maximum)
    if minimum is None or maximum is None:
        raise ValueError("processed split is empty")
    return count, minimum, maximum


def _sequence_stats(split: object, expected_length: int) -> tuple[int, int, int]:
    """Use Arrow kernels for full scans, with a tiny test-double fallback."""

    try:
        import pyarrow.compute as pc

        column = split.data.column("input_ids")  # type: ignore[attr-defined]
        lengths = pc.list_value_length(column)
        if pc.any(pc.not_equal(lengths, expected_length)).as_py():
            raise ValueError(f"at least one sequence does not have length {expected_length}")
        bounds = pc.min_max(pc.list_flatten(column)).as_py()
        if bounds["min"] is None or bounds["max"] is None:
            raise ValueError("processed split is empty")
        return len(split), int(bounds["min"]), int(bounds["max"])  # type: ignore[arg-type]
    except (AttributeError, ImportError):
        return _python_sequence_stats(split, expected_length)


def verify_processed_dataset(
    *,
    dataset: Mapping[str, object],
    manifest: Mapping[str, object],
    output_dir: Path,
    tokenizer: object,
    decode_samples_per_split: int = 3,
) -> dict[str, object]:
    """Fully verify row counts, lengths, token bounds, files, and decoding."""

    expected_splits = {"train", "validation"}
    if set(dataset) != expected_splits:
        raise ValueError(f"processed splits must be {sorted(expected_splits)}")

    expected_length = int(manifest["sequence_length"])
    expected_counts = manifest["packed_sequence_counts"]
    vocabulary = manifest["vocabulary"]
    vocab_size = int(vocabulary["size"])
    observed_minimum: int | None = None
    observed_maximum: int | None = None
    checked_sequences = 0
    decoded_samples = 0
    decoded_indices: dict[str, list[int]] = {}

    for split_name in sorted(expected_splits):
        split = dataset[split_name]
        count, minimum, maximum = _sequence_stats(split, expected_length)
        expected_count = int(expected_counts[split_name])
        if count != expected_count:
            raise ValueError(
                f"{split_name} has {count} sequences, manifest says {expected_count}"
            )
        if minimum < 0 or maximum >= vocab_size:
            raise ValueError(
                f"{split_name} token IDs [{minimum}, {maximum}] outside vocabulary "
                f"[0, {vocab_size - 1}]"
            )
        observed_minimum = minimum if observed_minimum is None else min(observed_minimum, minimum)
        observed_maximum = maximum if observed_maximum is None else max(observed_maximum, maximum)
        checked_sequences += count

        sample_count = min(decode_samples_per_split, count)
        indices = sorted(random.Random(f"dlb-data-{split_name}").sample(range(count), sample_count))
        decoded_indices[split_name] = indices
        for index in indices:
            decoded = tokenizer.decode(
                split[index]["input_ids"],  # type: ignore[index]
                skip_special_tokens=False,
            )
            if not isinstance(decoded, str) or not decoded:
                raise ValueError(f"{split_name} sequence {index} did not decode to text")
            decoded_samples += 1

    if observed_minimum != int(vocabulary["min_token_id"]):
        raise ValueError("observed minimum token ID does not match manifest")
    if observed_maximum != int(vocabulary["max_token_id"]):
        raise ValueError("observed maximum token ID does not match manifest")

    manifest_files = list(manifest["files"])
    current_files = inventory_files(output_dir)
    if current_files != manifest_files:
        raise ValueError("processed file inventory or SHA-256 digest does not match manifest")

    result = dict(manifest)
    result["verified"] = True
    result["verification"] = {
        "verified_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "checked_sequences": checked_sequences,
        "decoded_samples": decoded_samples,
        "decoded_indices": decoded_indices,
        "observed_min_token_id": observed_minimum,
        "observed_max_token_id": observed_maximum,
        "files_checked": len(current_files),
    }
    return result

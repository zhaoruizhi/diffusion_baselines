"""Deterministic data splitting, packing, manifests, and verification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
from typing import Iterable, Iterator, Mapping, Sequence

from dlb.io import atomic_json_write, sha256_file


OWT_REQUIRED_FREE_BYTES = 55 * 1024**3
PACKING_ALGORITHM = "document_eos_continuous_bos_eos"
PACKING_VERSION = 1
VOCAB_SIZES = {"lm1b": 30_522, "owt": 50_257}


def disk_preflight_evidence(free_bytes: int, *, allow_low_disk: bool) -> dict[str, object]:
    """Enforce OWT free space and return exact auditable decision evidence."""

    below_threshold = free_bytes < OWT_REQUIRED_FREE_BYTES
    if below_threshold and not allow_low_disk:
        raise RuntimeError(
            f"OpenWebText requires at least 55 GiB free; found {free_bytes} bytes. "
            "Pass --allow-low-disk to explicitly bypass."
        )
    return {
        "observed_free_bytes": free_bytes,
        "required_free_bytes": OWT_REQUIRED_FREE_BYTES,
        "below_threshold": below_threshold,
        "override_requested": allow_low_disk,
        "override_used": below_threshold and allow_low_disk,
    }


def processing_contract_sha256(contract: Mapping[str, object]) -> str:
    """Hash a processing contract using deterministic canonical JSON."""

    payload = json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_processing_contract(
    configuration: Mapping[str, object], dataset_name: str
) -> dict[str, object]:
    """Derive current processing expectations independently from any data manifest."""

    datasets = configuration["datasets"]
    models = configuration["models"]
    specification = datasets[dataset_name]
    if dataset_name == "lm1b":
        archive = configuration["lm1b_archive"]
        document_counts = {
            "train": int(archive["train_documents"]),
            "validation": int(archive["test_documents"]),
        }
    elif dataset_name == "owt":
        total = int(specification["documents"])
        document_counts = {"train": total - 100_000, "validation": 100_000}
    else:
        raise ValueError(f"unknown dataset {dataset_name!r}")
    tokenizer_id = str(specification["tokenizer"])
    return {
        "schema_version": 1,
        "dataset": dataset_name,
        "dataset_id": str(specification["repo_id"]),
        "source_revision": str(specification["revision"]),
        "source_materialization_revision": specification.get("parquet_revision"),
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": str(models[tokenizer_id]),
        "sequence_length": int(specification["sequence_length"]),
        "split_expression": dict(specification["splits"]),
        "expected_document_counts": document_counts,
        "vocab_size": VOCAB_SIZES[dataset_name],
        "packing": {"algorithm": PACKING_ALGORITHM, "version": PACKING_VERSION},
    }


def _expect_equal(field: str, observed: object, expected: object) -> None:
    if observed != expected:
        raise ValueError(f"{field} mismatch: observed {observed!r}, expected {expected!r}")


def _validate_snapshot_revision(field: str, value: object, revision: object) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field} missing")
    parts = Path(value).parts
    try:
        snapshot_revision = parts[parts.index("snapshots") + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"{field} is not a Hugging Face snapshot path") from error
    _expect_equal(field, snapshot_revision, revision)


def validate_download_manifest(
    configuration: Mapping[str, object],
    downloads: Mapping[str, object],
    dataset_name: str,
) -> Mapping[str, object]:
    """Validate downloaded dataset and tokenizer provenance against pinned config."""

    _expect_equal("schema_version", downloads.get("schema_version"), 1)
    contract = build_processing_contract(configuration, dataset_name)
    try:
        record = downloads["datasets"][dataset_name]
    except (KeyError, TypeError) as error:
        raise ValueError(f"missing download record for {dataset_name}") from error
    expected_raw_splits = (
        {
            "train": contract["expected_document_counts"]["train"],
            "test": contract["expected_document_counts"]["validation"],
        }
        if dataset_name == "lm1b"
        else {"train": int(configuration["datasets"]["owt"]["documents"])}
    )
    _expect_equal("repo_id", record.get("repo_id"), contract["dataset_id"])
    _expect_equal(
        "source_revision", record.get("source_revision"), contract["source_revision"]
    )
    _expect_equal(
        "materialization_revision",
        record.get("materialization_revision"),
        contract["source_materialization_revision"],
    )
    _expect_equal("split_rows", record.get("split_rows"), expected_raw_splits)
    expected_cache_splits = set(expected_raw_splits)
    cache_files = record.get("cache_files")
    if not isinstance(cache_files, Mapping) or set(cache_files) != expected_cache_splits:
        raise ValueError("cache_files split mismatch")
    if any(not isinstance(paths, list) or not paths for paths in cache_files.values()):
        raise ValueError("cache_files must contain at least one file for every split")
    if dataset_name == "lm1b":
        _validate_snapshot_revision(
            "dataset_snapshot",
            record.get("dataset_snapshot"),
            contract["source_materialization_revision"],
        )
        _validate_snapshot_revision(
            "source_metadata_snapshot",
            record.get("source_metadata_snapshot"),
            contract["source_revision"],
        )

    tokenizer_id = contract["tokenizer_id"]
    try:
        tokenizer_record = downloads["models"][tokenizer_id]
    except (KeyError, TypeError) as error:
        raise ValueError(f"missing tokenizer download record for {tokenizer_id}") from error
    _expect_equal("tokenizer repo_id", tokenizer_record.get("repo_id"), tokenizer_id)
    _expect_equal(
        "tokenizer revision",
        tokenizer_record.get("revision"),
        contract["tokenizer_revision"],
    )
    _validate_snapshot_revision(
        "tokenizer snapshot_path",
        tokenizer_record.get("snapshot_path"),
        contract["tokenizer_revision"],
    )
    return record


def validate_manifest_contract(
    manifest: Mapping[str, object], expected_contract: Mapping[str, object]
) -> None:
    """Reject any manifest not created under the complete current contract."""

    expected_fingerprint = processing_contract_sha256(expected_contract)
    if manifest.get("processing_contract") != dict(expected_contract):
        raise ValueError("processing contract fields do not match current configuration")
    if manifest.get("processing_contract_sha256") != expected_fingerprint:
        raise ValueError("processing contract fingerprint does not match current configuration")
    top_level_fields = {
        "dataset": "dataset",
        "dataset_id": "dataset_id",
        "source_revision": "source_revision",
        "source_materialization_revision": "source_materialization_revision",
        "tokenizer_id": "tokenizer_id",
        "tokenizer_revision": "tokenizer_revision",
        "sequence_length": "sequence_length",
        "split_expression": "split_expression",
        "document_counts": "expected_document_counts",
    }
    for manifest_field, contract_field in top_level_fields.items():
        _expect_equal(
            f"manifest {manifest_field}",
            manifest.get(manifest_field),
            expected_contract[contract_field],
        )
    vocabulary = manifest.get("vocabulary")
    if not isinstance(vocabulary, Mapping):
        raise ValueError("manifest vocabulary missing")
    _expect_equal("manifest vocabulary size", vocabulary.get("size"), expected_contract["vocab_size"])


def publication_is_reusable(
    output_dir: Path, manifest_path: Path, expected_contract: Mapping[str, object]
) -> bool:
    """Return true only for a complete contract- and digest-matching publication."""

    if not output_dir.is_dir() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest_contract(manifest, expected_contract)
        return list(manifest["files"]) == inventory_files(output_dir)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError, OSError):
        return False


def recover_incomplete_publication(
    output_dir: Path, manifest_path: Path, expected_contract: Mapping[str, object]
) -> bool:
    """Keep a valid publication or remove project-generated incomplete state."""

    if publication_is_reusable(output_dir, manifest_path, expected_contract):
        return True
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"refusing to replace non-directory output {output_dir}")
        shutil.rmtree(output_dir)
    if manifest_path.exists():
        if not manifest_path.is_file():
            raise ValueError(f"refusing to replace non-file manifest {manifest_path}")
        manifest_path.unlink()
    return False


def publish_staged_output(
    staged_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    manifest: Mapping[str, object],
) -> None:
    """Atomically publish staged data followed by its already-computed manifest."""

    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    if not staged_dir.is_dir():
        raise FileNotFoundError(f"staged output missing: {staged_dir}")
    if list(manifest["files"]) != inventory_files(staged_dir):
        raise ValueError("staged file inventory does not match publication manifest")
    staged_manifest = manifest_path.with_name(f".{manifest_path.name}.staging")
    atomic_json_write(staged_manifest, dict(manifest))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged_dir, output_dir)
    os.replace(staged_manifest, manifest_path)


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
    processing_contract: Mapping[str, object],
    published_output_dir: Path | None = None,
) -> dict[str, object]:
    """Build the reproducibility and integrity manifest for processed data."""

    manifest = {
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
        "processed_path": _relative_to_root(published_output_dir or output_dir, root),
        "files": inventory_files(output_dir),
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "verified": False,
    }
    manifest["source_materialization_revision"] = processing_contract[
        "source_materialization_revision"
    ]
    manifest["processing_contract"] = dict(processing_contract)
    manifest["processing_contract_sha256"] = processing_contract_sha256(
        processing_contract
    )
    return manifest


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
    expected_contract: Mapping[str, object],
    decode_samples_per_split: int = 3,
) -> dict[str, object]:
    """Fully verify row counts, lengths, token bounds, files, and decoding."""

    validate_manifest_contract(manifest, expected_contract)
    expected_splits = set(expected_contract["split_expression"])
    if set(dataset) != expected_splits:
        raise ValueError(f"processed splits must be {sorted(expected_splits)}")

    expected_length = int(expected_contract["sequence_length"])
    expected_counts = manifest["packed_sequence_counts"]
    vocabulary = manifest["vocabulary"]
    vocab_size = int(expected_contract["vocab_size"])
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

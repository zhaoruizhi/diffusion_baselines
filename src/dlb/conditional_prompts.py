"""Contracts and deterministic artifact handling for conditional-generation prompts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal, Mapping

import yaml
from pydantic import Field, StrictInt, field_validator

from dlb.schema import StrictModel

try:
    from datasets import load_from_disk
except ImportError:  # pragma: no cover - exercised only without the data extra installed.
    def load_from_disk(path: Path):
        raise RuntimeError("install the project's data dependencies to load processed datasets")


from dlb.io import atomic_json_write, sha256_file, write_compact_jsonl_atomic


OUTPUT_NAMES = {"lm1b": "lm1b-bert-128", "owt": "owt-gpt2-1024"}
SELECTION_ALGORITHM = "sha256_seed_index_digest_ascending_v1"
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonNegativeToken = Annotated[StrictInt, Field(ge=0)]


class ConditionalDataset(StrictModel):
    """The native canvas length for one conditional benchmark dataset."""

    model_length: StrictInt = Field(gt=0)


class ConditionalProtocol(StrictModel):
    """Versioned constants that define the C64 zero-shot benchmark."""

    schema_version: Literal[1]
    protocol: Literal["c64_zs_v1"]
    selection_seed: StrictInt
    sampling_seed: StrictInt
    prompt_count: Literal[1024]
    prefix_length: Literal[64]
    evaluation_continuation_length: Literal[64]
    diversity_prompt_count: Literal[256]
    completions_per_diversity_prompt: Literal[5]
    datasets: dict[str, ConditionalDataset]

    @field_validator("datasets")
    @classmethod
    def require_canonical_datasets(
        cls, value: dict[str, ConditionalDataset]
    ) -> dict[str, ConditionalDataset]:
        if set(value) != {"lm1b", "owt"}:
            raise ValueError("datasets must contain exactly lm1b and owt")
        return value


def load_protocol(path: Path) -> ConditionalProtocol:
    """Load and strictly validate a conditional protocol YAML document."""

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"could not read conditional protocol: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("conditional protocol must be a YAML mapping")
    return ConditionalProtocol.model_validate(value)


def select_source_indices(row_count: int, count: int, seed: int) -> list[int]:
    """Select unique source rows in deterministic SHA-256 priority order."""

    if row_count < count or count <= 0:
        raise ValueError("processed validation split has too few rows")
    return sorted(
        range(row_count),
        key=lambda index: hashlib.sha256(f"{seed}:{index}".encode()).digest(),
    )[:count]


class PromptRecord(StrictModel):
    """One source-aligned, fixed-prefix conditional generation prompt."""

    prompt_id: StrictInt = Field(ge=0)
    source_index: StrictInt = Field(ge=0)
    prefix_token_ids: list[NonNegativeToken] = Field(min_length=1)
    reference_token_ids: list[NonNegativeToken] = Field(min_length=1)
    source_sequence_sha256: Sha256


class PromptManifest(StrictModel):
    """Provenance and integrity binding for a deterministic prompt JSONL file."""

    schema_version: Literal[1]
    protocol: Literal["c64_zs_v1"]
    dataset: str = Field(min_length=1)
    source_split: Literal["validation"]
    source_processed_path: str = Field(min_length=1)
    source_manifest_path: str = Field(min_length=1)
    source_manifest_sha256: Sha256
    tokenizer_id: str = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)
    vocabulary_size: StrictInt = Field(gt=0)
    selection_algorithm: str = Field(min_length=1)
    selection_seed: StrictInt
    source_row_count: StrictInt = Field(gt=0)
    prompt_count: StrictInt = Field(gt=0)
    prefix_length: StrictInt = Field(gt=0)
    evaluation_continuation_length: StrictInt = Field(gt=0)
    model_length: StrictInt = Field(gt=0)
    prompt_file: str = Field(min_length=1)
    prompt_file_sha256: Sha256


def _load_yaml_mapping(path: Path, description: str) -> dict[str, object]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"could not read {description}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a YAML mapping")
    return value


def _load_json_mapping(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {description}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _dataset_contract(
    root: Path, dataset_id: str, protocol: ConditionalProtocol
) -> tuple[dict[str, object], dict[str, object], int, Path, Path]:
    if dataset_id not in protocol.datasets or dataset_id not in OUTPUT_NAMES:
        raise ValueError(f"unknown conditional dataset {dataset_id!r}")
    data_config = _load_yaml_mapping(root / "artifacts" / "data.yaml", "data configuration")
    try:
        dataset_config = data_config["datasets"][dataset_id]
        model_revisions = data_config["models"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"data configuration has no contract for {dataset_id!r}") from error
    if not isinstance(dataset_config, dict) or not isinstance(model_revisions, dict):
        raise ValueError("data configuration has malformed dataset or model contracts")
    model_length = protocol.datasets[dataset_id].model_length
    if dataset_config.get("sequence_length") != model_length:
        raise ValueError("protocol model length differs from processed data contract")
    tokenizer_id = dataset_config.get("tokenizer")
    if not isinstance(tokenizer_id, str) or not isinstance(model_revisions.get(tokenizer_id), str):
        raise ValueError("data configuration tokenizer contract is malformed")
    manifest_path = root / "data" / "manifests" / f"{dataset_id}.json"
    source_manifest = _load_json_mapping(manifest_path, "source data manifest")
    if source_manifest.get("dataset") != dataset_id:
        raise ValueError("source manifest dataset differs from requested dataset")
    if source_manifest.get("tokenizer_id") != tokenizer_id:
        raise ValueError("source manifest tokenizer differs from data configuration")
    if source_manifest.get("tokenizer_revision") != model_revisions[tokenizer_id]:
        raise ValueError("source manifest tokenizer revision differs from data configuration")
    if source_manifest.get("sequence_length") != model_length:
        raise ValueError("source manifest sequence length differs from protocol")
    vocabulary = source_manifest.get("vocabulary")
    if not isinstance(vocabulary, dict) or type(vocabulary.get("size")) is not int:
        raise ValueError("source manifest vocabulary size is missing")
    vocab_size = int(vocabulary["size"])
    if vocab_size <= 0:
        raise ValueError("source manifest vocabulary size must be positive")
    processed_path = root / "data" / "processed" / OUTPUT_NAMES[dataset_id] / "validation"
    return dataset_config, source_manifest, vocab_size, manifest_path, processed_path


def _source_sequence_sha256(token_ids: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _require_manifest_validation_count(source_manifest: Mapping[str, object], row_count: int) -> None:
    packed_counts = source_manifest.get("packed_sequence_counts")
    if not isinstance(packed_counts, Mapping) or type(packed_counts.get("validation")) is not int:
        raise ValueError("source manifest validation row count is missing")
    if packed_counts["validation"] != row_count:
        raise ValueError("source manifest validation row count differs from processed split")


def _checked_source_tokens(
    dataset: object, index: int, *, model_length: int, vocab_size: int
) -> list[int]:
    try:
        row = dataset[index]  # type: ignore[index]
        token_ids = row["input_ids"]
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f"processed validation row {index} has no input_ids") from error
    if not isinstance(token_ids, list) or len(token_ids) != model_length:
        raise ValueError(
            f"processed validation row {index} must contain exactly {model_length} input_ids"
        )
    if any(type(token_id) is not int or token_id < 0 or token_id >= vocab_size for token_id in token_ids):
        raise ValueError(f"processed validation row {index} contains a token outside vocabulary")
    return token_ids


def _artifact_paths(root: Path, dataset_id: str) -> tuple[Path, Path]:
    suffix = f"{dataset_id}-c64"
    return (
        root / "data" / "conditional" / suffix / "prompts.jsonl",
        root / "data" / "manifests" / f"conditional-{suffix}.json",
    )


def _relative_to_root(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _build_manifest(
    *,
    root: Path,
    dataset_id: str,
    protocol: ConditionalProtocol,
    source_manifest: Mapping[str, object],
    source_manifest_path: Path,
    processed_path: Path,
    vocab_size: int,
    source_row_count: int,
    prompts_path: Path,
) -> PromptManifest:
    return PromptManifest(
        schema_version=1,
        protocol=protocol.protocol,
        dataset=dataset_id,
        source_split="validation",
        source_processed_path=_relative_to_root(root, processed_path),
        source_manifest_path=_relative_to_root(root, source_manifest_path),
        source_manifest_sha256=sha256_file(source_manifest_path),
        tokenizer_id=str(source_manifest["tokenizer_id"]),
        tokenizer_revision=str(source_manifest["tokenizer_revision"]),
        vocabulary_size=vocab_size,
        selection_algorithm=SELECTION_ALGORITHM,
        selection_seed=protocol.selection_seed,
        source_row_count=source_row_count,
        prompt_count=protocol.prompt_count,
        prefix_length=protocol.prefix_length,
        evaluation_continuation_length=protocol.evaluation_continuation_length,
        model_length=protocol.datasets[dataset_id].model_length,
        prompt_file=_relative_to_root(root, prompts_path),
        prompt_file_sha256=sha256_file(prompts_path),
    )


def build_prompts(root: Path, dataset_id: str, protocol: ConditionalProtocol) -> PromptManifest:
    """Build and publish source-aligned prompt records for one dataset."""

    root = root.resolve()
    _, source_manifest, vocab_size, source_manifest_path, processed_path = _dataset_contract(
        root, dataset_id, protocol
    )
    dataset = load_from_disk(str(processed_path))
    row_count = len(dataset)
    _require_manifest_validation_count(source_manifest, row_count)
    selected = select_source_indices(row_count, protocol.prompt_count, protocol.selection_seed)
    records: list[PromptRecord] = []
    for prompt_id, source_index in enumerate(selected):
        token_ids = _checked_source_tokens(
            dataset,
            source_index,
            model_length=protocol.datasets[dataset_id].model_length,
            vocab_size=vocab_size,
        )
        records.append(
            PromptRecord(
                prompt_id=prompt_id,
                source_index=source_index,
                prefix_token_ids=token_ids[: protocol.prefix_length],
                reference_token_ids=token_ids[
                    protocol.prefix_length : protocol.prefix_length
                    + protocol.evaluation_continuation_length
                ],
                source_sequence_sha256=_source_sequence_sha256(token_ids),
            )
        )
    prompts_path, manifest_path = _artifact_paths(root, dataset_id)
    write_compact_jsonl_atomic(
        prompts_path, (record.model_dump(mode="json") for record in records)
    )
    manifest = _build_manifest(
        root=root,
        dataset_id=dataset_id,
        protocol=protocol,
        source_manifest=source_manifest,
        source_manifest_path=source_manifest_path,
        processed_path=processed_path,
        vocab_size=vocab_size,
        source_row_count=row_count,
        prompts_path=prompts_path,
    )
    atomic_json_write(manifest_path, manifest.model_dump(mode="json"))
    return manifest


def _read_prompt_records(path: Path) -> list[PromptRecord]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"could not read prompt file: {path}") from error
    if not lines:
        raise ValueError("prompt file is empty")
    records: list[PromptRecord] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise ValueError(f"prompt record line {line_number} is blank")
        try:
            value = json.loads(line, object_pairs_hook=_json_object_with_unique_keys)
            records.append(PromptRecord.model_validate(value))
        except Exception as error:
            raise ValueError(f"invalid prompt record line {line_number}: {error}") from error
    return records


def _json_object_with_unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def verify_prompts(root: Path, dataset_id: str, protocol: ConditionalProtocol) -> PromptManifest:
    """Recompute prompt/source integrity bindings and reject any drift or tampering."""

    root = root.resolve()
    _, source_manifest, vocab_size, source_manifest_path, processed_path = _dataset_contract(
        root, dataset_id, protocol
    )
    prompts_path, manifest_path = _artifact_paths(root, dataset_id)
    dataset = load_from_disk(str(processed_path))
    source_row_count = len(dataset)
    _require_manifest_validation_count(source_manifest, source_row_count)
    manifest = PromptManifest.model_validate(
        _load_json_mapping(manifest_path, "conditional prompt manifest")
    )
    expected = _build_manifest(
        root=root,
        dataset_id=dataset_id,
        protocol=protocol,
        source_manifest=source_manifest,
        source_manifest_path=source_manifest_path,
        processed_path=processed_path,
        vocab_size=vocab_size,
        source_row_count=source_row_count,
        prompts_path=prompts_path,
    )
    if manifest.source_manifest_sha256 != expected.source_manifest_sha256:
        raise ValueError("source manifest SHA-256 does not match")
    if manifest.prompt_file_sha256 != expected.prompt_file_sha256:
        raise ValueError("prompt file SHA-256 does not match")
    for field in (
        "protocol", "dataset", "source_split", "source_processed_path", "source_manifest_path",
        "tokenizer_id", "tokenizer_revision", "vocabulary_size", "selection_algorithm",
        "selection_seed", "source_row_count", "prompt_count", "prefix_length", "evaluation_continuation_length",
        "model_length", "prompt_file",
    ):
        if getattr(manifest, field) != getattr(expected, field):
            raise ValueError(f"conditional prompt manifest {field} does not match current contract")
    records = _read_prompt_records(prompts_path)
    if len(records) != protocol.prompt_count:
        raise ValueError("prompt file count does not match protocol")
    expected_indices = select_source_indices(source_row_count, protocol.prompt_count, protocol.selection_seed)
    for prompt_id, (record, expected_index) in enumerate(zip(records, expected_indices)):
        if record.prompt_id != prompt_id or record.source_index != expected_index:
            raise ValueError("prompt IDs or source selection do not match the protocol")
        if len(record.prefix_token_ids) != protocol.prefix_length:
            raise ValueError("prompt prefix length does not match protocol")
        if len(record.reference_token_ids) != protocol.evaluation_continuation_length:
            raise ValueError("prompt reference length does not match protocol")
        token_ids = _checked_source_tokens(
            dataset,
            record.source_index,
            model_length=protocol.datasets[dataset_id].model_length,
            vocab_size=vocab_size,
        )
        if record.prefix_token_ids != token_ids[: protocol.prefix_length]:
            raise ValueError("prompt prefix tokens do not match source row")
        reference_end = protocol.prefix_length + protocol.evaluation_continuation_length
        if record.reference_token_ids != token_ids[protocol.prefix_length:reference_end]:
            raise ValueError("prompt reference tokens do not match source row")
        if record.source_sequence_sha256 != _source_sequence_sha256(token_ids):
            raise ValueError("prompt source sequence SHA-256 does not match source row")
    return manifest

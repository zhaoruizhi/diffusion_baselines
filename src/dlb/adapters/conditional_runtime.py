"""Runtime helpers for zero-shot hard-prefix conditional generation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator, Sequence

from dlb.conditional_prompts import PromptManifest, PromptRecord
from dlb.io import SampleValidationError, sha256_file


@dataclass(frozen=True)
class ConditioningBatch:
    """A device-materialized batch of immutable prompt conditioning tensors."""

    prompt_ids: tuple[int, ...]
    source_indices: tuple[int, ...]
    prefix_token_ids: object
    reference_token_ids: object
    completion_id: int

    @property
    def prefix(self) -> object:
        return self.prefix_token_ids

    @property
    def reference(self) -> object:
        return self.reference_token_ids


def _require_torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - server environments install torch.
        raise RuntimeError("conditional runtime helpers require torch") from error
    return torch


def _shape(value: object) -> tuple[int, ...]:
    return tuple(int(part) for part in value.shape)  # type: ignore[attr-defined]


def clamp_token_prefix(state, prefix_ids):
    """Return a copy of a rank-2 token state with the prompt prefix restored."""

    if getattr(state, "ndim", None) != 2 or getattr(prefix_ids, "ndim", None) != 2:
        raise ValueError("token state and prefix must be rank two")
    if state.shape[0] != prefix_ids.shape[0] or state.shape[1] < prefix_ids.shape[1]:
        raise ValueError("token state is incompatible with prefix batch")
    result = state.clone()
    result[:, : prefix_ids.shape[1]] = prefix_ids.to(device=state.device, dtype=state.dtype)
    return result


def clamp_vocab_prefix(state, prefix_ids):
    """Return a copy of a rank-3 vocabulary state with clean one-hot prefix rows."""

    torch = _require_torch()
    if getattr(state, "ndim", None) != 3:
        raise ValueError("vocabulary state must be rank three")
    if getattr(prefix_ids, "ndim", None) != 2:
        raise ValueError("vocabulary prefix must be rank two")
    if state.shape[0] != prefix_ids.shape[0] or state.shape[1] < prefix_ids.shape[1]:
        raise ValueError("vocabulary state is incompatible with prefix batch")
    if int(prefix_ids.max().item()) >= state.shape[-1] or int(prefix_ids.min().item()) < 0:
        raise ValueError("prefix token is outside vocabulary dimension")
    clean = torch.nn.functional.one_hot(prefix_ids, num_classes=state.shape[-1]).to(state)
    result = state.clone()
    result[:, : prefix_ids.shape[1], :] = clean
    return result


def base_n_digits(token_ids, *, base: int, digits_per_token: int):
    """Encode rank-2 token IDs into RDLM's flattened base-n digit rows."""

    torch = _require_torch()
    if getattr(token_ids, "ndim", None) != 2:
        raise ValueError("token IDs must be rank two")
    if type(base) is not int or base <= 1:
        raise ValueError("base must be greater than one")
    if type(digits_per_token) is not int or digits_per_token <= 0:
        raise ValueError("digits_per_token must be positive")
    powers = base ** torch.arange(digits_per_token, device=token_ids.device)
    digits = (token_ids.unsqueeze(-1) // powers) % base
    return digits.reshape(token_ids.shape[0], -1)


def clamp_rdlm_prefix(state, prefix_ids, *, base: int, digits_per_token: int):
    """Return a copy of an RDLM manifold state with clean base-n prefix digits."""

    torch = _require_torch()
    if getattr(state, "ndim", None) != 3:
        raise ValueError("RDLM state must be rank three")
    prefix_digits = base_n_digits(
        prefix_ids.to(device=state.device),
        base=base,
        digits_per_token=digits_per_token,
    )
    if state.shape[0] != prefix_digits.shape[0] or state.shape[1] < prefix_digits.shape[1]:
        raise ValueError("RDLM state is incompatible with prefix batch")
    if int(prefix_digits.max().item()) >= state.shape[-1] or int(prefix_digits.min().item()) < 0:
        raise ValueError("RDLM prefix digit is outside state dimension")
    clean = torch.nn.functional.one_hot(prefix_digits, num_classes=state.shape[-1]).to(state)
    result = state.clone()
    result[:, : prefix_digits.shape[1], :] = clean
    return result


def clamp_embedding_prefix(state, clean_embeddings):
    """Return a copy of an embedding state with clean prefix embeddings restored."""

    if getattr(state, "ndim", None) != 3 or getattr(clean_embeddings, "ndim", None) != 3:
        raise ValueError("embedding state and clean prefix must be rank three")
    if (
        state.shape[0] != clean_embeddings.shape[0]
        or state.shape[1] < clean_embeddings.shape[1]
        or state.shape[2] != clean_embeddings.shape[2]
    ):
        raise ValueError("embedding state is incompatible with clean prefix embeddings")
    result = state.clone()
    result[:, : clean_embeddings.shape[1], :] = clean_embeddings.to(state)
    return result


def token_project_fn(prefix_ids):
    """Build a repeatable token-prefix projector closure."""

    if getattr(prefix_ids, "ndim", None) != 2:
        raise ValueError("token prefix must be rank two")
    prefix = prefix_ids.clone()

    def project(state):
        return clamp_token_prefix(state, prefix)

    project.conditioning_implementation = "zero_shot_runtime_projection"  # type: ignore[attr-defined]
    return project


def vocab_project_fn(prefix_ids):
    """Build a repeatable one-hot-prefix projector closure."""

    if getattr(prefix_ids, "ndim", None) != 2:
        raise ValueError("vocabulary prefix must be rank two")
    prefix = prefix_ids.clone()

    def project(state):
        return clamp_vocab_prefix(state, prefix.to(device=state.device))

    project.conditioning_implementation = "zero_shot_runtime_projection"  # type: ignore[attr-defined]
    return project


def rdlm_project_fn(prefix_ids, *, base: int, digits_per_token: int):
    """Build a repeatable RDLM base-n digit prefix projector closure."""

    if getattr(prefix_ids, "ndim", None) != 2:
        raise ValueError("RDLM prefix must be rank two")
    prefix = prefix_ids.clone()

    def project(state):
        return clamp_rdlm_prefix(
            state,
            prefix.to(device=state.device),
            base=base,
            digits_per_token=digits_per_token,
        )

    project.conditioning_implementation = "zero_shot_runtime_projection"  # type: ignore[attr-defined]
    return project


def embedding_project_fn(clean_embeddings):
    """Build a repeatable embedding-prefix projector closure."""

    if getattr(clean_embeddings, "ndim", None) != 3:
        raise ValueError("clean embedding prefix must be rank three")
    clean = clean_embeddings.clone()

    def project(state):
        return clamp_embedding_prefix(state, clean.to(device=state.device, dtype=state.dtype))

    project.conditioning_implementation = "zero_shot_runtime_projection"  # type: ignore[attr-defined]
    return project


def candi_prompt_mask(prefix_ids):
    """Return the numeric prompt mask expected by CANDI's upstream sampler."""

    torch = _require_torch()
    if getattr(prefix_ids, "ndim", None) != 2:
        raise ValueError("CANDI prompt prefix must be rank two")
    return torch.ones_like(prefix_ids, dtype=torch.float32)


@contextmanager
def patched_attribute(owner: object, name: str, replacement: object) -> Iterator[None]:
    """Temporarily replace an attribute and always restore the original value."""

    original = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        setattr(owner, name, original)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SampleValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_prompt_records(manifest: PromptManifest, root: Path) -> list[PromptRecord]:
    prompt_path = root / manifest.prompt_file
    if prompt_path.is_symlink() or not prompt_path.is_file():
        raise ValueError(f"conditional prompt file is missing or unsafe: {prompt_path}")
    if sha256_file(prompt_path) != manifest.prompt_file_sha256:
        raise ValueError("conditional prompt file SHA-256 differs from manifest")
    records: list[PromptRecord] = []
    with prompt_path.open("r", encoding="utf-8") as handle:
        for index, raw_line in enumerate(handle):
            if not raw_line.strip():
                raise ValueError(f"conditional prompt record {index} is blank")
            value = json.loads(raw_line, object_pairs_hook=_unique_object)
            record = PromptRecord.model_validate(value)
            if record.prompt_id != index:
                raise ValueError(f"expected prompt_id {index}, found {record.prompt_id}")
            records.append(record)
    if len(records) != manifest.prompt_count:
        raise ValueError(
            f"expected {manifest.prompt_count} prompt records, found {len(records)}"
        )
    return records


def _schedule_limit(manifest: PromptManifest, completion_id: int) -> int:
    if type(completion_id) is not int or completion_id < 0 or completion_id > 4:
        raise ValueError("completion_id must be an integer in [0, 4]")
    if completion_id == 0:
        return manifest.prompt_count
    return min(256, manifest.prompt_count)


def load_conditioning_batch(
    manifest_path: Path,
    expected_manifest_sha256: str,
    *,
    completion_id: int,
    prompt_start: int,
    batch_size: int,
    device: str,
    vocab_size: int,
) -> ConditioningBatch:
    """Load one exact prompt/completion batch from a SHA-bound prompt manifest."""

    torch = _require_torch()
    manifest_path = manifest_path.resolve()
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"conditional prompt manifest is missing or unsafe: {manifest_path}")
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("conditional prompt manifest SHA-256 differs from request")
    if type(prompt_start) is not int or prompt_start < 0:
        raise ValueError("prompt_start must be a non-negative integer")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("vocab_size must be a positive integer")
    manifest = PromptManifest.model_validate(
        json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    )
    manifest_vocab_size = int(manifest.vocabulary_size)
    if vocab_size < manifest_vocab_size:
        raise ValueError("conditioning runtime vocabulary size is smaller than prompt manifest")
    limit = _schedule_limit(manifest, completion_id)
    if prompt_start + batch_size > limit:
        raise ValueError("conditioning batch crosses its completion schedule boundary")
    records = _read_prompt_records(manifest, manifest_path.parents[2])
    selected = records[prompt_start : prompt_start + batch_size]
    prefix_rows = [record.prefix_token_ids for record in selected]
    reference_rows = [record.reference_token_ids for record in selected]
    for row in (*prefix_rows, *reference_rows):
        if len(row) != 64 or any(token_id >= manifest_vocab_size for token_id in row):
            raise ValueError("conditioning prompt tokens violate length or vocabulary bounds")
    return ConditioningBatch(
        prompt_ids=tuple(record.prompt_id for record in selected),
        source_indices=tuple(record.source_index for record in selected),
        prefix_token_ids=torch.as_tensor(prefix_rows, dtype=torch.long, device=device),
        reference_token_ids=torch.as_tensor(reference_rows, dtype=torch.long, device=device),
        completion_id=completion_id,
    )

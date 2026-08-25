"""Server-only runtime shared by the SDTT and Di4C sampling entrypoints."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
from collections.abc import Mapping

import yaml

from dlb.adapters.conditional_runtime import load_conditioning_batch, token_project_fn


class TokenizerBinding:
    __slots__ = ("tokenizer_id", "revision", "snapshot")

    def __init__(self, tokenizer_id: str, revision: str, snapshot: Path) -> None:
        self.tokenizer_id = tokenizer_id
        self.revision = revision
        self.snapshot = snapshot


def require_file(path: Path, label: str) -> Path:
    path = path.absolute()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or unsafe: {path}")
    return path


def require_directory(path: Path, label: str) -> Path:
    path = path.absolute()
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} is missing or unsafe: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with require_file(path, "hashed file").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str, label: str) -> Path:
    path = require_file(path, label)
    if (
        len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
        or sha256_file(path) != expected
    ):
        raise ValueError(f"{label} SHA-256 differs from canonical provenance")
    return path


def parse_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError("boolean values must be 'true' or 'false'")


@contextmanager
def offline_huggingface(enabled: bool):
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    if enabled:
        os.environ.update({name: "1" for name in names})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def install_upstream(upstream_root: Path) -> None:
    source = require_directory(upstream_root / "src", "upstream Python source")
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy

        numpy.random.seed(seed % (2**32))
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(snapshot: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(require_directory(snapshot, "locked tokenizer snapshot")),
        local_files_only=True,
        trust_remote_code=False,
    )


def load_config(path: Path, expected_sha256: str):
    from omegaconf import OmegaConf

    return OmegaConf.load(verify_sha256(path, expected_sha256, "sampling config"))


def checkpoint_state(
    path: Path, expected_sha256: str
) -> tuple[dict[str, object], object | None]:
    path = verify_sha256(path, expected_sha256, "student checkpoint")
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(path))), None
    import torch

    # Lightning 2.2 checkpoints contain OmegaConf DictConfig objects. Full pickle
    # loading is deliberately restricted to bytes authenticated above by the
    # canonical runner/checkpoint manifest provenance.
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("student checkpoint is not a mapping")
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise ValueError("student checkpoint state_dict is invalid")
    config = None
    hyperparameters = payload.get("hyper_parameters")
    if isinstance(hyperparameters, dict):
        config = hyperparameters.get("config")
    return dict(state), config


_MISSING = object()
_STANDARD_REFERENCE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_.]*)\}$")


def _select_config_value(
    config: object, path: str, references: frozenset[str] = frozenset()
) -> object:
    """Resolve one selected scalar without walking unrelated Hydra config nodes."""

    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(config):
            try:
                return OmegaConf.select(
                    config,
                    path,
                    default=_MISSING,
                    throw_on_resolution_failure=True,
                )
            except Exception as error:
                raise ValueError(
                    f"sampling config field {path} cannot be resolved"
                ) from error
    except ImportError:
        pass
    value = config
    for component in path.split("."):
        try:
            value = (
                value[component]
                if isinstance(value, Mapping)
                else getattr(value, component)
            )
        except (AttributeError, KeyError, TypeError):
            return _MISSING
    if isinstance(value, str):
        reference = _STANDARD_REFERENCE.fullmatch(value)
        if reference is not None:
            target = reference.group(1)
            if target in references or target == path:
                raise ValueError(f"sampling config has a cyclic reference at {path}")
            return _select_config_value(config, target, references | {path})
    return value


def _nested(config: object, path: str) -> object:
    value = _select_config_value(config, path)
    if value is _MISSING:
        raise ValueError(f"sampling config is missing required field {path}")
    return value


def _optional_nested(config: object, path: str) -> object:
    return _select_config_value(config, path)


_CHECKPOINT_CONFIG_FIELDS = (
    "T",
    "time_conditioning",
    "model.type",
    "model.hidden_size",
    "model.cond_dim",
    "model.length",
    "model.n_blocks",
    "model.n_heads",
    "model.scale_by_sigma",
    "model.dropout",
    "model.tie_word_embeddings",
    "noise.type",
    "noise.sigma_min",
    "noise.sigma_max",
    "training.ema",
    "training.antithetic_sampling",
    "training.importance_sampling",
    "training.sampling_eps",
    "training.change_of_variables",
    "tokenizer.name",
    "parameterization.name",
    "parameterization.log_loss_buckets",
    "parameterization.start_from_hf",
    "parameterization.distill_mode",
    "parameterization.num_distill_steps",
    "parameterization.min_num_sampling_steps",
    "parameterization.grow_dt_every",
    "parameterization.orig_num_sampling_steps",
    "parameterization.sampling_mode",
    "parameterization.loss_precision",
    "parameterization.reset_optimizer_on_growth",
    "parameterization.use_ema_on_growth",
)

_OPTIONAL_CHECKPOINT_CONFIG_FIELDS = ("model.causal", "is_di4c")


def validate_embedded_config(
    authoritative: object,
    embedded: object | None,
    *,
    allow_missing_embedded_config: bool = False,
) -> None:
    if embedded is None:
        if allow_missing_embedded_config:
            return
        raise ValueError("Lightning checkpoint does not embed its training config")
    for path in (*_CHECKPOINT_CONFIG_FIELDS, *_OPTIONAL_CHECKPOINT_CONFIG_FIELDS):
        expected = _optional_nested(authoritative, path)
        if expected is _MISSING:
            continue
        observed = _optional_nested(embedded, path)
        if observed is _MISSING:
            raise ValueError(f"embedded checkpoint config is missing required field {path}")
        if observed != expected:
            raise ValueError(
                f"embedded checkpoint config differs at {path}: "
                f"expected {expected!r}, found {observed!r}"
            )


def validate_sampling_config(
    config: object,
    *,
    binding: TokenizerBinding,
    sequence_length: int,
    require_di4c: bool,
) -> None:
    for path in _CHECKPOINT_CONFIG_FIELDS:
        _nested(config, path)
    tokenizer_name = _nested(config, "tokenizer.name")
    if tokenizer_name != binding.tokenizer_id and Path(str(tokenizer_name)).resolve() != binding.snapshot.resolve():
        raise ValueError("sampling config tokenizer differs from canonical data contract")
    if _nested(config, "model.length") != sequence_length:
        raise ValueError("sampling config sequence length differs from canonical dataset")
    if _nested(config, "data_preprocess.seq_len") != sequence_length:
        raise ValueError("sampling config preprocessing length differs from canonical dataset")
    if _nested(config, "parameterization.sampling_mode") != "ancestral":
        raise ValueError("sampling config does not use ancestral sampling")
    di4c_identity = _optional_nested(config, "is_di4c")
    if require_di4c:
        if di4c_identity is not True:
            raise ValueError("sampling config Di4C identity is inconsistent")
        if _nested(config, "parameterization.start_from_hf") is not False:
            raise ValueError("Di4C sampling config must instantiate the local student")
    elif di4c_identity not in {_MISSING, False}:
        raise ValueError("sampling config Di4C identity is inconsistent")


def load_tokenizer_binding(
    data_config_path: Path,
    downloads_manifest_path: Path,
    dataset_id: str,
    requested_snapshot: Path,
) -> TokenizerBinding:
    data_config_path = require_file(data_config_path, "data config")
    downloads_manifest_path = require_file(
        downloads_manifest_path, "download manifest"
    )
    try:
        configuration = yaml.safe_load(data_config_path.read_text(encoding="utf-8"))
        downloads = json.loads(downloads_manifest_path.read_text(encoding="utf-8"))
        tokenizer_id = configuration["datasets"][dataset_id]["tokenizer"]
        revision = configuration["models"][tokenizer_id]
        record = downloads["models"][tokenizer_id]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("locked tokenizer metadata is invalid") from error
    if downloads.get("schema_version") != 1:
        raise ValueError("download manifest schema is not supported")
    if (
        not isinstance(tokenizer_id, str)
        or not isinstance(revision, str)
        or len(revision) != 40
    ):
        raise ValueError("data config tokenizer binding is invalid")
    if record.get("repo_id") != tokenizer_id or record.get("revision") != revision:
        raise ValueError("download manifest tokenizer binding differs from data config")
    recorded = record.get("snapshot_path")
    if not isinstance(recorded, str) or not recorded:
        raise ValueError("download manifest tokenizer snapshot is missing")
    snapshot = Path(recorded)
    if not snapshot.is_absolute():
        snapshot = data_config_path.parents[1] / snapshot
    requested_snapshot = requested_snapshot.absolute()
    if requested_snapshot != snapshot:
        raise ValueError("requested tokenizer snapshot differs from download manifest")
    parts = snapshot.parts
    try:
        recorded_revision = parts[parts.index("snapshots") + 1]
    except (ValueError, IndexError) as error:
        raise ValueError("download manifest tokenizer snapshot path is invalid") from error
    if recorded_revision != revision:
        raise ValueError("download manifest tokenizer snapshot revision differs from data config")
    require_directory(snapshot, "locked tokenizer snapshot")
    return TokenizerBinding(tokenizer_id, revision, snapshot)


def instantiate_without_training_teacher(model_type, config, tokenizer):
    """Instantiate the released student architecture without loading a network teacher."""

    original = model_type.prepare_teacher_and_student

    def sampling_only(self):
        self.teacher = []

    model_type.prepare_teacher_and_student = sampling_only
    try:
        return model_type(config, tokenizer, verbose=False)
    finally:
        model_type.prepare_teacher_and_student = original


def normalize_state_dict(state: dict[str, object]) -> dict[str, object]:
    return {key.replace("_orig_mod.", ""): value for key, value in state.items()}


def write_capture_atomic(
    output: Path,
    *,
    model,
    tokenizer,
    sample_count: int,
    batch_size: int,
    num_steps: int,
    seq_len: int,
    sampler: str,
    generation_mode: str = "unconditional",
    conditioning_manifest: Path | None = None,
    conditioning_manifest_sha256: str | None = None,
    prompt_count: int | None = None,
    diversity_prompt_count: int | None = None,
    completions_per_diversity_prompt: int | None = None,
) -> None:
    """Sample exact batches, immediately move IDs to CPU, and atomically publish JSON."""

    if min(sample_count, batch_size, num_steps, seq_len) <= 0:
        raise ValueError("sampling counts and lengths must be positive")
    if sampler != "ancestral":
        raise ValueError("distilled language baselines require ancestral sampling")
    if generation_mode not in {"unconditional", "conditional_prefix"}:
        raise ValueError(f"unsupported generation mode: {generation_mode!r}")
    output = output.absolute()
    if not output.is_absolute() or output.is_symlink():
        raise ValueError(f"output path is unsafe: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    temporary = Path(temporary_name)
    written = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write('{"schema":"dlb-upstream-token-capture-v1","samples":[')
            if generation_mode == "conditional_prefix":
                if (
                    conditioning_manifest is None
                    or conditioning_manifest_sha256 is None
                    or prompt_count is None
                    or diversity_prompt_count is None
                    or completions_per_diversity_prompt is None
                ):
                    raise ValueError("conditional distilled sampling requires a prompt contract")
                batches: list[tuple[int, int, int]] = []
                for start in range(0, prompt_count, batch_size):
                    batches.append((0, start, min(batch_size, prompt_count - start)))
                for completion_id in range(1, completions_per_diversity_prompt):
                    for start in range(0, diversity_prompt_count, batch_size):
                        batches.append(
                            (
                                completion_id,
                                start,
                                min(batch_size, diversity_prompt_count - start),
                            )
                        )
            else:
                batches = [(-1, written, min(batch_size, sample_count - written)) for written in range(0, sample_count, batch_size)]
            for completion_id, prompt_start, current in batches:
                current = min(batch_size, sample_count - written)
                batch = None
                project_fn = lambda value: value
                if generation_mode == "conditional_prefix":
                    batch = load_conditioning_batch(
                        conditioning_manifest,
                        conditioning_manifest_sha256,
                        completion_id=completion_id,
                        prompt_start=prompt_start,
                        batch_size=current,
                        device=str(model.device),
                        vocab_size=len(tokenizer),
                    )
                    project_fn = token_project_fn(batch.prefix_token_ids)
                result = project_fn(
                    model.sample(
                        n_samples=current,
                        num_steps=num_steps,
                        seq_len=seq_len,
                        sampler=sampler,
                        verbose=False,
                        project_fn=project_fn,
                    )
                )
                rows = result.detach().cpu().tolist()
                if not isinstance(rows, list) or len(rows) != current:
                    raise ValueError("upstream sampler returned the wrong batch size")
                texts = list(tokenizer.batch_decode(rows))
                if len(texts) != current:
                    raise ValueError("tokenizer returned the wrong decoded batch size")
                for batch_index, (row, text) in enumerate(zip(rows, texts, strict=True)):
                    if not isinstance(row, list) or len(row) != seq_len:
                        raise ValueError("upstream sampler returned a noncanonical sequence")
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError("upstream tokenizer returned empty text")
                    payload = {"sample_id": written, "text": text, "token_ids": row}
                    if batch is not None:
                        prefix = batch.prefix_token_ids[batch_index].detach().cpu().tolist()
                        reference = batch.reference_token_ids[batch_index].detach().cpu().tolist()
                        if row[: len(prefix)] != prefix:
                            raise ValueError("conditional sampler returned a prefix mismatch")
                        payload.update(
                            {
                                "prompt_id": batch.prompt_ids[batch_index],
                                "completion_id": completion_id,
                                "source_index": batch.source_indices[batch_index],
                                "prefix_token_ids": prefix,
                                "reference_token_ids": reference,
                                "full_token_ids": row,
                                "prefix_text": tokenizer.decode(prefix),
                                "continuation_text": tokenizer.decode(row[len(prefix) :]),
                                "reference_text": tokenizer.decode(reference),
                                "full_text": text,
                                "prefix_exact_match": True,
                            }
                        )
                    if written:
                        handle.write(",")
                    json.dump(
                        payload,
                        handle,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    written += 1
                del result, rows, texts
            handle.write("]}")
            handle.flush()
            os.fsync(handle.fileno())
        if written != sample_count:
            raise ValueError(f"expected {sample_count} samples, wrote {written}")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def benchmark_model(
    *,
    model,
    output: Path,
    metadata_path: Path,
    precision: str,
    num_steps: int,
    seq_len: int,
    sampler: str,
    generation_mode: str = "unconditional",
    conditioning_manifest: Path | None = None,
    conditioning_manifest_sha256: str | None = None,
    prompt_count: int | None = None,
    diversity_prompt_count: int | None = None,
    completions_per_diversity_prompt: int | None = None,
):
    """Time the already-materialized student's real one-sample tensor callback."""

    if min(num_steps, seq_len) <= 0 or sampler != "ancestral":
        raise ValueError("distilled benchmark sampling arguments are invalid")
    from dlb.timing import benchmark_and_publish

    project_fn = lambda value: value
    if generation_mode == "conditional_prefix":
        if conditioning_manifest is None or conditioning_manifest_sha256 is None:
            raise ValueError("conditional benchmark requires a prompt manifest")
        del prompt_count, diversity_prompt_count, completions_per_diversity_prompt
        batch = load_conditioning_batch(
            conditioning_manifest,
            conditioning_manifest_sha256,
            completion_id=0,
            prompt_start=0,
            batch_size=1,
            device=str(model.device),
            vocab_size=len(model.tokenizer),
        )
        project_fn = token_project_fn(batch.prefix_token_ids)
    return benchmark_and_publish(
        lambda: project_fn(model.sample(
            n_samples=1,
            num_steps=num_steps,
            seq_len=seq_len,
            sampler=sampler,
            verbose=False,
            project_fn=project_fn,
        )),
        model=model,
        output=output,
        metadata_path=metadata_path,
        precision=precision,
    )


def configure_for_sampling(config, *, tokenizer_snapshot: Path, seq_len: int) -> None:
    config.tokenizer.name = str(tokenizer_snapshot)
    config.model.length = seq_len
    config.data_preprocess.seq_len = seq_len
    config.mode = "sample"
    config.compile = False


def materialize_model(
    *,
    model_type,
    config,
    tokenizer,
    state: dict[str, object],
    strict: bool,
):
    model = instantiate_without_training_teacher(model_type, config, tokenizer)
    result = model.load_state_dict(normalize_state_dict(state), strict=strict)
    if not strict:
        missing = [key for key in result.missing_keys if not key.startswith("teacher.")]
        unexpected = [key for key in result.unexpected_keys if not key.startswith("teacher.")]
        if missing or unexpected:
            raise ValueError(
                "student checkpoint does not match the configured architecture: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
    model.eval()
    if not __import__("torch").cuda.is_available():
        raise RuntimeError("distilled sampling requires a CUDA server")
    model.cuda()
    return model

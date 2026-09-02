"""Run a pinned upstream entrypoint while capturing returned token-ID samples."""

from __future__ import annotations

import builtins
from contextlib import contextmanager
from contextlib import ExitStack
from contextlib import nullcontext
from dataclasses import dataclass
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Callable

import yaml

from dlb.io import atomic_json_write
from dlb.adapters.conditional_runtime import (
    adapt_candi_generate_sample_prompt,
    candi_prompt_conditioning,
    clamp_token_prefix,
    embedding_project_fn,
    load_conditioning_batch,
    patched_attribute,
    rdlm_project_fn,
    token_project_fn,
    vocab_project_fn,
)
from dlb.timing import benchmark_and_publish


@dataclass(frozen=True)
class CaptureInvocation:
    entrypoint: Path
    capture_path: Path
    kind: str = "teacher"
    expected_samples: int | None = None
    saved_config_path: Path | None = None
    saved_sde_path: Path | None = None
    data_config_path: Path | None = None
    downloads_manifest_path: Path | None = None
    dataset_id: str | None = None
    tokenizer_snapshot: Path | None = None
    benchmark_output: Path | None = None
    benchmark_metadata: Path | None = None
    benchmark_precision: str | None = None
    generation_mode: str = "unconditional"
    conditioning_manifest: Path | None = None
    conditioning_manifest_sha256: str | None = None
    conditioning_config_sha256: str | None = None
    prefix_length: int | None = None
    evaluation_continuation_length: int | None = None
    prompt_count: int | None = None
    diversity_prompt_count: int | None = None
    completions_per_diversity_prompt: int | None = None
    completion_schedule: str | None = None


@dataclass(frozen=True)
class TokenizerBinding:
    tokenizer_id: str
    revision: str
    snapshot: Path


def _split_arguments(argv: list[str]) -> tuple[CaptureInvocation, list[str]]:
    try:
        separator = argv.index("--")
    except ValueError as error:
        raise ValueError("capture wrapper requires -- before upstream arguments") from error
    wrapper = argv[:separator]
    forwarded = argv[separator + 1 :]
    allowed = {
        "--upstream-entrypoint",
        "--capture-path",
        "--capture-kind",
        "--expected-samples",
        "--saved-config-path",
        "--saved-sde-path",
        "--data-config-path",
        "--downloads-manifest-path",
        "--dataset-id",
        "--tokenizer-snapshot",
        "--benchmark-output",
        "--benchmark-metadata",
        "--benchmark-precision",
        "--generation-mode",
        "--conditioning-manifest",
        "--conditioning-manifest-sha256",
        "--conditioning-config-sha256",
        "--prefix-length",
        "--evaluation-continuation-length",
        "--prompt-count",
        "--diversity-prompt-count",
        "--completions-per-diversity-prompt",
        "--completion-schedule",
    }
    values: dict[str, str] = {}
    for argument in wrapper:
        key, marker, value = argument.partition("=")
        if not marker or key not in allowed or not value:
            raise ValueError(f"invalid capture-wrapper argument: {argument!r}")
        if key in values:
            raise ValueError(f"duplicate capture-wrapper argument: {key}")
        values[key] = value
    required = {"--upstream-entrypoint", "--capture-path"}
    if not required.issubset(values):
        raise ValueError("capture wrapper requires entrypoint and capture path")

    entrypoint = Path(values["--upstream-entrypoint"])
    capture_path = Path(values["--capture-path"])
    if not entrypoint.is_absolute() or entrypoint.is_symlink() or not entrypoint.is_file():
        raise ValueError(f"upstream entrypoint is missing or unsafe: {entrypoint}")
    if not capture_path.is_absolute():
        raise ValueError("capture path must be absolute")
    kind = values.get("--capture-kind", "teacher")
    if kind not in {"teacher", "langflow", "rdlm"}:
        raise ValueError(f"unsupported capture kind: {kind!r}")
    expected_text = values.get("--expected-samples")
    if expected_text is None:
        expected_samples = None
    else:
        try:
            expected_samples = int(expected_text)
        except ValueError as error:
            raise ValueError("expected sample count must be an integer") from error
        if expected_samples <= 0:
            raise ValueError("expected sample count must be positive")
    saved_config_path = (
        Path(values["--saved-config-path"])
        if "--saved-config-path" in values
        else None
    )
    saved_sde_path = (
        Path(values["--saved-sde-path"])
        if "--saved-sde-path" in values
        else None
    )
    data_config_path = (
        Path(values["--data-config-path"])
        if "--data-config-path" in values
        else None
    )
    downloads_manifest_path = (
        Path(values["--downloads-manifest-path"])
        if "--downloads-manifest-path" in values
        else None
    )
    dataset_id = values.get("--dataset-id")
    tokenizer_snapshot = (
        Path(values["--tokenizer-snapshot"])
        if "--tokenizer-snapshot" in values
        else None
    )
    benchmark_output = (
        Path(values["--benchmark-output"])
        if "--benchmark-output" in values
        else None
    )
    benchmark_metadata = (
        Path(values["--benchmark-metadata"])
        if "--benchmark-metadata" in values
        else None
    )
    benchmark_precision = values.get("--benchmark-precision")
    benchmark_fields = (benchmark_output, benchmark_metadata, benchmark_precision)
    if any(value is not None for value in benchmark_fields) and not all(
        value is not None for value in benchmark_fields
    ):
        raise ValueError("benchmark capture arguments must be provided together")
    if benchmark_output is not None:
        if not benchmark_output.is_absolute() or not benchmark_metadata.is_absolute():
            raise ValueError("benchmark paths must be absolute")
        if benchmark_precision != "author":
            raise ValueError("benchmark precision is invalid")
    if kind in {"langflow", "rdlm"} and expected_samples is None:
        raise ValueError(f"{kind} capture requires an expected sample count")
    if kind == "rdlm" and (saved_config_path is None or saved_sde_path is None):
        raise ValueError("RDLM capture requires saved config and SDE paths")
    generation_mode = values.get("--generation-mode", "unconditional")
    if generation_mode not in {"unconditional", "conditional_prefix"}:
        raise ValueError(f"unsupported generation mode: {generation_mode!r}")
    conditioning_manifest = (
        Path(values["--conditioning-manifest"])
        if "--conditioning-manifest" in values
        else None
    )

    def optional_int(name: str) -> int | None:
        text = values.get(name)
        if text is None:
            return None
        try:
            parsed = int(text)
        except ValueError as error:
            raise ValueError(f"{name} must be an integer") from error
        if parsed < 0:
            raise ValueError(f"{name} must be non-negative")
        return parsed

    conditional_fields = (
        conditioning_manifest,
        values.get("--conditioning-manifest-sha256"),
        values.get("--conditioning-config-sha256"),
        optional_int("--prefix-length"),
        optional_int("--evaluation-continuation-length"),
        optional_int("--prompt-count"),
        optional_int("--diversity-prompt-count"),
        optional_int("--completions-per-diversity-prompt"),
        values.get("--completion-schedule"),
    )
    if generation_mode == "conditional_prefix":
        if any(field is None for field in conditional_fields):
            raise ValueError("conditional capture requires the full conditioning contract")
        if conditioning_manifest is None or not conditioning_manifest.is_absolute():
            raise ValueError("conditional manifest path must be absolute")
    elif any(field is not None for field in conditional_fields):
        raise ValueError("unconditional capture must not include conditional fields")
    tokenizer_fields = (
        data_config_path,
        downloads_manifest_path,
        dataset_id,
        tokenizer_snapshot,
    )
    if kind in {"langflow", "rdlm"} and any(value is None for value in tokenizer_fields):
        raise ValueError(f"{kind} capture requires a locked tokenizer snapshot")
    return (
        CaptureInvocation(
            entrypoint=entrypoint,
            capture_path=capture_path,
            kind=kind,
            expected_samples=expected_samples,
            saved_config_path=saved_config_path,
            saved_sde_path=saved_sde_path,
            data_config_path=data_config_path,
            downloads_manifest_path=downloads_manifest_path,
            dataset_id=dataset_id,
            tokenizer_snapshot=tokenizer_snapshot,
            benchmark_output=benchmark_output,
            benchmark_metadata=benchmark_metadata,
            benchmark_precision=benchmark_precision,
            generation_mode=generation_mode,
            conditioning_manifest=conditioning_manifest,
            conditioning_manifest_sha256=values.get("--conditioning-manifest-sha256"),
            conditioning_config_sha256=values.get("--conditioning-config-sha256"),
            prefix_length=optional_int("--prefix-length"),
            evaluation_continuation_length=optional_int("--evaluation-continuation-length"),
            prompt_count=optional_int("--prompt-count"),
            diversity_prompt_count=optional_int("--diversity-prompt-count"),
            completions_per_diversity_prompt=optional_int("--completions-per-diversity-prompt"),
            completion_schedule=values.get("--completion-schedule"),
        ),
        forwarded,
    )


def _require_absolute_file(path: Path | None, label: str) -> Path:
    if path is None or not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or unsafe: {path}")
    return path


def _load_tokenizer_binding(invocation: CaptureInvocation) -> TokenizerBinding:
    """Resolve one local snapshot against both immutable project data records."""

    config_path = _require_absolute_file(invocation.data_config_path, "data config")
    downloads_path = _require_absolute_file(
        invocation.downloads_manifest_path, "download manifest"
    )
    try:
        configuration = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        downloads = json.loads(downloads_path.read_text(encoding="utf-8"))
        dataset = configuration["datasets"][invocation.dataset_id]
        tokenizer_id = dataset["tokenizer"]
        revision = configuration["models"][tokenizer_id]
        record = downloads["models"][tokenizer_id]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("locked tokenizer metadata is invalid") from error
    if downloads.get("schema_version") != 1:
        raise ValueError("download manifest schema is not supported")
    if not isinstance(tokenizer_id, str) or not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("data config tokenizer binding is invalid")
    if record.get("repo_id") != tokenizer_id or record.get("revision") != revision:
        raise ValueError("download manifest tokenizer binding differs from data config")
    recorded_snapshot = record.get("snapshot_path")
    if not isinstance(recorded_snapshot, str) or not recorded_snapshot:
        raise ValueError("download manifest tokenizer snapshot is missing")
    snapshot_path = Path(recorded_snapshot)
    if not snapshot_path.is_absolute():
        snapshot_path = config_path.parents[1] / snapshot_path
    requested_snapshot = invocation.tokenizer_snapshot
    if requested_snapshot is None or not requested_snapshot.is_absolute():
        raise ValueError("tokenizer snapshot path must be absolute")
    if requested_snapshot != snapshot_path:
        raise ValueError("capture tokenizer snapshot differs from download manifest")
    parts = snapshot_path.parts
    try:
        recorded_revision = parts[parts.index("snapshots") + 1]
    except (ValueError, IndexError) as error:
        raise ValueError("download manifest tokenizer snapshot path is invalid") from error
    if recorded_revision != revision:
        raise ValueError("download manifest tokenizer snapshot revision differs from data config")
    if snapshot_path.is_symlink() or not snapshot_path.is_dir():
        raise ValueError(f"locked tokenizer snapshot is missing or unsafe: {snapshot_path}")
    return TokenizerBinding(tokenizer_id, revision, snapshot_path)


@contextmanager
def _offline_huggingface():
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update({name: "1" for name in names})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _duo_canonical_logits_vocab_size(model: object) -> int | None:
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    vocab_size = getattr(config, "vocab_size", None)
    if (
        isinstance(model_type, str)
        and model_type.lower() == "duo"
        and vocab_size == 50_258
    ):
        return 50_257
    return None


def _duo_canonical_sampler_vocab_size(owner: object) -> int | None:
    size = _duo_canonical_logits_vocab_size(getattr(owner, "backbone", None))
    current = getattr(owner, "vocab_size", None)
    if size is not None and current == size + 1:
        return size
    return None


def _adapt_hf_masked_lm_backbone(model: object) -> object:
    forward = getattr(model, "forward", None)
    if not callable(forward):
        return model
    try:
        parameters = inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return model
    if {"x", "sigma"} <= set(parameters):
        return model
    if not {"input_ids", "timesteps"} <= set(parameters):
        return model
    try:
        import torch
    except ImportError:
        module_base = object
    else:
        module_base = torch.nn.Module

    class KeywordBackboneAdapter(module_base):
        def __init__(self, inner: object) -> None:
            if module_base is not object:
                super().__init__()
            self.inner = inner
            self._canonical_logits_vocab_size = _duo_canonical_logits_vocab_size(inner)

        def _canonical_logits(self, logits):
            size = self._canonical_logits_vocab_size
            shape = getattr(logits, "shape", ())
            if size is not None and shape and shape[-1] == size + 1:
                return logits[..., :size]
            return logits

        def forward(self, *args, **kwargs):
            if "x" in kwargs:
                kwargs["input_ids"] = kwargs.pop("x")
            elif args:
                kwargs["input_ids"] = args[0]
                args = args[1:]
            if "sigma" in kwargs:
                kwargs["timesteps"] = kwargs.pop("sigma")
            elif args:
                kwargs["timesteps"] = args[0]
                args = args[1:]
            kwargs.pop("class_cond", None)
            kwargs.pop("weights", None)
            call = self.inner if callable(self.inner) else self.inner.forward
            output = call(*args, **kwargs)
            if hasattr(output, "logits"):
                return self._canonical_logits(output.logits)
            if isinstance(output, tuple) and output:
                return self._canonical_logits(output[0])
            return self._canonical_logits(output)

        def __getattr__(self, name: str) -> object:
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.inner, name)

    if module_base is object:
        KeywordBackboneAdapter.__call__ = KeywordBackboneAdapter.forward
    return KeywordBackboneAdapter(model)


@contextmanager
def _patched_hf_masked_lm_backbone():
    try:
        transformers = importlib.import_module("transformers")
    except ImportError:
        yield
        return
    auto_model = getattr(transformers, "AutoModelForMaskedLM", None)
    original = getattr(auto_model, "from_pretrained", None)
    if not callable(original):
        yield
        return

    def from_pretrained(*args, **kwargs):
        return _adapt_hf_masked_lm_backbone(original(*args, **kwargs))

    auto_model.from_pretrained = from_pretrained
    try:
        yield
    finally:
        auto_model.from_pretrained = original


def _load_entrypoint(path: Path) -> ModuleType:
    sys.path.insert(0, str(path.parent))
    module_name = "dlb_pinned_upstream_main"
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ValueError(f"cannot load upstream entrypoint: {path}")
    module = importlib.util.module_from_spec(specification)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


def _sample_owner(module: ModuleType) -> type:
    diffusion = getattr(module, "diffusion", None)
    if diffusion is not None and hasattr(diffusion, "Diffusion"):
        return diffusion.Diffusion
    algo = getattr(module, "algo", None)
    trainer_base = getattr(algo, "trainer_base", None)
    owner = getattr(trainer_base, "TrainerBase", None)
    if isinstance(owner, type):
        return owner
    raise ValueError("upstream entrypoint has no supported sampling owner")


def _write_capture(
    path: Path,
    texts: list[object],
    token_rows: list[object],
    expected: int | None,
    extras: list[dict[str, object]] | None = None,
) -> None:
    if len(texts) != len(token_rows):
        raise ValueError("upstream token and text sample counts differ")
    if extras is not None and len(extras) != len(token_rows):
        raise ValueError("upstream conditional metadata and token sample counts differ")
    if expected is not None and len(texts) != expected:
        raise ValueError(f"expected {expected} captured samples, found {len(texts)}")
    samples: list[dict[str, object]] = []
    for index, (text, tokens) in enumerate(zip(texts, token_rows)):
        if not isinstance(text, str):
            raise ValueError(f"captured text {index} is not a string")
        if not isinstance(tokens, list):
            raise ValueError(f"captured token row {index} is not a list")
        payload = {"sample_id": index, "text": text, "token_ids": tokens}
        if extras is not None:
            payload.update(extras[index])
        samples.append(payload)
    atomic_json_write(path, {"schema": "dlb-upstream-token-capture-v1", "samples": samples})


def _rows(value: object, *, label: str = "token rows") -> list[object]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if not isinstance(value, list):
        raise ValueError(f"{label} returned invalid token rows")
    rows: list[object] = []
    for row in value:
        if hasattr(row, "detach"):
            row = row.detach().cpu().tolist()
        elif not isinstance(row, list):
            row = list(row)
        rows.append(row)
    return rows


def _run_main(
    module: ModuleType,
    entrypoint: Path,
    forwarded: list[str],
    *,
    hydra_config_path: bool = True,
) -> None:
    previous_argv = sys.argv
    upstream_arguments = (
        _forwarded_with_hydra_config_path(entrypoint, forwarded)
        if hydra_config_path
        else forwarded
    )
    try:
        sys.argv = [str(entrypoint), *upstream_arguments]
        module.main()
    finally:
        sys.argv = previous_argv


def _forwarded_with_hydra_config_path(entrypoint: Path, forwarded: list[str]) -> list[str]:
    if any(
        argument in {"--config-path", "-cp"}
        or argument.startswith("--config-path=")
        or argument.startswith("-cp=")
        for argument in forwarded
    ):
        return forwarded
    config_dir = entrypoint.parent / "configs"
    if config_dir.is_dir() and not config_dir.is_symlink():
        return [f"--config-path={config_dir.resolve()}", *forwarded]
    return forwarded


def _capture_teacher(
    module: ModuleType, invocation: CaptureInvocation, forwarded: list[str]
) -> None:
    owner = _sample_owner(module)
    original = owner.restore_model_and_sample
    captured: list[dict[str, object]] = []

    def current_conditioning_batch(self):
        if invocation.generation_mode != "conditional_prefix":
            return None
        if (
            invocation.conditioning_manifest is None
            or invocation.conditioning_manifest_sha256 is None
            or invocation.prompt_count is None
            or invocation.diversity_prompt_count is None
            or invocation.completions_per_diversity_prompt is None
        ):
            raise ValueError("conditional capture invocation is incomplete")
        offset = len(captured)
        if offset < invocation.prompt_count:
            completion_id = 0
            prompt_start = offset
            limit = invocation.prompt_count
        else:
            diversity_offset = offset - invocation.prompt_count
            completion_id = diversity_offset // invocation.diversity_prompt_count + 1
            prompt_start = diversity_offset % invocation.diversity_prompt_count
            limit = invocation.diversity_prompt_count
        if completion_id >= invocation.completions_per_diversity_prompt:
            raise ValueError("conditional capture schedule is exhausted")
        batch_size = int(getattr(self.config.loader, "eval_batch_size"))
        current = min(batch_size, limit - prompt_start)
        if current != batch_size:
            raise ValueError("conditional batch size crosses a schedule boundary")
        vocab_size = int(getattr(self, "vocab_size", 0) or len(self.tokenizer))
        return load_conditioning_batch(
            invocation.conditioning_manifest,
            invocation.conditioning_manifest_sha256,
            completion_id=completion_id,
            prompt_start=prompt_start,
            batch_size=batch_size,
            device=str(self.device),
            vocab_size=vocab_size,
        )

    def install_conditioning(self, batch):
        if batch is None:
            return ExitStack()
        family = invocation.entrypoint.parent.name
        stack = ExitStack()
        if family == "mdlm":
            project = token_project_fn(batch.prefix_token_ids)
            if hasattr(self, "_sample_prior"):
                original_prior = self._sample_prior

                def prior(*args, **kwargs):
                    return project(original_prior(*args, **kwargs).to(self.device))

                stack.enter_context(patched_attribute(self, "_sample_prior", prior))
            for name in ("_ddpm_update", "_ddpm_caching_update", "_analytic_update", "_denoiser_update"):
                original_update = getattr(self, name, None)
                if not callable(original_update):
                    continue

                def update(*args, __original=original_update, **kwargs):
                    if "x" in kwargs:
                        kwargs["x"] = project(kwargs["x"])
                    elif args:
                        args = (project(args[0]), *args[1:])
                    output = __original(*args, **kwargs)
                    if isinstance(output, tuple):
                        return (*output[:-1], project(output[-1]))
                    return project(output)

                stack.enter_context(patched_attribute(self, name, update))
        elif family == "duo":
            sampler_vocab_size = _duo_canonical_sampler_vocab_size(self)
            if sampler_vocab_size is not None:
                stack.enter_context(patched_attribute(self, "vocab_size", sampler_vocab_size))
            project = token_project_fn(batch.prefix_token_ids)
            if hasattr(self, "prior_sample"):
                original_prior = self.prior_sample

                def prior(*args, **kwargs):
                    return project(original_prior(*args, **kwargs).to(self.device))

                stack.enter_context(patched_attribute(self, "prior_sample", prior))
            for name in ("_ancestral_update", "_analytic_update", "_denoiser_update"):
                original_update = getattr(self, name, None)
                if not callable(original_update):
                    continue

                def update(*args, __original=original_update, **kwargs):
                    if "x" in kwargs:
                        kwargs["x"] = project(kwargs["x"])
                    elif args:
                        args = (project(args[0]), *args[1:])
                    output = __original(*args, **kwargs)
                    if isinstance(output, tuple):
                        return (*output[:-1], project(output[-1]))
                    return project(output)

                stack.enter_context(patched_attribute(self, name, update))
        elif family == "flm":
            project = vocab_project_fn(batch.prefix_token_ids)
            if hasattr(self, "prior_sample"):
                original_prior = self.prior_sample

                def prior(*args, **kwargs):
                    return project(original_prior(*args, **kwargs).to(self.device))

                stack.enter_context(patched_attribute(self, "prior_sample", prior))
            original_forward = self.forward

            def forward(*args, **kwargs):
                if "xt" in kwargs:
                    kwargs["xt"] = project(kwargs["xt"])
                elif args:
                    args = (project(args[0]), *args[1:])
                return original_forward(*args, **kwargs)

            stack.enter_context(patched_attribute(self, "forward", forward))
        elif family == "candi":
            original_generate = self.generate_samples
            original_generate_sample_prompt = self.generate_sample_prompt
            stack.enter_context(
                patched_attribute(
                    self,
                    "generate_sample_prompt",
                    adapt_candi_generate_sample_prompt(original_generate_sample_prompt),
                )
            )

            def generate(*args, **kwargs):
                prompt_tokens, prompt_mask = candi_prompt_conditioning(
                    batch.prefix_token_ids,
                    sequence_length=int(getattr(self, "num_tokens")),
                )
                kwargs["prompt_tokens"] = prompt_tokens
                kwargs["prompt_mask"] = prompt_mask
                return original_generate(*args, **kwargs)

            stack.enter_context(patched_attribute(self, "generate_samples", generate))
        else:
            raise ValueError(f"unsupported conditional teacher family: {family}")
        return stack

    def capture(self, *args, **kwargs):
        if invocation.benchmark_output is not None:
            num_steps = kwargs.get("num_steps", args[0] if args else None)
            eps = kwargs.get("eps", 1e-5)
            if type(num_steps) is not int or num_steps <= 0:
                raise ValueError("benchmark sampler requires a positive num_steps")
            if invocation.entrypoint.parent.name == "mdlm":
                if self.ema:
                    parameters = tuple(
                        list(self.backbone.parameters()) + list(self.noise.parameters())
                    )
                    self.ema.store(parameters)
                    self.ema.copy_to(parameters)
                self.backbone.eval()
                self.noise.eval()
                generate_one = lambda: self._sample(num_steps=num_steps, eps=eps)
            else:
                self._eval_mode()
                generate_one = lambda: self.generate_samples(
                    num_samples=1, num_steps=num_steps, eps=eps
                )
            try:
                return benchmark_and_publish(
                    generate_one,
                    model=self,
                    output=invocation.benchmark_output,
                    metadata_path=invocation.benchmark_metadata,
                    precision=invocation.benchmark_precision,
                )
            finally:
                if invocation.entrypoint.parent.name == "mdlm":
                    if self.ema:
                        self.ema.restore(parameters)
                    self.backbone.train()
                    self.noise.train()
                else:
                    self._train_mode()
        batch = current_conditioning_batch(self)
        with install_conditioning(self, batch):
            result = original(self, *args, **kwargs)
        if batch is not None:
            if isinstance(result, list) and result and all(hasattr(row, "shape") for row in result):
                import torch

                result = torch.stack(result)
            result = clamp_token_prefix(result, batch.prefix_token_ids)
        token_rows = _rows(result, label="teacher sampler")
        text_rows = list(self.tokenizer.batch_decode(result))
        if len(token_rows) != len(text_rows):
            raise ValueError("upstream token and text batch sizes differ")
        for batch_index, (tokens, text) in enumerate(zip(token_rows, text_rows)):
            payload = {"sample_id": len(captured), "text": text, "token_ids": tokens}
            if batch is not None:
                prefix = batch.prefix_token_ids[batch_index].detach().cpu().tolist()
                reference = batch.reference_token_ids[batch_index].detach().cpu().tolist()
                if tokens[: len(prefix)] != prefix:
                    raise ValueError("conditional sampler returned a prefix mismatch")
                payload.update(
                    {
                        "prompt_id": batch.prompt_ids[batch_index],
                        "completion_id": batch.completion_id,
                        "source_index": batch.source_indices[batch_index],
                        "prefix_token_ids": prefix,
                        "reference_token_ids": reference,
                        "full_token_ids": tokens,
                        "prefix_text": self.tokenizer.decode(prefix),
                        "continuation_text": self.tokenizer.decode(tokens[len(prefix) :]),
                        "reference_text": self.tokenizer.decode(reference),
                        "full_text": text,
                        "prefix_exact_match": True,
                    }
                )
            captured.append(payload)
        return result

    owner.restore_model_and_sample = capture
    with _patched_hf_masked_lm_backbone():
        _run_main(module, invocation.entrypoint, forwarded)
    atomic_json_write(
        invocation.capture_path,
        {"schema": "dlb-upstream-token-capture-v1", "samples": captured},
    )


def _capture_langflow(
    module: ModuleType,
    invocation: CaptureInvocation,
    forwarded: list[str],
    tokenizer: TokenizerBinding,
) -> None:
    owner = getattr(module, "LangFlow", None)
    auto_tokenizer = getattr(module, "AutoTokenizer", None)
    if not isinstance(owner, type) or auto_tokenizer is None:
        raise ValueError("LangFlow entrypoint lacks its model or tokenizer")
    original_generate = owner.generate_samples
    token_rows: list[object] = []
    texts: list[object] = []
    conditional_extras: list[dict[str, object]] = []

    def generate(self, *args, **kwargs):
        conditioning = None
        project_embeddings = None
        original_forward = None
        if invocation.generation_mode == "conditional_prefix":
            if invocation.conditioning_manifest is None or invocation.conditioning_manifest_sha256 is None:
                raise ValueError("LangFlow conditional capture requires a prompt manifest")
            batch_size = int(kwargs.get("num_samples", 1))
            conditioning = load_conditioning_batch(
                invocation.conditioning_manifest,
                invocation.conditioning_manifest_sha256,
                completion_id=0 if len(token_rows) < (invocation.prompt_count or 0) else (len(token_rows) - (invocation.prompt_count or 0)) // (invocation.diversity_prompt_count or 1) + 1,
                prompt_start=len(token_rows) if len(token_rows) < (invocation.prompt_count or 0) else (len(token_rows) - (invocation.prompt_count or 0)) % (invocation.diversity_prompt_count or 1),
                batch_size=batch_size,
                device=str(next(self.parameters()).device),
                vocab_size=int(self.config.vocab_size),
            )
            import torch

            clean_one_hot = torch.nn.functional.one_hot(
                conditioning.prefix_token_ids, num_classes=int(self.config.vocab_size)
            ).to(dtype=next(self.parameters()).dtype, device=conditioning.prefix_token_ids.device)
            clean_embeddings = self._embed_tokens(clean_one_hot)
            project_embeddings = embedding_project_fn(clean_embeddings)
            original_forward = self.forward

            def forward(*args, **kwargs):
                if "noisy_embeds" in kwargs and kwargs["noisy_embeds"] is not None:
                    kwargs["noisy_embeds"] = project_embeddings(kwargs["noisy_embeds"])
                if "x_self_cond" in kwargs and kwargs["x_self_cond"] is not None:
                    kwargs["x_self_cond"] = project_embeddings(kwargs["x_self_cond"])
                return original_forward(*args, **kwargs)

        with (
            patched_attribute(self, "forward", forward)
            if project_embeddings is not None and original_forward is not None
            else nullcontext()
        ):
            if invocation.benchmark_output is None:
                result = original_generate(self, *args, **kwargs)
            else:
                benchmark_kwargs = dict(kwargs)
                benchmark_kwargs["num_samples"] = 1
                result = benchmark_and_publish(
                    lambda: original_generate(self, *args, **benchmark_kwargs),
                    model=self,
                    output=invocation.benchmark_output,
                    metadata_path=invocation.benchmark_metadata,
                    precision=invocation.benchmark_precision,
                )
        if conditioning is not None:
            result = clamp_token_prefix(result, conditioning.prefix_token_ids)
        rows = result.detach().cpu().tolist()
        if not isinstance(rows, list):
            raise ValueError("LangFlow sampler returned invalid token rows")
        if conditioning is not None:
            for batch_index, row in enumerate(rows):
                prefix = conditioning.prefix_token_ids[batch_index].detach().cpu().tolist()
                reference = conditioning.reference_token_ids[batch_index].detach().cpu().tolist()
                if row[: len(prefix)] != prefix:
                    raise ValueError("LangFlow conditional sampler returned a prefix mismatch")
                conditional_extras.append(
                    {
                        "prompt_id": conditioning.prompt_ids[batch_index],
                        "completion_id": conditioning.completion_id,
                        "source_index": conditioning.source_indices[batch_index],
                        "prefix_token_ids": prefix,
                        "reference_token_ids": reference,
                        "full_token_ids": row,
                        "prefix_exact_match": True,
                    }
                )
        token_rows.extend(rows)
        return result

    class TokenizerProxy:
        def __init__(self, tokenizer: object) -> None:
            self._tokenizer = tokenizer

        def __getattr__(self, name: str) -> object:
            return getattr(self._tokenizer, name)

        def batch_decode(self, *args, **kwargs):
            decoded = list(self._tokenizer.batch_decode(*args, **kwargs))
            texts[:] = decoded
            return decoded

    class AutoTokenizerProxy:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            del args
            kwargs.pop("revision", None)
            kwargs["local_files_only"] = True
            loaded = auto_tokenizer.from_pretrained(str(tokenizer.snapshot), **kwargs)
            return TokenizerProxy(loaded)

    owner.generate_samples = generate
    module.AutoTokenizer = AutoTokenizerProxy
    with _offline_huggingface():
        parameters = inspect.signature(_run_main).parameters
        if "hydra_config_path" in parameters:
            _run_main(module, invocation.entrypoint, forwarded, hydra_config_path=False)
        else:
            _run_main(module, invocation.entrypoint, forwarded)
    _write_capture(
        invocation.capture_path,
        texts,
        token_rows,
        invocation.expected_samples,
        conditional_extras if invocation.generation_mode == "conditional_prefix" else None,
    )


def _model_path(forwarded: list[str]) -> Path:
    prefix = "model_path="
    values = [argument[len(prefix) :] for argument in forwarded if argument.startswith(prefix)]
    if len(values) != 1 or not values[0]:
        raise ValueError("RDLM capture requires one model_path override")
    path = Path(values[0])
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"RDLM checkpoint is missing or unsafe: {path}")
    return path


def _positive_override(forwarded: list[str], key: str) -> int:
    prefix = key + "="
    values = [argument[len(prefix) :] for argument in forwarded if argument.startswith(prefix)]
    if len(values) != 1:
        raise ValueError(f"RDLM capture requires one {key} override")
    try:
        value = int(values[0])
    except ValueError as error:
        raise ValueError(f"RDLM {key} must be an integer") from error
    if value <= 0:
        raise ValueError(f"RDLM {key} must be positive")
    return value


def _require_saved_file(path: Path | None, label: str) -> Path:
    if (
        path is None
        or not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size <= 0
    ):
        raise ValueError(f"saved RDLM {label} is missing or unsafe: {path}")
    return path


def _rdlm_benchmark_sampling_fn(
    sampling_fn: Callable[[object], object], invocation: CaptureInvocation
) -> Callable[[object], object]:
    """Publish one benchmark the first time RDLM reaches its loaded sampling_fn."""

    if invocation.benchmark_output is None:
        return sampling_fn
    published = False

    def sample(model: object) -> object:
        nonlocal published
        if published:
            return sampling_fn(model)
        published = True
        return benchmark_and_publish(
            lambda: sampling_fn(model),
            model=model,
            output=invocation.benchmark_output,
            metadata_path=invocation.benchmark_metadata,
            precision=invocation.benchmark_precision,
        )

    return sample


def _capture_rdlm(
    module: ModuleType, invocation: CaptureInvocation, forwarded: list[str]
) -> None:
    from omegaconf import OmegaConf

    tokenizer = _load_tokenizer_binding(invocation)
    saved_config_path = _require_saved_file(invocation.saved_config_path, "config")
    saved_sde_path = _require_saved_file(invocation.saved_sde_path, "SDE")
    checkpoint_path = _model_path(forwarded)
    expected = invocation.expected_samples
    if type(expected) is not int or expected <= 0:
        raise ValueError("RDLM capture requires a positive expected sample count")
    batch_size = _positive_override(forwarded, "sampling.batch_per_gpu")
    sample_batches = math.ceil(expected / batch_size)
    saved_config = OmegaConf.load(saved_config_path)
    if OmegaConf.select(saved_config, "model.length") != 128:
        raise ValueError("saved RDLM config does not declare LM1B model length 128")
    if OmegaConf.select(saved_config, "data.train") != "lm1b":
        raise ValueError("saved RDLM config is not the LM1B training config")

    run_sample = importlib.import_module("run_sample")
    data_module = getattr(run_sample, "data", None)
    transformers_module = getattr(data_module, "transformers", None)
    original_bert_tokenizer = getattr(transformers_module, "BertTokenizer", None)
    original_auto_tokenizer = getattr(data_module, "AutoTokenizer", None)
    if original_bert_tokenizer is None or original_auto_tokenizer is None:
        raise ValueError("RDLM data module lacks its tokenizer constructors")
    original_torch = run_sample.torch
    original_open: Callable[..., object] = builtins.open
    original_tqdm = run_sample.tqdm
    original_instantiate = run_sample.instantiate
    original_shift_factory = run_sample.sutils.find_bos_and_shift_fn
    original_sampling_factory = (
        run_sample.sampling.get_sampling_fn
        if invocation.benchmark_output is not None
        or invocation.generation_mode == "conditional_prefix"
        else None
    )
    texts: list[object] = []
    token_rows: list[object] = []
    conditional_extras: list[dict[str, object]] = []
    saved_config_used = False
    saved_sde_opened = False
    saved_sde_used = False

    def locked_tokenizer_proxy(original):
        class LocalTokenizerProxy:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                del args
                kwargs.pop("revision", None)
                kwargs["local_files_only"] = True
                return original.from_pretrained(str(tokenizer.snapshot), **kwargs)

        return LocalTokenizerProxy

    class TorchProxy:
        def __getattr__(self, name: str) -> object:
            return getattr(original_torch, name)

        def load(self, path, *args, **kwargs):
            nonlocal saved_config_used
            state = original_torch.load(path, *args, **kwargs)
            if Path(path).resolve() != checkpoint_path.resolve():
                return state
            if not isinstance(state, dict):
                raise ValueError("RDLM checkpoint state is not a mapping")
            saved_config_used = True
            return {**state, "config": saved_config}

    def routed_open(path, mode="r", *args, **kwargs):
        nonlocal saved_sde_opened
        requested = Path(path)
        if "r" in mode and "b" in mode and requested.name == "sde.pkl":
            saved_sde_opened = True
            return original_open(saved_sde_path, mode, *args, **kwargs)
        return original_open(path, mode, *args, **kwargs)

    def instantiate(config, *args, **kwargs):
        nonlocal saved_sde_used
        is_sde = {"manifold", "scheduler", "prior_dist"}.issubset(kwargs)
        if is_sde and "preprocessed" not in kwargs:
            raise ValueError("saved SDE preprocessing was not used by RDLM")
        if is_sde:
            saved_sde_used = True
        return original_instantiate(config, *args, **kwargs)

    def exact_sample_batches(iterable, *args, **kwargs):
        del iterable
        return original_tqdm(range(sample_batches), *args, **kwargs)

    def rdlm_conditioning_batch(batch_index: int, device: str):
        if invocation.generation_mode != "conditional_prefix":
            return None
        if (
            invocation.conditioning_manifest is None
            or invocation.conditioning_manifest_sha256 is None
            or invocation.prompt_count is None
            or invocation.diversity_prompt_count is None
            or invocation.completions_per_diversity_prompt is None
        ):
            raise ValueError("RDLM conditional capture invocation is incomplete")
        offset = batch_index * batch_size
        if offset < invocation.prompt_count:
            completion_id = 0
            prompt_start = offset
            limit = invocation.prompt_count
        else:
            diversity_offset = offset - invocation.prompt_count
            completion_id = diversity_offset // invocation.diversity_prompt_count + 1
            prompt_start = diversity_offset % invocation.diversity_prompt_count
            limit = invocation.diversity_prompt_count
        if completion_id >= invocation.completions_per_diversity_prompt:
            raise ValueError("RDLM conditional capture schedule is exhausted")
        if min(batch_size, limit - prompt_start) != batch_size:
            raise ValueError("RDLM conditional batch size crosses a schedule boundary")
        original_tokens = OmegaConf.select(saved_config, "original_tokens")
        if type(original_tokens) is not int or original_tokens <= 0:
            raise ValueError("saved RDLM config does not declare original_tokens")
        return load_conditioning_batch(
            invocation.conditioning_manifest,
            invocation.conditioning_manifest_sha256,
            completion_id=completion_id,
            prompt_start=prompt_start,
            batch_size=batch_size,
            device=device,
            vocab_size=original_tokens,
        )

    def sampling_factory(*args, **kwargs):
        if original_sampling_factory is None:
            raise ValueError("RDLM benchmark sampling factory is unavailable")
        device = kwargs.get("device", args[4] if len(args) > 4 else "cpu")
        active_project = None
        batch_index = 0
        if invocation.generation_mode == "conditional_prefix":
            kwargs = dict(kwargs)

            def project(state):
                return state if active_project is None else active_project(state)

            kwargs["proj_fn"] = project
        sampling_fn = original_sampling_factory(*args, **kwargs)
        sampling_fn = _rdlm_benchmark_sampling_fn(sampling_fn, invocation)

        def sample(model: object) -> object:
            nonlocal active_project, batch_index
            batch = rdlm_conditioning_batch(batch_index, str(device))
            if batch is not None:
                token_size = OmegaConf.select(saved_config, "tokens")
                original_tokens = OmegaConf.select(saved_config, "original_tokens")
                if (
                    type(token_size) is not int
                    or token_size <= 1
                    or type(original_tokens) is not int
                    or original_tokens <= 0
                ):
                    raise ValueError("saved RDLM config has invalid base token contract")
                digits_per_token = math.ceil(math.log(original_tokens) / math.log(token_size))
                active_project = rdlm_project_fn(
                    batch.prefix_token_ids,
                    base=token_size,
                    digits_per_token=digits_per_token,
                )
                for row_index in range(batch.prefix_token_ids.shape[0]):
                    conditional_extras.append(
                        {
                            "prompt_id": batch.prompt_ids[row_index],
                            "completion_id": batch.completion_id,
                            "source_index": batch.source_indices[row_index],
                            "prefix_token_ids": batch.prefix_token_ids[row_index]
                            .detach()
                            .cpu()
                            .tolist(),
                            "reference_token_ids": batch.reference_token_ids[row_index]
                            .detach()
                            .cpu()
                            .tolist(),
                        }
                    )
                batch_index += 1
            try:
                return sampling_fn(model)
            finally:
                active_project = None

        return sample

    def capture_shift_factory(*args, **kwargs):
        shift = original_shift_factory(*args, **kwargs)
        tokenizer_object = args[2] if len(args) >= 3 else kwargs.get("tokenizer")

        def capture_shift(samples):
            sentences, shifted = shift(samples)
            texts[:] = list(sentences)
            token_rows[:] = _rows(shifted)
            if invocation.generation_mode == "conditional_prefix":
                if len(conditional_extras) < len(token_rows):
                    raise ValueError("RDLM conditional metadata is shorter than generated samples")
                for index, row in enumerate(token_rows):
                    prefix = conditional_extras[index]["prefix_token_ids"]
                    reference = conditional_extras[index]["reference_token_ids"]
                    if not isinstance(prefix, list) or row[: len(prefix)] != prefix:
                        raise ValueError("RDLM conditional sampler returned a prefix mismatch")
                    conditional_extras[index].update(
                        {
                            "full_token_ids": row,
                            "prefix_exact_match": True,
                            "prefix_text": (
                                tokenizer_object.decode(prefix)
                                if tokenizer_object is not None
                                else " ".join(map(str, prefix))
                            ),
                            "continuation_text": (
                                tokenizer_object.decode(row[len(prefix) :])
                                if tokenizer_object is not None
                                else " ".join(map(str, row[len(prefix) :]))
                            ),
                            "reference_text": (
                                tokenizer_object.decode(reference)
                                if tokenizer_object is not None
                                else " ".join(map(str, reference))
                            ),
                            "full_text": texts[index],
                        }
                    )
            return sentences, shifted

        return capture_shift

    def inline_spawn(function, args=(), nprocs=1, join=True, **kwargs):
        del join, kwargs
        if nprocs != 1:
            raise ValueError("RDLM capture requires ngpus=1")
        return function(0, *args)

    run_sample.torch = TorchProxy()
    data_module.transformers.BertTokenizer = locked_tokenizer_proxy(
        original_bert_tokenizer
    )
    data_module.AutoTokenizer = locked_tokenizer_proxy(original_auto_tokenizer)
    run_sample.open = routed_open
    run_sample.tqdm = exact_sample_batches
    if original_sampling_factory is not None:
        run_sample.sampling.get_sampling_fn = sampling_factory
    run_sample.instantiate = instantiate
    run_sample.sutils.find_bos_and_shift_fn = capture_shift_factory
    module.mp.spawn = inline_spawn
    with _offline_huggingface():
        _run_main(module, invocation.entrypoint, forwarded)
    if not saved_config_used:
        raise ValueError("saved RDLM config was not used")
    if not saved_sde_opened or not saved_sde_used:
        raise ValueError("saved RDLM SDE was not used")
    if len(texts) < expected or len(token_rows) < expected:
        raise ValueError(
            f"expected {expected} captured samples, found {min(len(texts), len(token_rows))}"
        )
    _write_capture(
        invocation.capture_path,
        texts[:expected],
        token_rows[:expected],
        expected,
        conditional_extras[:expected]
        if invocation.generation_mode == "conditional_prefix"
        else None,
    )


def main(argv: list[str] | None = None) -> int:
    invocation, forwarded = _split_arguments(
        list(sys.argv[1:] if argv is None else argv)
    )
    module = _load_entrypoint(invocation.entrypoint)
    if invocation.kind == "langflow":
        tokenizer = _load_tokenizer_binding(invocation)
        _capture_langflow(module, invocation, forwarded, tokenizer)
    elif invocation.kind == "rdlm":
        _capture_rdlm(module, invocation, forwarded)
    else:
        _capture_teacher(module, invocation, forwarded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

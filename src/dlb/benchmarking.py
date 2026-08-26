"""Render and execute registry-bound, concrete sampler timing commands."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Iterable, Literal
import uuid

from dlb.adapters.base import AdapterError, BaseTeacherAdapter
from dlb.checkpoints import load_checkpoint_manifest
from dlb.command import ADAPTERS
from dlb.io import atomic_json_write, sha256_file
from dlb.registry import load_registry, step_grid_for_model
from dlb.runner import RunRequest, _resolve_request
from dlb.timing import _validate_metadata


def _run_dir(root: Path, request: RunRequest) -> Path:
    result_root = (
        root / "results/conditional"
        if request.generation_mode == "conditional_prefix"
        else root / "results"
    )
    return (
        result_root
        / "samples"
        / request.dataset_id
        / request.model_id
        / f"steps_{request.step_count}"
    )


def _output_path(output_root: Path, request: RunRequest) -> Path:
    return (
        output_root
        / request.dataset_id
        / request.model_id
        / f"steps_{request.step_count}"
        / "timing.json"
    )


def _request(
    model: str,
    dataset: str,
    steps: int,
    seed: int,
    generation_mode: Literal["unconditional", "conditional_prefix"] = "unconditional",
) -> RunRequest:
    return RunRequest(
        run_id=f"benchmark-{model}-{dataset}-steps-{steps}",
        model_id=model,
        dataset_id=dataset,
        step_count=steps,
        seed=seed,
        sample_count=2048 if generation_mode == "conditional_prefix" else 1,
        generation_mode=generation_mode,
    )


def render_benchmark_matrix(
    *,
    root: Path,
    models: Iterable[str] | None,
    datasets: Iterable[str],
    steps: int,
    seed: int,
    precision: str,
    output_root: Path,
    dry_run: bool,
    generation_mode: Literal["unconditional", "conditional_prefix"] = "unconditional",
) -> list[dict[str, object]]:
    """Render every selected cell, preserving explicit unsupported records."""

    root = root.resolve()
    registry = load_registry(root / "configs/experiments.yaml")
    selected_models = tuple(registry.models) if models is None else tuple(models)
    records: list[dict[str, object]] = []
    for model_id in selected_models:
        for dataset_id in datasets:
            model = registry.models.get(model_id)
            support = model.datasets.get(dataset_id) if model is not None else None
            if model is None or support is None or support.status != "supported":
                records.append(
                    {
                        "status": "unsupported",
                        "model": model_id,
                        "dataset": dataset_id,
                        "reason": getattr(support, "reason", None) or "unknown model/dataset cell",
                    }
                )
                continue
            allowed_steps = step_grid_for_model(registry, model_id)
            if steps not in allowed_steps:
                joined_steps = ",".join(str(step) for step in allowed_steps)
                records.append(
                    {
                        "status": "error",
                        "model": model_id,
                        "dataset": dataset_id,
                        "reason": (
                            f"invalid step count {steps} for {model_id}/{dataset_id}; "
                            f"allowed: {joined_steps}"
                        ),
                    }
                )
                continue
            adapter = ADAPTERS[model_id]
            request = _request(model_id, dataset_id, steps, seed, generation_mode)
            if generation_mode == "conditional_prefix":
                try:
                    request, _ = _resolve_request(request, root, adapter)
                except (OSError, ValueError) as error:
                    records.append(
                        {
                            "status": "error",
                            "model": model_id,
                            "dataset": dataset_id,
                            "reason": str(error),
                        }
                    )
                    continue
            output = _output_path(output_root.resolve(), request)
            dry_attempt = "0" * 32
            staged_output = output.with_name(
                f".{output.name}.{dry_attempt}.staged.json"
            )
            metadata_path = output.with_name(
                f".benchmark_metadata.{dry_attempt}.json"
            )
            try:
                command = adapter.render_benchmark_command(
                    request,
                    _run_dir(root, request),
                    output=staged_output,
                    metadata_path=metadata_path,
                    precision=precision,
                    dry_run=dry_run,
                )
            except (AdapterError, OSError, ValueError) as error:
                records.append(
                    {
                        "status": "error",
                        "model": model_id,
                        "dataset": dataset_id,
                        "reason": str(error),
                    }
                )
                continue
            records.append(
                {
                    "status": "supported",
                    "model": model_id,
                    "dataset": dataset_id,
                    "steps": steps,
                    "environment": model.environment,
                    "hook": adapter.benchmark_hook(request),
                    **adapter.author_precision_policy(request),
                    "batch_size": 1,
                    "warmups": 5,
                    "repeats": 32,
                    "output": str(output),
                    "staged_output": str(staged_output),
                    "metadata_path": str(metadata_path),
                    "generation_mode": generation_mode,
                    "command": command,
                }
            )
    return records


def _precision_policy_binding(
    root: Path, request: RunRequest, adapter: BaseTeacherAdapter
) -> dict[str, object]:
    """Bind an audited static policy to exact run, checkpoint and config bytes."""

    binding: dict[str, object] = {
        "model": request.model_id,
        "dataset": request.dataset_id,
        "upstream": adapter.upstream,
        "source_commit": request.source_sha256,
        "experiment_config_sha256": request.config_sha256,
        "checkpoint_sha256": request.checkpoint_sha256,
        "checkpoint_lock_id": request.checkpoint_lock_id,
        "checkpoint_selection": request.checkpoint_selection,
        "adapter_identity": request.adapter_identity,
    }
    if request.generation_mode == "conditional_prefix":
        binding.update(
            {
                "generation_mode": request.generation_mode,
                "conditioning_manifest": request.conditioning_manifest,
                "conditioning_manifest_sha256": request.conditioning_manifest_sha256,
                "conditioning_config_sha256": request.conditioning_config_sha256,
                "prefix_length": request.prefix_length,
                "evaluation_continuation_length": request.evaluation_continuation_length,
                "prompt_count": request.prompt_count,
                "diversity_prompt_count": request.diversity_prompt_count,
                "completions_per_diversity_prompt": request.completions_per_diversity_prompt,
                "completion_schedule": request.completion_schedule,
            }
        )
    selection = request.checkpoint_selection
    resource_id = selection.get("resource") if isinstance(selection, dict) else None
    if isinstance(resource_id, str):
        manifest = load_checkpoint_manifest(root / "artifacts/checkpoints.yaml")
        resource = manifest.resources.get(resource_id)
        if resource is None:
            raise ValueError(f"precision policy references unknown checkpoint {resource_id}")
        binding.update(
            {
                "checkpoint_resource": resource_id,
                "checkpoint_repository": getattr(resource.source, "repo_id", None),
                "checkpoint_revision": getattr(resource.source, "revision", None),
            }
        )
        if request.model_id in {"flm", "fmlm"}:
            config_path = root / "checkpoints" / resource.destination / "config.json"
            if config_path.is_symlink() or not config_path.is_file():
                raise ValueError(
                    "FLM/FMLM precision policy requires the downloaded checkpoint config.json"
                )
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("FLM/FMLM checkpoint config.json is invalid") from error
            if not isinstance(config, dict):
                raise ValueError("FLM/FMLM checkpoint config.json must be a mapping")
            dtype = config.get("torch_dtype")
            if not isinstance(dtype, str) or not dtype:
                raise ValueError(
                    "FLM/FMLM checkpoint config does not declare torch_dtype"
                )
            binding.update(
                {
                    "checkpoint_config_path": str(config_path.resolve()),
                    "checkpoint_config_sha256": sha256_file(config_path),
                    "checkpoint_config_torch_dtype": dtype,
                    "checkpoint_config_architectures": config.get("architectures"),
                    "checkpoint_config_auto_map": config.get("auto_map"),
                }
            )
    return binding


def _metadata(
    root: Path, request: RunRequest, adapter: BaseTeacherAdapter, attempt_id: str
) -> dict[str, object]:
    binding = _precision_policy_binding(root, request, adapter)
    policy: dict[str, object] = adapter.author_precision_policy(request)
    if policy.get("precision") == "resolved-from-checkpoint-config-at-execution":
        policy["precision"] = "bf16-mixed-static-author-policy"
    return {
        "seed": request.seed,
        "sample_count": request.sample_count,
        "dataset": request.dataset_id,
        "model": request.model_id,
        "steps": request.step_count,
        "environment": request.environment,
        "source_commit": request.source_sha256,
        "config_sha256": request.config_sha256,
        "checkpoint_sha256": request.checkpoint_sha256,
        "checkpoint_lock_id": request.checkpoint_lock_id,
        "checkpoint_selection": request.checkpoint_selection,
        "checkpoint_teacher_family": request.checkpoint_teacher_family,
        "adapter_identity": request.adapter_identity,
        "requested_precision": "author",
        "attempt_id": attempt_id,
        "precision_policy_binding": binding,
        **policy,
        **(
            {
                "generation_mode": request.generation_mode,
                "conditioning_manifest": request.conditioning_manifest,
                "conditioning_manifest_sha256": request.conditioning_manifest_sha256,
                "conditioning_config_sha256": request.conditioning_config_sha256,
                "prefix_length": request.prefix_length,
                "evaluation_continuation_length": request.evaluation_continuation_length,
                "prompt_count": request.prompt_count,
                "diversity_prompt_count": request.diversity_prompt_count,
                "completions_per_diversity_prompt": request.completions_per_diversity_prompt,
                "completion_schedule": request.completion_schedule,
            }
            if request.generation_mode == "conditional_prefix"
            else {}
        ),
    }


def _load_staged_timing(
    path: Path, expected_metadata: dict[str, object], attempt_id: str
) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("benchmark sampler completed without a fresh staged timing result")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        timing = payload["timing"]
        metadata = payload["metadata"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError("fresh staged timing result is invalid") from error
    if payload.get("schema") != "dlb-generation-timing-v1":
        raise RuntimeError("fresh staged timing result has the wrong schema")
    try:
        _validate_metadata(metadata)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"fresh staged timing metadata is invalid: {error}") from error
    if metadata.get("attempt_id") != attempt_id:
        raise RuntimeError("fresh staged timing attempt provenance does not match")
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise RuntimeError(f"fresh staged timing provenance differs at {key}")
    if not isinstance(timing, dict) or any(
        timing.get(key) != expected
        for key, expected in {
            "mode": "primary_latency",
            "warmups": 5,
            "repeats": 32,
            "batch_size": 1,
            "num_timed_samples": 32,
        }.items()
    ):
        raise RuntimeError("fresh staged timing protocol is invalid")
    raw = timing.get("raw_durations_seconds")
    seconds = timing.get("seconds_per_sample")
    if (
        not isinstance(raw, list)
        or len(raw) != 32
        or any(type(value) not in {int, float} or not math.isfinite(value) or value < 0 for value in raw)
        or type(seconds) not in {int, float}
        or not math.isfinite(seconds)
        or seconds < 0
    ):
        raise RuntimeError("fresh staged timing durations are invalid")


def run_timing_attempt(
    command: list[str],
    *,
    cwd: Path,
    final_output: Path,
    staged_output: Path,
    expected_metadata: dict[str, object],
    attempt_id: str,
) -> Path:
    """Publish a staged result only after the complete sampler process succeeds."""

    if re.fullmatch(r"[0-9a-f]{32}", attempt_id) is None:
        raise ValueError("attempt_id must be 32 lowercase hexadecimal characters")
    final_output = final_output.absolute()
    staged_output = staged_output.absolute()
    expected_stage = final_output.with_name(
        f".{final_output.name}.{attempt_id}.staged.json"
    )
    if staged_output != expected_stage:
        raise ValueError("staged timing path does not match this attempt")
    final_output.parent.mkdir(parents=True, exist_ok=True)
    if staged_output.exists() or staged_output.is_symlink():
        raise RuntimeError("staged timing attempt path already exists")
    superseded = final_output.with_name(
        f".{final_output.name}.{attempt_id}.superseded.json"
    )
    if superseded.exists() or superseded.is_symlink():
        raise RuntimeError("superseded timing path already exists")
    if final_output.exists() or final_output.is_symlink():
        if final_output.is_symlink() or not final_output.is_file():
            raise RuntimeError("existing timing result is unsafe")
        os.replace(final_output, superseded)
    try:
        completed = subprocess.run(command, cwd=cwd, check=False)
        if completed.returncode:
            raise RuntimeError(
                f"benchmark sampler exited with status {completed.returncode}"
            )
        _load_staged_timing(staged_output, expected_metadata, attempt_id)
        os.replace(staged_output, final_output)
    except BaseException:
        staged_output.unlink(missing_ok=True)
        raise
    return final_output


def execute_one(
    *,
    root: Path,
    model: str,
    dataset: str,
    steps: int,
    seed: int,
    precision: str,
    generation_mode: Literal["unconditional", "conditional_prefix"] = "unconditional",
) -> Path:
    """Resolve provenance, then execute one real sampler wrapper in this environment."""

    root = root.resolve()
    request = _request(model, dataset, steps, seed, generation_mode)
    adapter: BaseTeacherAdapter = ADAPTERS[model]
    resolved, adapter = _resolve_request(request, root, adapter)
    timing_root = (
        root / "results/conditional/timing"
        if generation_mode == "conditional_prefix"
        else root / "results/timing"
    )
    output = _output_path(timing_root, resolved)
    attempt_id = uuid.uuid4().hex
    staged_output = output.with_name(f".{output.name}.{attempt_id}.staged.json")
    metadata_path = output.with_name(f".benchmark_metadata.{attempt_id}.json")
    metadata = _metadata(root, resolved, adapter, attempt_id)
    atomic_json_write(metadata_path, metadata)
    try:
        command = adapter.render_benchmark_command(
            resolved,
            _run_dir(root, resolved),
            output=staged_output,
            metadata_path=metadata_path,
            precision=precision,
            dry_run=False,
        )
        return run_timing_attempt(
            command,
            cwd=root,
            final_output=output,
            staged_output=staged_output,
            expected_metadata=metadata,
            attempt_id=attempt_id,
        )
    finally:
        metadata_path.unlink(missing_ok=True)


def _csv(value: str) -> tuple[str, ...]:
    items = tuple(value.split(","))
    if not items or any(not item for item in items):
        raise argparse.ArgumentTypeError("comma-separated values must not be empty")
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", type=_csv, required=True)
    parser.add_argument("--datasets", type=_csv, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--precision", choices=("author",), required=True)
    parser.add_argument(
        "--generation-mode",
        choices=("unconditional", "conditional_prefix"),
        default="unconditional",
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.dry_run:
        records = render_benchmark_matrix(
            root=arguments.root,
            models=arguments.models,
            datasets=arguments.datasets,
            steps=arguments.steps,
            seed=arguments.seed,
            precision=arguments.precision,
            output_root=(
                arguments.root / "results/conditional/timing"
                if arguments.generation_mode == "conditional_prefix"
                else arguments.root / "results/timing"
            ),
            dry_run=True,
            generation_mode=arguments.generation_mode,
        )
        for record in records:
            print(json.dumps(record, sort_keys=True))
        return 1 if any(record["status"] == "error" for record in records) else 0
    if len(arguments.models) != 1 or len(arguments.datasets) != 1:
        parser.error("execution requires exactly one model and one dataset")
    try:
        output = execute_one(
            root=arguments.root,
            model=arguments.models[0],
            dataset=arguments.datasets[0],
            steps=arguments.steps,
            seed=arguments.seed,
            precision=arguments.precision,
            generation_mode=arguments.generation_mode,
        )
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"status": "succeeded", "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

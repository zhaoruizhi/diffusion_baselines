"""Render and execute registry-bound, concrete sampler timing commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Iterable

from dlb.adapters.base import AdapterError, BaseTeacherAdapter
from dlb.command import ADAPTERS
from dlb.io import atomic_json_write
from dlb.registry import load_registry
from dlb.runner import RunRequest, _resolve_request


def _run_dir(root: Path, request: RunRequest) -> Path:
    return (
        root
        / "results/samples"
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


def _request(model: str, dataset: str, steps: int, seed: int) -> RunRequest:
    return RunRequest(
        run_id=f"benchmark-{model}-{dataset}-steps-{steps}",
        model_id=model,
        dataset_id=dataset,
        step_count=steps,
        seed=seed,
        sample_count=1,
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
            if steps not in registry.step_grids[model.category]:
                records.append(
                    {
                        "status": "error",
                        "model": model_id,
                        "dataset": dataset_id,
                        "reason": f"invalid step count {steps} for {model.category}",
                    }
                )
                continue
            adapter = ADAPTERS[model_id]
            request = _request(model_id, dataset_id, steps, seed)
            output = _output_path(output_root.resolve(), request)
            metadata_path = output.with_name("benchmark_metadata.json")
            try:
                command = adapter.render_benchmark_command(
                    request,
                    _run_dir(root, request),
                    output=output,
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
                    "metadata_path": str(metadata_path),
                    "command": command,
                }
            )
    return records


def _metadata(request: RunRequest, adapter: BaseTeacherAdapter) -> dict[str, object]:
    return {
        "seed": request.seed,
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
        **adapter.author_precision_policy(request),
    }


def execute_one(
    *, root: Path, model: str, dataset: str, steps: int, seed: int, precision: str
) -> Path:
    """Resolve provenance, then execute one real sampler wrapper in this environment."""

    root = root.resolve()
    request = _request(model, dataset, steps, seed)
    adapter: BaseTeacherAdapter = ADAPTERS[model]
    resolved, adapter = _resolve_request(request, root, adapter)
    output = _output_path(root / "results/timing", resolved)
    metadata_path = output.with_name("benchmark_metadata.json")
    atomic_json_write(metadata_path, _metadata(resolved, adapter))
    command = adapter.render_benchmark_command(
        resolved,
        _run_dir(root, resolved),
        output=output,
        metadata_path=metadata_path,
        precision=precision,
        dry_run=False,
    )
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode:
        raise RuntimeError(f"benchmark sampler exited with status {completed.returncode}")
    if output.is_symlink() or not output.is_file():
        raise RuntimeError("benchmark sampler completed without atomic timing output")
    return output


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
            output_root=arguments.root / "results/timing",
            dry_run=True,
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
        )
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"status": "succeeded", "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Resumable, provenance-bound execution of one baseline sample request."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import argparse
import hashlib
import importlib
import inspect
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import socket
import stat
import subprocess
import sys
from typing import Iterable, Protocol, Sequence

from dlb.checkpoints import load_checkpoint_manifest
from dlb.io import (
    SampleValidationError,
    atomic_json_write,
    ensure_safe_directory,
    open_safe_output,
    remove_safe_file,
    sha256_file,
    validate_samples,
    write_samples_atomic,
)
from dlb.registry import load_registry
from dlb.schema import SampleRecord


class SampleAdapter(Protocol):
    """Small adapter boundary implemented by the model-specific Task 7 modules."""

    identity: str

    def build_command(self, request: "RunRequest", run_dir: Path) -> Sequence[str]: ...

    def convert_outputs(
        self, request: "RunRequest", run_dir: Path
    ) -> Iterable[SampleRecord | dict[str, object]]: ...


@dataclass(frozen=True)
class RunRequest:
    run_id: str
    model_id: str
    dataset_id: str
    step_count: int
    seed: int
    sample_count: int = 1024
    command: list[str] | None = None
    config_sha256: str | None = None
    source_sha256: str | None = None
    checkpoint_sha256: str | None = None
    checkpoint_lock_id: str | None = None
    checkpoint_selection: dict[str, object] | None = None
    checkpoint_teacher_family: str | None = None
    adapter_identity: str | None = None
    environment: str | None = None
    device: str | None = None
    results_root: str | None = None


@dataclass(frozen=True)
class RunResult:
    status: str
    run_dir: Path
    returncode: int


@dataclass(frozen=True)
class CheckpointProvenance:
    sha256: str
    lock_id: str
    selection: dict[str, object]
    teacher_family: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tail(path: Path, limit: int = 8192) -> str:
    with path.open("rb") as log_file:
        log_file.seek(0, os.SEEK_END)
        size = log_file.tell()
        log_file.seek(max(0, size - limit))
        return log_file.read().decode("utf-8", errors="replace")


def _safe_json_load(path: Path) -> dict[str, object] | None:
    try:
        if path.is_symlink() or not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _environment_evidence() -> dict[str, object]:
    torch_version: str | None
    try:
        torch_version = importlib_metadata.version("torch")
    except importlib_metadata.PackageNotFoundError:
        torch_version = None
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_version": os.environ.get("CUDA_VERSION"),
        "torch_version": torch_version,
    }


def _adapter_identity(adapter: SampleAdapter, request: RunRequest) -> str:
    del request
    declared_identity = getattr(adapter, "identity", None)
    if not isinstance(declared_identity, str) or not declared_identity:
        raise ValueError("adapter must declare a non-empty identity")
    try:
        source_path = inspect.getsourcefile(type(adapter))
    except (OSError, TypeError):
        source_path = None
    class_identity = f"{type(adapter).__module__}:{type(adapter).__qualname__}"
    if source_path is not None and Path(source_path).is_file():
        implementation = sha256_file(Path(source_path))
    else:
        implementation = "source-unavailable"
    return f"{declared_identity}|{class_identity}|{implementation}"


def _resolve_checkpoint_provenance(
    root: Path, request: RunRequest, recipe_id: str | None
) -> CheckpointProvenance:
    """Resolve a coverage resource to the exact verified lock-file inventory hash."""

    manifest_path = root / "artifacts" / "checkpoints.yaml"
    if not manifest_path.is_file():
        raise ValueError("canonical checkpoint manifest is missing")
    manifest = load_checkpoint_manifest(manifest_path)
    coverage = manifest.coverage.get((request.model_id, request.dataset_id))
    manifest_sha256 = sha256_file(manifest_path)
    if coverage is None and recipe_id is not None:
        recipe = manifest.recipes.get(recipe_id)
        if recipe is None:
            raise ValueError(f"unknown training recipe: {recipe_id}")
        output = root / recipe.output
        inventory = _checkpoint_inventory(root, output)
        selector = {
            "recipe": recipe_id,
            "output": recipe.output,
            "source": recipe.source,
            "source_commit": recipe.source_commit,
            "teacher_family": recipe.teacher_family,
            "teacher_adapter": recipe.teacher_adapter,
            "sampling_checkpoint": recipe.sampling_checkpoint,
            "sampling_config": recipe.sampling_config,
        }
        digest = _canonical_sha256(
            {"manifest_sha256": manifest_sha256, "selector": selector, "files": inventory}
        )
        return CheckpointProvenance(
            sha256=digest,
            lock_id=f"recipe:{recipe_id}:{manifest_sha256}",
            selection=selector,
            teacher_family=recipe.teacher_family,
        )
    if coverage is None:
        raise ValueError(
            f"canonical checkpoint selection is missing for {request.model_id}/{request.dataset_id}"
        )
    lock = _safe_json_load(root / "artifacts" / "checkpoint_lock.json")
    if lock is None:
        raise ValueError("canonical checkpoint lock is missing or invalid")
    if lock.get("manifest_sha256") != manifest_sha256:
        raise ValueError("checkpoint lock does not match the current checkpoint manifest")
    resources = lock.get("resources")
    record = resources.get(coverage.resource) if isinstance(resources, dict) else None
    files = record.get("files") if isinstance(record, dict) else None
    if record is None or record.get("status") != "downloaded" or not isinstance(files, list):
        raise ValueError(f"checkpoint lock is not verified for {coverage.resource}")
    inventory: list[dict[str, object]] = []
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("sha256"), str):
            raise ValueError(f"checkpoint lock has invalid file inventory for {coverage.resource}")
        if coverage.path is not None and not str(item.get("path", "")).endswith(
            "/" + coverage.path
        ):
            continue
        inventory.append(
            {
                "path": item.get("path"),
                "size_bytes": item.get("size_bytes"),
                "sha256": item["sha256"],
            }
        )
    if coverage.sampling_config_source == "project":
        config_path = root / str(coverage.sampling_config)
        if config_path.is_symlink() or not config_path.is_file():
            raise ValueError("project sampling config is missing or unsafe")
        observed_config_sha256 = sha256_file(config_path)
        if observed_config_sha256 != coverage.sampling_config_sha256:
            raise ValueError("project sampling config differs from checkpoint manifest")
        inventory.append(
            {
                "path": config_path.relative_to(root).as_posix(),
                "size_bytes": config_path.stat().st_size,
                "sha256": observed_config_sha256,
            }
        )
    if not inventory:
        raise ValueError(f"checkpoint lock lacks selected file for {coverage.resource}")
    selector = {
        "resource": coverage.resource,
        "path": coverage.path,
        "teacher_family": coverage.teacher_family,
    }
    if coverage.sampling_config is not None:
        selector.update(
            {
                "sampling_config": coverage.sampling_config,
                "sampling_config_source": coverage.sampling_config_source,
                "sampling_config_sha256": coverage.sampling_config_sha256,
                "sampling_config_source_commit": coverage.sampling_config_source_commit,
            }
        )
    digest = _canonical_sha256(
        {
            "manifest_sha256": manifest_sha256,
            "selector": selector,
            "files": sorted(inventory, key=lambda item: str(item["path"])),
        }
    )
    resource = manifest.resources[coverage.resource]
    return CheckpointProvenance(
        sha256=digest,
        lock_id=f"{coverage.resource}:{manifest_sha256}:{coverage.path or 'all'}",
        selection=selector,
        teacher_family=coverage.teacher_family or resource.teacher_family,
    )


def _resolve_checkpoint_identity(
    root: Path, request: RunRequest, recipe_id: str | None
) -> tuple[str, str]:
    """Compatibility wrapper for callers that need only the content and lock identities."""

    provenance = _resolve_checkpoint_provenance(root, request, recipe_id)
    return provenance.sha256, provenance.lock_id


def _checkpoint_inventory(root: Path, output: Path) -> list[dict[str, object]]:
    """Hash a recipe output tree while refusing symlinks and empty outputs."""

    if not output.exists() or output.is_symlink():
        raise ValueError(f"recipe checkpoint output is missing or unsafe: {output}")
    paths = [output] if output.is_file() else sorted(output.rglob("*"))
    inventory: list[dict[str, object]] = []
    for path in paths:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"recipe checkpoint output contains a symlink: {path}")
        if not stat.S_ISREG(metadata.st_mode):
            continue
        if metadata.st_size <= 0:
            raise ValueError(f"recipe checkpoint output contains an empty file: {path}")
        inventory.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": metadata.st_size,
                "sha256": sha256_file(path),
            }
        )
    if not inventory:
        raise ValueError(f"recipe checkpoint output contains no files: {output}")
    return inventory


def load_adapter(adapter_id: str) -> SampleAdapter:
    """Import a Task 7 adapter without encoding model implementation details here."""

    module_name, separator, attribute = adapter_id.partition(":")
    module = importlib.import_module(module_name if separator else f"dlb.adapters.{adapter_id}")
    candidate = getattr(module, attribute, None) if separator else getattr(module, "adapter", None)
    candidate = candidate if candidate is not None else getattr(module, "ADAPTER", None)
    if candidate is None:
        candidate = getattr(module, "Adapter", None)
    if candidate is None:
        raise ValueError(f"adapter {adapter_id!r} has no adapter, ADAPTER, or Adapter export")
    return candidate() if isinstance(candidate, type) else candidate


def _resolve_request(request: RunRequest, root: Path, adapter: SampleAdapter | None) -> tuple[RunRequest, SampleAdapter]:
    registry_path = root / "configs" / "experiments.yaml"
    if not registry_path.is_file():
        raise ValueError("canonical registry is missing: configs/experiments.yaml")
    registry = load_registry(registry_path)
    try:
        model = registry.models[request.model_id]
        support = model.datasets[request.dataset_id]
    except KeyError as error:
        raise ValueError(f"unknown model/dataset cell: {request.model_id}/{request.dataset_id}") from error
    if support.status != "supported":
        raise ValueError(f"unsupported model/dataset cell: {request.model_id}/{request.dataset_id}")
    if request.step_count not in registry.step_grids[model.category]:
        raise ValueError(f"invalid step count {request.step_count} for {model.category} category")
    if request.environment is not None and request.environment != model.environment:
        raise ValueError("environment assertion differs from canonical registry")
    if request.sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if request.step_count <= 0:
        raise ValueError("step_count must be positive")

    source_lock = _safe_json_load(root / "artifacts" / "source_lock.json")
    sources = source_lock.get("sources") if source_lock is not None else None
    source = sources.get(model.source) if isinstance(sources, dict) else None
    source_sha256 = source.get("commit") if isinstance(source, dict) else None
    if not isinstance(source_sha256, str) or not source_sha256:
        raise ValueError(f"canonical source lock lacks SHA for {model.source}")
    if request.source_sha256 is not None and request.source_sha256 != source_sha256:
        raise ValueError("source SHA assertion differs from canonical source lock")

    config_sha256 = sha256_file(registry_path)
    if request.config_sha256 is not None and request.config_sha256 != config_sha256:
        raise ValueError("config SHA assertion differs from canonical registry")

    checkpoint = _resolve_checkpoint_provenance(
        root, request, support.train_recipe
    )
    if request.checkpoint_sha256 is not None and request.checkpoint_sha256 != checkpoint.sha256:
        raise ValueError("checkpoint SHA assertion differs from canonical checkpoint identity")
    if request.checkpoint_lock_id is not None and request.checkpoint_lock_id != checkpoint.lock_id:
        raise ValueError("checkpoint lock assertion differs from canonical checkpoint identity")
    if (
        request.checkpoint_selection is not None
        and request.checkpoint_selection != checkpoint.selection
    ):
        raise ValueError("checkpoint selection assertion differs from canonical selection")
    if (
        request.checkpoint_teacher_family is not None
        and request.checkpoint_teacher_family != checkpoint.teacher_family
    ):
        raise ValueError("checkpoint teacher family differs from canonical selection")
    resolved_adapter = adapter or load_adapter(model.adapter)
    resolved = RunRequest(
        **{
            **asdict(request),
            "config_sha256": config_sha256,
            "source_sha256": source_sha256,
            "adapter_identity": _adapter_identity(resolved_adapter, request),
            "environment": model.environment,
            "checkpoint_sha256": checkpoint.sha256,
            "checkpoint_lock_id": checkpoint.lock_id,
            "checkpoint_selection": checkpoint.selection,
            "checkpoint_teacher_family": checkpoint.teacher_family,
        }
    )
    if resolved.checkpoint_sha256 is None or resolved.checkpoint_lock_id is None:
        raise ValueError("checkpoint SHA and checkpoint lock ID are required")
    return resolved, resolved_adapter


def _identity(request: RunRequest, command: list[str]) -> dict[str, object]:
    return {
        "run_id": request.run_id,
        "model_id": request.model_id,
        "dataset_id": request.dataset_id,
        "step_count": request.step_count,
        "seed": request.seed,
        "sample_count": request.sample_count,
        "device": request.device,
        "environment": request.environment,
        "adapter_identity": request.adapter_identity,
        "config_sha256": request.config_sha256,
        "source_sha256": request.source_sha256,
        "checkpoint_sha256": request.checkpoint_sha256,
        "checkpoint_lock_id": request.checkpoint_lock_id,
        "checkpoint_selection": request.checkpoint_selection,
        "checkpoint_teacher_family": request.checkpoint_teacher_family,
        "results_root": request.results_root,
        "command_sha256": _canonical_sha256(command),
    }


def _write_failure(
    run_dir: Path,
    request: RunRequest,
    *,
    stage: str,
    message: str,
    returncode: int,
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
    started_at: str,
) -> None:
    atomic_json_write(
        run_dir / "failure.json",
        {
            "run_id": request.run_id,
            "stage": stage,
            "message": message,
            "exit_code": returncode if returncode >= 0 else None,
            "signal": -returncode if returncode < 0 else None,
            "command": command,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "stdout_tail": _tail(stdout_path) if stdout_path.exists() else "",
            "stderr_tail": _tail(stderr_path) if stderr_path.exists() else "",
            "started_at": started_at,
            "finished_at": _utc_now(),
        },
    )


def run_experiment(
    request: RunRequest, root: Path | None = None, *, adapter: SampleAdapter | None = None
) -> RunResult:
    """Run, convert, validate, and atomically publish one baseline request."""

    root = (root or Path.cwd()).resolve()
    request, adapter = _resolve_request(request, root, adapter)
    results_root = (
        Path(request.results_root).absolute()
        if request.results_root is not None
        else root / "results"
    )
    if results_root.is_symlink() or not results_root.is_absolute():
        raise ValueError("results_root must be an absolute, non-symlink path")
    run_dir = (
        results_root
        / "samples"
        / request.dataset_id
        / request.model_id
        / f"steps_{request.step_count}"
    )
    ensure_safe_directory(run_dir)
    command = list(request.command) if request.command is not None else list(adapter.build_command(request, run_dir))
    if not command or any(not isinstance(argument, str) or not argument for argument in command):
        raise ValueError("adapter command must be a non-empty argv array of non-empty strings")
    identity = _identity(request, command)
    samples_path = run_dir / "samples.jsonl"
    metadata_path = run_dir / "run_metadata.json"
    metadata = _safe_json_load(metadata_path)
    if metadata is not None and metadata.get("status") == "succeeded" and metadata.get("identity") == identity:
        try:
            validate_samples(samples_path, expected=request.sample_count)
        except SampleValidationError:
            pass
        else:
            remove_safe_file(run_dir / "failure.json")
            return RunResult("skipped", run_dir, 0)
    if metadata_path.exists():
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise ValueError(f"run metadata path is unsafe: {metadata_path}")
        metadata_path.unlink()

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    started_at = _utc_now()
    atomic_json_write(
        run_dir / "request.json",
        {"request": asdict(request), "command": command, "identity": identity, "started_at": started_at},
    )
    try:
        with open_safe_output(stdout_path) as stdout_file, open_safe_output(stderr_path) as stderr_file:
            completed = subprocess.run(command, check=False, stdout=stdout_file, stderr=stderr_file)
    except OSError as error:
        _write_failure(
            run_dir, request, stage="command", message=str(error), returncode=1, command=command,
            stdout_path=stdout_path, stderr_path=stderr_path, started_at=started_at,
        )
        return RunResult("failed", run_dir, 1)
    if completed.returncode != 0:
        _write_failure(
            run_dir, request, stage="command", message="child command failed", returncode=completed.returncode,
            command=command, stdout_path=stdout_path, stderr_path=stderr_path, started_at=started_at,
        )
        return RunResult("failed", run_dir, completed.returncode)
    try:
        records = adapter.convert_outputs(request, run_dir)
        write_samples_atomic(samples_path, records, expected=request.sample_count)
        validate_samples(samples_path, expected=request.sample_count)
    except Exception as error:
        _write_failure(
            run_dir, request, stage="conversion", message=str(error), returncode=1, command=command,
            stdout_path=stdout_path, stderr_path=stderr_path, started_at=started_at,
        )
        return RunResult("failed", run_dir, 1)
    atomic_json_write(
        metadata_path,
        {
            "status": "succeeded",
            "identity": identity,
            "command": command,
            "command_sha256": identity["command_sha256"],
            "config_sha256": request.config_sha256,
            "source_sha256": request.source_sha256,
            "checkpoint_sha256": request.checkpoint_sha256,
            "checkpoint_lock_id": request.checkpoint_lock_id,
            "checkpoint_selection": request.checkpoint_selection,
            "checkpoint_teacher_family": request.checkpoint_teacher_family,
            "environment": _environment_evidence(),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "started_at": started_at,
            "finished_at": _utc_now(),
        },
    )
    remove_safe_file(run_dir / "failure.json")
    return RunResult("succeeded", run_dir, 0)


def _validate_registry_request(root: Path, model_id: str, dataset_id: str, step_count: int) -> str:
    registry_path = root / "configs" / "experiments.yaml"
    if not registry_path.is_file():
        raise ValueError("canonical registry is missing: configs/experiments.yaml")
    registry = load_registry(registry_path)
    try:
        model = registry.models[model_id]
        support = model.datasets[dataset_id]
    except KeyError as error:
        raise ValueError(f"unknown model/dataset cell: {model_id}/{dataset_id}") from error
    if support.status != "supported":
        raise ValueError(f"unsupported model/dataset cell: {model_id}/{dataset_id}")
    if step_count not in registry.step_grids[model.category]:
        raise ValueError(f"invalid step count {step_count} for {model.category} category")
    return model.environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device")
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.validate_only:
        try:
            print(_validate_registry_request(arguments.root, arguments.model, arguments.dataset, arguments.steps))
        except ValueError as error:
            parser.error(str(error))
        return 0
    request = RunRequest(
        run_id=f"{arguments.model}-{arguments.dataset}-steps-{arguments.steps}",
        model_id=arguments.model,
        dataset_id=arguments.dataset,
        step_count=arguments.steps,
        seed=arguments.seed,
        sample_count=arguments.num_samples,
        device=arguments.device,
        results_root=str(arguments.results_root.resolve()) if arguments.results_root else None,
    )
    try:
        result = run_experiment(request, arguments.root)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({"status": result.status, "run_dir": str(result.run_dir)}))
    if result.returncode < 0:
        signum = -result.returncode
        if os.name == "posix":
            import signal

            sys.stdout.flush()
            sys.stderr.flush()
            if signum not in {signal.SIGKILL, signal.SIGSTOP}:
                signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        return 128 + signum
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

"""Sampler-only generation latency measurement and atomic publication."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
from typing import Callable, Mapping


EXCLUSIONS = (
    "model_and_checkpoint_loading",
    "first_compilation",
    "token_decoding",
    "metrics",
    "file_io",
)
REQUIRED_METADATA = frozenset(
    {
        "seed",
        "dataset",
        "model",
        "steps",
        "gpu_name",
        "cuda_runtime",
        "driver_version",
        "precision",
        "requested_precision",
        "parameter_precision",
        "precision_policy",
        "synchronization_policy",
        "environment",
        "source_commit",
        "config_sha256",
        "checkpoint_sha256",
        "checkpoint_lock_id",
        "checkpoint_selection",
        "checkpoint_teacher_family",
        "adapter_identity",
    }
)


@dataclass(frozen=True)
class TimingResult:
    mode: str
    warmups: int
    repeats: int
    batch_size: int
    num_timed_samples: int
    seconds_per_sample: float | None
    median_seconds: float
    standard_deviation_seconds: float
    standard_deviation_convention: str
    raw_durations_seconds: tuple[float, ...]
    exclusions: tuple[str, ...] = EXCLUSIONS


def _integer(name: str, value: object, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        qualifier = "a positive integer" if positive else "a nonnegative integer"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def benchmark(
    generate_one: Callable[[], object],
    synchronize: Callable[[], object],
    warmups: int = 5,
    repeats: int = 32,
    *,
    clock: Callable[[], float] = time.perf_counter,
    batch_size: int = 1,
    mode: str = "primary_latency",
) -> TimingResult:
    """Warm a loaded sampler, then time only synchronized in-memory calls."""

    warmups = _integer("warmups", warmups)
    repeats = _integer("repeats", repeats, positive=True)
    batch_size = _integer("batch_size", batch_size, positive=True)
    if mode not in {"primary_latency", "throughput"}:
        raise ValueError("mode must be primary_latency or throughput")
    if mode == "primary_latency" and batch_size != 1:
        raise ValueError("primary_latency requires batch size 1")
    if not callable(generate_one) or not callable(synchronize) or not callable(clock):
        raise ValueError("generate_one, synchronize, and clock must be callable")

    for _ in range(warmups):
        generate_one()

    durations: list[float] = []
    for _ in range(repeats):
        synchronize()
        started = clock()
        generate_one()
        synchronize()
        stopped = clock()
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (started, stopped)):
            raise ValueError("clock samples must be finite numbers")
        duration = float(stopped - started)
        if duration < 0 or not math.isfinite(duration):
            raise ValueError("clock duration must be finite and nonnegative")
        durations.append(duration)

    mean = statistics.fmean(durations)
    return TimingResult(
        mode=mode,
        warmups=warmups,
        repeats=repeats,
        batch_size=batch_size,
        num_timed_samples=repeats * batch_size,
        seconds_per_sample=mean if mode == "primary_latency" else None,
        median_seconds=statistics.median(durations),
        standard_deviation_seconds=statistics.pstdev(durations),
        standard_deviation_convention="population",
        raw_durations_seconds=tuple(durations),
    )


def _validate_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    missing = sorted(REQUIRED_METADATA - metadata.keys())
    if missing:
        raise ValueError("timing metadata is missing: " + ", ".join(missing))
    value = dict(metadata)
    if type(value["seed"]) is not int or type(value["steps"]) is not int or value["steps"] <= 0:
        raise ValueError("timing metadata seed/steps are invalid")
    for key in REQUIRED_METADATA - {"seed", "steps", "checkpoint_selection"}:
        if not isinstance(value[key], str) or not value[key]:
            raise ValueError(f"timing metadata {key} must be a non-empty string")
    if not isinstance(value["checkpoint_selection"], dict) or not value["checkpoint_selection"]:
        raise ValueError("timing metadata checkpoint_selection must be a non-empty mapping")
    for key, length in (("source_commit", 40), ("config_sha256", 64), ("checkpoint_sha256", 64)):
        digest = value[key]
        if len(digest) != length or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"timing metadata {key} is not canonical hexadecimal")
    return value


def publish_timing(path: Path, result: TimingResult, metadata: Mapping[str, object]) -> None:
    """Validate everything before atomically replacing a completed timing result."""

    metadata_document = _validate_metadata(metadata)
    output = Path(path).absolute()
    if output.is_symlink() or output.parent.is_symlink():
        raise ValueError(f"timing output path is unsafe: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "dlb-generation-timing-v1",
        "timing": asdict(result),
        "metadata": metadata_document,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_metadata(path: Path) -> dict[str, object]:
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"benchmark metadata is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("benchmark metadata is invalid") from error
    if not isinstance(value, dict):
        raise ValueError("benchmark metadata must be a mapping")
    return value


def _driver_version(torch_module: object) -> str:
    internal = getattr(getattr(torch_module, "_C", None), "_cuda_getDriverVersion", None)
    if callable(internal):
        observed = internal()
        if observed:
            return str(observed)
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        value = completed.stdout.splitlines()[0].strip()
        if value:
            return value
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return "unavailable"


def cuda_runtime_metadata() -> dict[str, str]:
    """Collect GPU evidence after initialization but outside every timed interval."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("generation timing requires a CUDA server")
    device = torch.cuda.current_device()
    runtime = getattr(torch.version, "cuda", None)
    return {
        "gpu_name": str(torch.cuda.get_device_name(device)),
        "cuda_runtime": str(runtime or "unavailable"),
        "driver_version": _driver_version(torch),
        "synchronization_policy": "torch.cuda.synchronize_before_start_and_after_generate",
    }


def observed_model_precision(model: object) -> str:
    """Record parameter storage dtype separately from upstream compute autocast."""

    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        raise ValueError("benchmark model does not expose parameters for precision validation")
    aliases = {
        "torch.float32": "fp32",
        "float32": "fp32",
        "torch.float16": "fp16",
        "float16": "fp16",
        "torch.bfloat16": "bf16",
        "bfloat16": "bf16",
    }
    observed = {aliases.get(str(parameter.dtype), str(parameter.dtype)) for parameter in parameters()}
    if not observed:
        raise ValueError("benchmark model has no parameters for precision validation")
    unsupported = observed - {"fp32", "fp16", "bf16"}
    if unsupported:
        raise ValueError("benchmark model has unsupported parameter precision: " + ", ".join(sorted(unsupported)))
    return next(iter(observed)) if len(observed) == 1 else "mixed(" + ",".join(sorted(observed)) + ")"


def benchmark_and_publish(
    generate_one: Callable[[], object],
    *,
    model: object,
    output: Path,
    metadata_path: Path,
    precision: str,
    synchronize: Callable[[], object] | None = None,
) -> object:
    """Server hook used only after a concrete adapter has loaded its real model."""

    import torch

    if precision != "author":
        raise ValueError("server timing accepts only the pinned author precision policy")
    parameter_precision = observed_model_precision(model)
    metadata = load_metadata(metadata_path)
    if (
        metadata.get("requested_precision") != "author"
        or metadata.get("precision") != "bf16-mixed"
        or not isinstance(metadata.get("precision_policy"), str)
    ):
        raise ValueError("benchmark metadata does not bind an authoritative precision policy")
    synchronize = synchronize or torch.cuda.synchronize
    latest: object = None

    def generate() -> object:
        nonlocal latest
        latest = generate_one()
        return latest

    result = benchmark(generate, synchronize, warmups=5, repeats=32, batch_size=1)
    metadata = {
        **metadata,
        "parameter_precision": parameter_precision,
        **cuda_runtime_metadata(),
    }
    publish_timing(output, result, metadata)
    return latest

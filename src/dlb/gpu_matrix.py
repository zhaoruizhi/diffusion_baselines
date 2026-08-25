"""Run matrix tasks concurrently across a fixed local GPU set."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import csv
import json
import os
from pathlib import Path
from queue import Queue
import re
import subprocess
import sys
import threading
from typing import Sequence

from dlb.conditional_matrix import (
    CONDITIONAL_MATRIX_COLUMNS,
    CONDITIONAL_MATRIX_SCHEMA,
    ConditionalMatrixTask,
)
from dlb.matrix import MatrixTask, read_matrix


DEFAULT_GPUS = "0,1,2,3"
DEFAULT_MAX_JOBS = 4
DEFAULT_METRICS = "gen_ppl,entropy,self_bleu"
SAFE_LOG_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class TaskResult:
    task: MatrixTask | ConditionalMatrixTask
    gpu: str
    exit_code: int
    stdout_log: Path
    stderr_log: Path


def parse_gpus(value: str) -> tuple[str, ...]:
    """Parse comma/space separated GPU IDs while rejecting shell syntax."""

    parts = tuple(part for part in re.split(r"[,\s]+", value.strip()) if part)
    if not parts:
        raise ValueError("at least one GPU must be supplied")
    allowed = re.compile(r"^[A-Za-z0-9_.:-]+$")
    invalid = [part for part in parts if not allowed.fullmatch(part)]
    if invalid:
        raise ValueError("GPU IDs may contain only letters, numbers, '.', '_', ':', or '-'")
    if len(set(parts)) != len(parts):
        raise ValueError("GPU IDs must not be duplicated")
    return parts


def active_gpus(gpus: Sequence[str], max_jobs: int) -> tuple[str, ...]:
    if max_jobs <= 0:
        raise ValueError("max jobs must be positive")
    return tuple(gpus[index % len(gpus)] for index in range(max_jobs))


def _script(root: Path, name: str) -> str:
    path = root / "scripts" / name
    return str(path)


def command_for_task(
    task: MatrixTask | ConditionalMatrixTask,
    *,
    root: Path,
    stage: str,
    metrics: str = DEFAULT_METRICS,
    precision: str = "author",
) -> list[str]:
    """Return the argv for one matrix row and stage."""

    conditional = getattr(task, "protocol", None) == "c64_zs_v1"
    if stage in {"smoke", "generate"}:
        script = "run_conditional_one.sh" if conditional else "run_one.sh"
        command = [
            "bash",
            _script(root, script),
            "--model",
            task.model,
            "--dataset",
            task.dataset,
            "--steps",
            str(task.steps),
        ]
        if conditional:
            command.extend(["--seed", str(task.seed)])
        else:
            command.extend(["--num-samples", str(task.sample_count), "--seed", str(task.seed)])
        if stage == "smoke":
            smoke_root = root / "results" / ("conditional/smoke" if conditional else "smoke")
            command.extend(["--results-root", str(smoke_root)])
        return command
    if stage == "evaluate":
        eval_python = os.environ.get("DLB_EVAL_PYTHON")
        if eval_python:
            command = [eval_python]
        else:
            command = [
                os.environ.get("DLB_CONDA", "conda"),
                "run",
                "-n",
                os.environ.get("DLB_EVAL_ENV", "dlb-eval"),
                "python",
            ]
        return [
            *command,
            "-m",
            "evaluation.conditional_evaluate" if conditional else "evaluation.evaluate",
            "--root",
            str(root),
            "--samples",
            str(Path(task.sample_dir) / "samples.jsonl"),
            "--metrics",
            metrics,
            "--dataset",
            task.dataset,
            "--output",
            task.metrics_path,
        ]
    if stage == "benchmark":
        return [
            "bash",
            _script(root, "benchmark_conditional_one.sh" if conditional else "benchmark_one.sh"),
            "--model",
            task.model,
            "--dataset",
            task.dataset,
            "--steps",
            str(task.steps),
            "--seed",
            str(task.seed),
            "--precision",
            precision,
        ]
    raise ValueError(f"unsupported stage: {stage}")


def _safe_log_name(task_id: str) -> str:
    return SAFE_LOG_COMPONENT.sub("_", task_id)


def _task_environment(root: Path, gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["DLB_ROOT"] = str(root)
    source_path = str(root / "src")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = source_path if not current else source_path + os.pathsep + current
    return env


def _run_task(
    task: MatrixTask,
    *,
    root: Path,
    stage: str,
    gpu: str,
    metrics: str,
    precision: str,
    print_lock: threading.Lock,
) -> TaskResult:
    logs = root / "results" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    name = _safe_log_name(task.task_id)
    stdout_log = logs / f"4gpu-{stage}-{name}.out"
    stderr_log = logs / f"4gpu-{stage}-{name}.err"
    command = command_for_task(
        task, root=root, stage=stage, metrics=metrics, precision=precision
    )
    with print_lock:
        print(f"{stage.upper()} {task.task_id} gpu={gpu}", flush=True)
    with stdout_log.open("w", encoding="utf-8") as stdout, stderr_log.open(
        "w", encoding="utf-8"
    ) as stderr:
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                env=_task_environment(root, gpu),
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
            exit_code = completed.returncode
        except OSError as error:
            stderr.write(f"failed to start command: {error}\n")
            exit_code = 127
    return TaskResult(
        task=task,
        gpu=gpu,
        exit_code=exit_code,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
    )


def _failure_path(root: Path, stage: str) -> Path:
    names = {"generate": "generation", "evaluate": "evaluation"}
    name = names.get(stage, stage)
    return root / "results" / "logs" / f"{name}_4gpu_failures.tsv"


def _write_failures(root: Path, stage: str, results: Sequence[TaskResult]) -> None:
    failures = [result for result in results if result.exit_code != 0]
    path = _failure_path(root, stage)
    if not failures:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "task_id",
                    "model",
                    "dataset",
                    "steps",
                    "gpu",
                    "exit_code",
                    "stdout_log",
                    "stderr_log",
                ),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for result in failures:
                writer.writerow(
                    {
                        "task_id": result.task.task_id,
                        "model": result.task.model,
                        "dataset": result.task.dataset,
                        "steps": result.task.steps,
                        "gpu": result.gpu,
                        "exit_code": result.exit_code,
                        "stdout_log": str(result.stdout_log),
                        "stderr_log": str(result.stderr_log),
                    }
                )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def dry_run(
    tasks: Sequence[MatrixTask | ConditionalMatrixTask],
    *,
    root: Path,
    stage: str,
    gpus: Sequence[str],
    metrics: str,
    precision: str,
) -> int:
    for index, task in enumerate(tasks):
        gpu = gpus[index % len(gpus)]
        command = command_for_task(
            task, root=root, stage=stage, metrics=metrics, precision=precision
        )
        print(
            json.dumps(
                {
                    "stage": stage,
                    "task_id": task.task_id,
                    "gpu": gpu,
                    "cuda_visible_devices": gpu,
                    "command": command,
                },
                sort_keys=True,
            )
        )
    return 0


def run_tasks(
    tasks: Sequence[MatrixTask | ConditionalMatrixTask],
    *,
    root: Path,
    stage: str,
    gpus: Sequence[str],
    metrics: str,
    precision: str,
) -> int:
    gpu_queue: Queue[str] = Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)
    print_lock = threading.Lock()
    results: list[TaskResult] = []

    def run_with_gpu(task: MatrixTask | ConditionalMatrixTask) -> TaskResult:
        gpu = gpu_queue.get()
        try:
            return _run_task(
                task,
                root=root,
                stage=stage,
                gpu=gpu,
                metrics=metrics,
                precision=precision,
                print_lock=print_lock,
            )
        finally:
            gpu_queue.put(gpu)

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(run_with_gpu, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result.exit_code != 0:
                with print_lock:
                    print(
                        f"FAILED {result.task.task_id} gpu={result.gpu} "
                        f"exit={result.exit_code}",
                        file=sys.stderr,
                        flush=True,
                    )

    _write_failures(root, stage, results)
    return 1 if any(result.exit_code != 0 for result in results) else 0


def _read_conditional_matrix(path: Path) -> list[ConditionalMatrixTask]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"conditional matrix TSV is missing or unsafe: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        schema = handle.readline().rstrip("\r\n")
        if schema != f"# schema={CONDITIONAL_MATRIX_SCHEMA}":
            raise ValueError("conditional matrix TSV has an unsupported schema header")
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != CONDITIONAL_MATRIX_COLUMNS:
            raise ValueError("conditional matrix TSV columns do not match the canonical schema")
        tasks: list[ConditionalMatrixTask] = []
        for row_number, row in enumerate(reader, start=3):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"conditional matrix TSV row {row_number} has missing fields")
            try:
                tasks.append(
                    ConditionalMatrixTask(
                        task_id=row["task_id"],
                        category=row["category"],
                        model=row["model"],
                        dataset=row["dataset"],
                        steps=int(row["steps"]),
                        sample_count=2048,
                        seed=int(row["seed"]),
                        environment=row["environment"],
                        adapter=row["adapter"],
                        source=row["source"],
                        provenance=row["provenance"],
                        protocol="c64_zs_v1",
                        conditioning_manifest=row["conditioning_manifest"],
                        conditioning_manifest_sha256=row["conditioning_manifest_sha256"],
                        sample_dir=row["sample_dir"],
                        metrics_path=row["metrics_path"],
                        timing_path=row["timing_path"],
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid conditional matrix TSV row {row_number}") from error
    return tasks


def read_any_matrix(path: Path) -> list[MatrixTask | ConditionalMatrixTask]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        schema = handle.readline().rstrip("\r\n")
    if schema == f"# schema={CONDITIONAL_MATRIX_SCHEMA}":
        return _read_conditional_matrix(path)
    return read_matrix(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "generate", "evaluate", "benchmark"), required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--gpus", default=os.environ.get("DLB_GPUS", DEFAULT_GPUS))
    parser.add_argument("--max-jobs", type=int, default=int(os.environ.get("DLB_MAX_JOBS", DEFAULT_MAX_JOBS)))
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument("--precision", choices=("author",), default="author")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)

    root = arguments.root.resolve()
    try:
        gpu_list = active_gpus(parse_gpus(arguments.gpus), arguments.max_jobs)
    except ValueError as error:
        parser.error(str(error))
    tasks = read_any_matrix(arguments.matrix)
    if arguments.dry_run:
        return dry_run(
            tasks,
            root=root,
            stage=arguments.stage,
            gpus=gpu_list,
            metrics=arguments.metrics,
            precision=arguments.precision,
        )
    return run_tasks(
        tasks,
        root=root,
        stage=arguments.stage,
        gpus=gpu_list,
        metrics=arguments.metrics,
        precision=arguments.precision,
    )


if __name__ == "__main__":
    raise SystemExit(main())

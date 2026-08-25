"""Canonical generation matrix construction and TSV serialization.

The matrix is deliberately a small, dependency-light artifact.  It contains
only supported registry cells; unsupported cells are written to a separate
inventory so a skipped experiment is distinguishable from a missing row.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Iterable

from dlb.registry import ExperimentRegistry, load_registry, step_grid_for_model


MATRIX_SCHEMA = "dlb-generation-matrix-v1"
MATRIX_COLUMNS = (
    "task_id",
    "category",
    "model",
    "dataset",
    "steps",
    "sample_count",
    "seed",
    "environment",
    "adapter",
    "source",
    "provenance",
    "sample_dir",
    "metrics_path",
    "timing_path",
)
UNSUPPORTED_COLUMNS = ("status", "model", "dataset", "category", "reason")
SUPPORTED_STATUS = "supported"


@dataclass(frozen=True)
class MatrixTask:
    """One supported model/dataset/step generation request."""

    task_id: str
    category: str
    model: str
    dataset: str
    steps: int
    sample_count: int
    seed: int
    environment: str
    adapter: str
    source: str
    provenance: str
    sample_dir: str
    metrics_path: str
    timing_path: str

    @property
    def model_id(self) -> str:
        return self.model

    @property
    def dataset_id(self) -> str:
        return self.dataset

    @property
    def step_count(self) -> int:
        return self.steps

    def row(self) -> dict[str, str]:
        value = asdict(self)
        return {key: str(value[key]) for key in MATRIX_COLUMNS}


def _root_path(root: Path | None) -> Path:
    return (root or Path(".")).resolve()


def _task_paths(root: Path, model: str, dataset: str, steps: int) -> tuple[str, str, str]:
    sample_dir = root / "results" / "samples" / dataset / model / f"steps_{steps}"
    metrics = root / "results" / "metrics" / dataset / model / f"steps_{steps}" / "metrics.json"
    timing = root / "results" / "timing" / dataset / model / f"steps_{steps}" / "timing.json"
    return tuple(path.as_posix() for path in (sample_dir, metrics, timing))


def build_matrix(
    registry: ExperimentRegistry,
    *,
    root: Path | None = None,
    sample_count: int = 1024,
    seed: int = 42,
) -> list[MatrixTask]:
    """Expand every supported registry cell over its declared step grid."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    root_path = _root_path(root)
    tasks: list[MatrixTask] = []
    for model_id, model in registry.models.items():
        for dataset_id, support in model.datasets.items():
            if support.status != SUPPORTED_STATUS:
                continue
            for steps in step_grid_for_model(registry, model_id):
                task_id = f"{model_id}-{dataset_id}-steps-{steps}"
                sample_dir, metrics_path, timing_path = _task_paths(
                    root_path, model_id, dataset_id, steps
                )
                tasks.append(
                    MatrixTask(
                        task_id=task_id,
                        category=model.category,
                        model=model_id,
                        dataset=dataset_id,
                        steps=steps,
                        sample_count=sample_count,
                        seed=seed,
                        environment=model.environment,
                        adapter=model.adapter,
                        source=model.source,
                        provenance=support.provenance or "",
                        sample_dir=sample_dir,
                        metrics_path=metrics_path,
                        timing_path=timing_path,
                    )
                )
    tasks.sort(key=lambda item: (item.model, item.dataset, item.steps))
    _validate_tasks(tasks)
    return tasks


def unsupported_inventory(registry: ExperimentRegistry) -> list[dict[str, str]]:
    """Return explicit unsupported cells without expanding them into steps."""

    records: list[dict[str, str]] = []
    for model_id, model in registry.models.items():
        for dataset_id, support in model.datasets.items():
            if support.status == SUPPORTED_STATUS:
                continue
            records.append(
                {
                    "status": "unsupported",
                    "model": model_id,
                    "dataset": dataset_id,
                    "category": model.category,
                    "reason": support.reason or "no reason recorded",
                }
            )
    records.sort(key=lambda item: (item["model"], item["dataset"]))
    return records


def write_unsupported_inventory(
    path: Path, records: Iterable[dict[str, str]], *, schema: str = MATRIX_SCHEMA
) -> Path:
    """Write the explicit unsupported model/dataset inventory as TSV."""

    path = Path(path).absolute()
    values = list(records)
    for record in values:
        if tuple(record) != UNSUPPORTED_COLUMNS:
            raise ValueError("unsupported inventory columns do not match the canonical schema")
        if any(
            "\t" in value or "\n" in value or "\r" in value
            for value in record.values()
        ):
            raise ValueError("unsupported inventory contains a control character")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(f"# schema={schema}-unsupported\n")
            writer = csv.DictWriter(
                handle, fieldnames=UNSUPPORTED_COLUMNS, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _validate_tasks(tasks: Iterable[MatrixTask]) -> None:
    seen: set[str] = set()
    previous: tuple[str, str, int] | None = None
    for task in tasks:
        if task.task_id in seen:
            raise ValueError(f"duplicate matrix task ID: {task.task_id}")
        seen.add(task.task_id)
        key = (task.model, task.dataset, task.steps)
        if previous is not None and key < previous:
            raise ValueError("matrix tasks are not in stable order")
        previous = key
        if task.category not in {"many", "few", "fixed_1024"}:
            raise ValueError(f"invalid task category: {task.category}")
        if task.steps <= 0 or task.sample_count <= 0:
            raise ValueError("steps and sample_count must be positive")
        if any("\t" in value or "\n" in value or "\r" in value for value in task.row().values()):
            raise ValueError(f"matrix task contains a control character: {task.task_id}")


def write_matrix(path: Path, tasks: Iterable[MatrixTask]) -> Path:
    """Write a deterministic, versioned TSV matrix."""

    path = Path(path).absolute()
    task_list = list(tasks)
    _validate_tasks(task_list)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(f"# schema={MATRIX_SCHEMA}\n")
            writer = csv.DictWriter(
                handle, fieldnames=MATRIX_COLUMNS, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            for task in task_list:
                writer.writerow(task.row())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def read_matrix(path: Path) -> list[MatrixTask]:
    """Read and validate a matrix TSV without evaluating any field as code."""

    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"matrix TSV is missing or unsafe: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        schema = handle.readline().rstrip("\r\n")
        if schema != f"# schema={MATRIX_SCHEMA}":
            raise ValueError("matrix TSV has an unsupported schema header")
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != MATRIX_COLUMNS:
            raise ValueError("matrix TSV columns do not match the canonical schema")
        tasks: list[MatrixTask] = []
        for row_number, row in enumerate(reader, start=3):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"matrix TSV row {row_number} has missing fields")
            try:
                task = MatrixTask(
                    task_id=row["task_id"],
                    category=row["category"],
                    model=row["model"],
                    dataset=row["dataset"],
                    steps=int(row["steps"]),
                    sample_count=int(row["sample_count"]),
                    seed=int(row["seed"]),
                    environment=row["environment"],
                    adapter=row["adapter"],
                    source=row["source"],
                    provenance=row["provenance"],
                    sample_dir=row["sample_dir"],
                    metrics_path=row["metrics_path"],
                    timing_path=row["timing_path"],
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid matrix TSV row {row_number}") from error
            tasks.append(task)
    _validate_tasks(tasks)
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("matrix TSV contains duplicate task IDs")
    return tasks


def validate_matrix(path: Path, registry: ExperimentRegistry | None = None) -> list[MatrixTask]:
    """Validate a matrix and optionally compare it with the canonical registry."""

    tasks = read_matrix(path)
    if registry is not None:
        expected = build_matrix(
            registry,
            root=Path("."),
            sample_count=tasks[0].sample_count if tasks else 1024,
            seed=tasks[0].seed if tasks else 42,
        )
        actual_keys = [(task.model, task.dataset, task.steps) for task in tasks]
        expected_keys = [(task.model, task.dataset, task.steps) for task in expected]
        if actual_keys != expected_keys:
            raise ValueError("matrix TSV coverage differs from the canonical registry")
    return tasks


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unsupported-output", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    registry = load_registry(root / "configs" / "experiments.yaml")
    if args.validate:
        tasks = validate_matrix(args.output, registry)
    else:
        tasks = build_matrix(
            registry, root=root, sample_count=args.sample_count, seed=args.seed
        )
        write_matrix(args.output, tasks)
        write_unsupported_inventory(
            args.unsupported_output
            or args.output.with_name("unsupported.tsv"),
            unsupported_inventory(registry),
        )
    print(f"matrix={args.output} tasks={len(tasks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

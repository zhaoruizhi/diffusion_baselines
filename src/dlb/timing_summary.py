"""Summarize independent generation timing artifacts without quality metrics."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, TextIO

from dlb.matrix import MatrixTask, build_matrix
from dlb.registry import load_registry


TIMING_SUMMARY_COLUMNS = ("dataset", "model", "steps", "seconds_per_sample")


@dataclass(frozen=True)
class TimingSummaryRow:
    dataset: str
    model: str
    steps: int
    seconds_per_sample: float

    def as_csv_row(self) -> dict[str, str]:
        return {
            "dataset": self.dataset,
            "model": self.model,
            "steps": str(self.steps),
            "seconds_per_sample": f"{self.seconds_per_sample:.6f}",
        }


@dataclass(frozen=True)
class TimingIssue:
    dataset: str
    model: str
    steps: int
    path: Path
    reason: str


@dataclass(frozen=True)
class TimingSummaryReport:
    rows: tuple[TimingSummaryRow, ...]
    missing: tuple[TimingIssue, ...]
    invalid: tuple[TimingIssue, ...]

    @property
    def expected_tasks(self) -> int:
        return len(self.rows) + len(self.missing) + len(self.invalid)


def _finite_seconds(value: object) -> float:
    if type(value) not in {int, float} or isinstance(value, bool):
        raise ValueError("seconds_per_sample must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("seconds_per_sample must be a finite nonnegative number")
    return result


def _load_seconds_per_sample(path: Path) -> float:
    if path.is_symlink() or not path.is_file():
        raise ValueError("timing artifact is missing or unsafe")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("timing artifact is not valid JSON") from error
    if not isinstance(document, Mapping):
        raise ValueError("timing artifact must be a JSON object")
    if document.get("schema") != "dlb-generation-timing-v1":
        raise ValueError("timing artifact has an unsupported schema")
    timing = document.get("timing")
    if not isinstance(timing, Mapping):
        raise ValueError("timing artifact has no timing object")
    protocol = {
        "mode": "primary_latency",
        "warmups": 5,
        "repeats": 32,
        "batch_size": 1,
        "num_timed_samples": 32,
    }
    if any(timing.get(key) != expected for key, expected in protocol.items()):
        raise ValueError("timing artifact does not use the pinned latency protocol")
    return _finite_seconds(timing.get("seconds_per_sample"))


def summarize_timing(tasks: Iterable[MatrixTask]) -> TimingSummaryReport:
    rows: list[TimingSummaryRow] = []
    missing: list[TimingIssue] = []
    invalid: list[TimingIssue] = []
    for task in tasks:
        path = Path(task.timing_path)
        if not path.exists():
            missing.append(
                TimingIssue(task.dataset, task.model, task.steps, path, "missing timing.json")
            )
            continue
        try:
            seconds = _load_seconds_per_sample(path)
        except ValueError as error:
            invalid.append(TimingIssue(task.dataset, task.model, task.steps, path, str(error)))
            continue
        rows.append(TimingSummaryRow(task.dataset, task.model, task.steps, seconds))
    return TimingSummaryReport(tuple(rows), tuple(missing), tuple(invalid))


def write_timing_csv(rows: Iterable[TimingSummaryRow], output: TextIO) -> None:
    writer = csv.DictWriter(
        output, fieldnames=TIMING_SUMMARY_COLUMNS, lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row.as_csv_row())


def _filtered_tasks(
    root: Path, *, model: str | None = None, dataset: str | None = None
) -> list[MatrixTask]:
    registry = load_registry(root / "configs" / "experiments.yaml")
    tasks = build_matrix(registry, root=root)
    if model is not None:
        tasks = [task for task in tasks if task.model == model]
    if dataset is not None:
        tasks = [task for task in tasks if task.dataset == dataset]
    return tasks


def _write_diagnostics(report: TimingSummaryReport, output: TextIO) -> None:
    print(
        " ".join(
            [
                f"expected={report.expected_tasks}",
                f"present={len(report.rows)}",
                f"missing={len(report.missing)}",
                f"invalid={len(report.invalid)}",
            ]
        ),
        file=output,
    )
    for issue in report.missing:
        print(
            f"missing,{issue.dataset},{issue.model},{issue.steps},{issue.path}",
            file=output,
        )
    for issue in report.invalid:
        print(
            ",".join(
                [
                    "invalid",
                    issue.dataset,
                    issue.model,
                    str(issue.steps),
                    str(issue.path),
                    issue.reason,
                ]
            ),
            file=output,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize available timing seconds_per_sample artifacts."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args(argv)

    root = args.root.resolve()
    tasks = _filtered_tasks(root, model=args.model, dataset=args.dataset)
    if not tasks:
        print("no matrix tasks matched the requested filters", file=sys.stderr)
        return 2
    report = summarize_timing(tasks)
    write_timing_csv(report.rows, sys.stdout)
    _write_diagnostics(report, sys.stderr)
    return 0


__all__ = [
    "TIMING_SUMMARY_COLUMNS",
    "TimingIssue",
    "TimingSummaryReport",
    "TimingSummaryRow",
    "main",
    "summarize_timing",
    "write_timing_csv",
]

"""Audit generation, evaluation and timing artifacts into reproducible tables."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from dlb.io import validate_samples
from dlb.matrix import MatrixTask, build_matrix, unsupported_inventory
from dlb.registry import ExperimentRegistry, load_registry


QUALITY_METRICS = (
    "generative_perplexity",
    "unigram_entropy",
    "self_bleu",
)
ALL_METRICS = QUALITY_METRICS + ("generation_seconds_per_sample",)
AGGREGATE_SCHEMA = "dlb-results-summary-v1"


class IncompleteMatrixError(RuntimeError):
    """Raised when strict aggregation cannot prove a complete matrix."""

    def __init__(self, failures: Iterable[Mapping[str, Any]]) -> None:
        self.failures = [dict(item) for item in failures]
        super().__init__(
            f"matrix is incomplete: {len(self.failures)} task(s) failed validation"
        )


@dataclass(frozen=True)
class AggregateReport:
    """In-memory result of an aggregation pass."""

    rows: tuple[dict[str, Any], ...]
    failures: tuple[dict[str, Any], ...]
    unsupported: tuple[dict[str, Any], ...]
    expected_tasks: int
    complete: bool
    output_dir: Path | None = None


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"artifact is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    return value


def _finite(value: object, *, positive: bool = False) -> float:
    if type(value) not in {int, float} or isinstance(value, bool):
        raise ValueError("metric value must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0) or (not positive and result < 0):
        raise ValueError("metric value is outside the allowed finite range")
    return result


def _lookup_metric(metrics: Mapping[str, Any], name: str) -> float:
    aliases = {
        "generative_perplexity": ("generative_perplexity", "gen_ppl", "ppl"),
        "unigram_entropy": ("unigram_entropy", "entropy"),
        "self_bleu": ("self_bleu", "self_bleu_score"),
    }
    value: object = None
    found = False
    for key in aliases[name]:
        if key in metrics:
            value = metrics[key]
            found = True
            break
    if not found:
        raise ValueError(f"missing metric {name}")
    if isinstance(value, Mapping):
        field = {
            "generative_perplexity": "perplexity",
            "unigram_entropy": "mean_entropy",
            "self_bleu": "score",
        }[name]
        if field not in value:
            raise ValueError(f"metric {name} is missing field {field}")
        value = value[field]
    result = _finite(value, positive=name == "generative_perplexity")
    if name == "self_bleu" and result > 1:
        raise ValueError("self_bleu must be in [0, 1]")
    return result


def _provenance(metadata: Mapping[str, Any], task: MatrixTask) -> dict[str, Any]:
    identity = metadata.get("identity")
    identity = identity if isinstance(identity, Mapping) else {}
    nested = metadata.get("provenance")
    nested = nested if isinstance(nested, Mapping) else {}

    def first(*keys: str) -> object:
        for source in (metadata, identity, nested):
            for key in keys:
                if key in source and source[key] not in (None, ""):
                    return source[key]
        return None

    expected = {
        "model_id": task.model,
        "dataset_id": task.dataset,
        "step_count": task.steps,
        "seed": task.seed,
        "sample_count": task.sample_count,
    }
    for key, value in expected.items():
        observed = first(key, key.removesuffix("_id"))
        if observed != value:
            raise ValueError(f"run metadata {key} does not match matrix task")
    required = (
        "source_sha256",
        "config_sha256",
        "checkpoint_sha256",
        "checkpoint_lock_id",
        "checkpoint_selection",
        "checkpoint_teacher_family",
        "adapter_identity",
        "environment",
    )
    result: dict[str, Any] = {}
    for key in required:
        value = first(key)
        if value in (None, "", {}):
            raise ValueError(f"run metadata is missing provenance field {key}")
        result[key] = value
    return result


def _validate_task(task: MatrixTask) -> dict[str, Any]:
    sample_dir = Path(task.sample_dir)
    samples = sample_dir / "samples.jsonl"
    metadata_path = sample_dir / "run_metadata.json"
    metrics_path = Path(task.metrics_path)
    timing_path = Path(task.timing_path)
    validate_samples(samples, expected=task.sample_count)
    metadata = _json(metadata_path)
    if metadata.get("status") != "succeeded":
        raise ValueError("run metadata is not marked succeeded")
    provenance = _provenance(metadata, task)

    metrics_document = _json(metrics_path)
    if metrics_document.get("sample_count") != task.sample_count:
        raise ValueError("metrics sample_count does not match matrix task")
    if metrics_document.get("partial") is True:
        raise ValueError("partial metrics cannot be used for a complete aggregation")
    metric_document = metrics_document.get("metrics")
    if not isinstance(metric_document, Mapping):
        raise ValueError("metrics document has no metrics object")
    quality = {
        name: _lookup_metric(metric_document, name) for name in QUALITY_METRICS
    }

    timing_document = _json(timing_path)
    if timing_document.get("schema") != "dlb-generation-timing-v1":
        raise ValueError("timing document has an unsupported schema")
    timing = timing_document.get("timing")
    if not isinstance(timing, Mapping):
        raise ValueError("timing document has no timing object")
    protocol = {
        "mode": "primary_latency",
        "warmups": 5,
        "repeats": 32,
        "batch_size": 1,
        "num_timed_samples": 32,
    }
    if any(timing.get(key) != value for key, value in protocol.items()):
        raise ValueError("timing document does not use the pinned latency protocol")
    seconds = timing.get("seconds_per_sample")
    seconds_value = _finite(seconds)
    timing_metadata = timing_document.get("metadata")
    if not isinstance(timing_metadata, Mapping):
        raise ValueError("timing document has no metadata object")
    _provenance(timing_metadata, task)
    if timing_metadata.get("attempt_id") in (None, ""):
        raise ValueError("timing metadata is missing attempt_id")

    return {
        "task_id": task.task_id,
        "model": task.model,
        "dataset": task.dataset,
        "category": task.category,
        "steps": task.steps,
        "sample_count": task.sample_count,
        "seed": task.seed,
        "provenance": task.provenance,
        **quality,
        "generation_seconds_per_sample": seconds_value,
        "source_sha256": provenance["source_sha256"],
        "config_sha256": provenance["config_sha256"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "checkpoint_lock_id": provenance["checkpoint_lock_id"],
        "checkpoint_teacher_family": provenance["checkpoint_teacher_family"],
        "adapter_identity": provenance["adapter_identity"],
        "environment": (
            provenance["environment"]
            if isinstance(provenance["environment"], str)
            else json.dumps(provenance["environment"], sort_keys=True)
        ),
    }


def _failure(task: MatrixTask, error: Exception) -> dict[str, Any]:
    return {
        "status": "failed",
        "task_id": task.task_id,
        "model": task.model,
        "dataset": task.dataset,
        "category": task.category,
        "steps": task.steps,
        "reason": str(error),
    }


def _write_csv(path: Path, rows: list[Mapping[str, Any]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in columns})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_outputs(
    output_dir: Path,
    report: AggregateReport,
) -> None:
    output_dir = output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(report.rows)
    long_rows = [
        {
            **{
                key: row[key]
                for key in (
                    "task_id",
                    "model",
                    "dataset",
                    "category",
                    "steps",
                    "sample_count",
                    "seed",
                    "provenance",
                )
            },
            "metric": metric,
            "value": row[metric],
            "source_sha256": row["source_sha256"],
            "config_sha256": row["config_sha256"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "checkpoint_lock_id": row["checkpoint_lock_id"],
            "adapter_identity": row["adapter_identity"],
            "environment": row["environment"],
        }
        for row in rows
        for metric in ALL_METRICS
    ]
    long_columns = (
        "task_id",
        "model",
        "dataset",
        "category",
        "steps",
        "sample_count",
        "seed",
        "provenance",
        "metric",
        "value",
        "source_sha256",
        "config_sha256",
        "checkpoint_sha256",
        "checkpoint_lock_id",
        "adapter_identity",
        "environment",
    )
    wide_columns = (
        "task_id",
        "model",
        "dataset",
        "category",
        "steps",
        "sample_count",
        "seed",
        *ALL_METRICS,
        "provenance",
        "source_sha256",
        "config_sha256",
        "checkpoint_sha256",
        "checkpoint_lock_id",
        "checkpoint_teacher_family",
        "adapter_identity",
        "environment",
    )
    _write_csv(output_dir / "results_long.csv", long_rows, long_columns)
    _write_csv(output_dir / "results_wide.csv", rows, wide_columns)
    failures = list(report.failures) + [
        {**item, "status": "unsupported"} for item in report.unsupported
    ]
    failure_columns = (
        "status",
        "task_id",
        "model",
        "dataset",
        "category",
        "steps",
        "reason",
    )
    _write_csv(output_dir / "failures.csv", failures, failure_columns)
    _write_csv(
        output_dir / "unsupported.csv",
        list(report.unsupported),
        ("status", "model", "dataset", "category", "reason"),
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# Baseline Results ({AGGREGATE_SCHEMA})",
                "",
                f"- complete: `{str(report.complete).lower()}`",
                f"- expected supported tasks: `{report.expected_tasks}`",
                f"- valid result rows: `{len(report.rows)}`",
                f"- failures: `{len(report.failures)}`",
                f"- unsupported cells: `{len(report.unsupported)}`",
                "",
                "A strict result set requires every supported task to have 1,024 "
                "samples, three quality metrics, one independent timing artifact, "
                "and matching provenance.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def aggregate(
    root: Path,
    *,
    strict: bool = True,
    partial: bool = False,
    output_dir: Path | None = None,
    registry: ExperimentRegistry | None = None,
) -> AggregateReport:
    """Validate all supported tasks and optionally publish summary tables."""

    root = Path(root).resolve()
    registry = registry or load_registry(root / "configs" / "experiments.yaml")
    tasks = build_matrix(registry, root=root)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for task in tasks:
        try:
            rows.append(_validate_task(task))
        except (OSError, ValueError, TypeError) as error:
            failures.append(_failure(task, error))
    unsupported = unsupported_inventory(registry)
    complete = not failures and len(rows) == len(tasks)
    report = AggregateReport(
        rows=tuple(rows),
        failures=tuple(failures),
        unsupported=tuple(unsupported),
        expected_tasks=len(tasks),
        complete=complete,
        output_dir=Path(output_dir).resolve() if output_dir is not None else None,
    )
    if strict and not partial and not complete:
        raise IncompleteMatrixError(failures)
    if output_dir is not None:
        _write_outputs(Path(output_dir), report)
    return report


__all__ = [
    "AGGREGATE_SCHEMA",
    "ALL_METRICS",
    "AggregateReport",
    "IncompleteMatrixError",
    "QUALITY_METRICS",
    "aggregate",
]

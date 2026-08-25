"""Audit C64 conditional generation, evaluation, and timing artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from dlb.aggregate import (
    IncompleteMatrixError,
    _finite,
    _json,
    _provenance,
    _write_csv,
)
from dlb.conditional_matrix import (
    ConditionalMatrixTask,
    build_conditional_matrix,
    conditional_unsupported_inventory,
)
from dlb.io import expected_conditional_schedule, validate_conditional_samples
from dlb.registry import ExperimentRegistry, load_registry


QUALITY_METRICS = (
    "conditional_generative_perplexity",
    "mauve_suffix",
    "sample_entropy",
    "sample_entropy_delta",
    "self_bleu",
    "prefix_exact_match",
)
ALL_METRICS = QUALITY_METRICS + ("generation_seconds_per_sample",)
CONDITIONAL_AGGREGATE_SCHEMA = "dlb-conditional-results-summary-v1"
DATASET_CONTRACTS = {
    "lm1b": {"sequence_length": 128, "vocab_size": 30_522},
    "owt": {"sequence_length": 1024, "vocab_size": 50_257},
}


@dataclass(frozen=True)
class ConditionalAggregateReport:
    rows: tuple[dict[str, Any], ...]
    failures: tuple[dict[str, Any], ...]
    unsupported: tuple[dict[str, Any], ...]
    expected_tasks: int
    complete: bool
    output_dir: Path | None = None


def _finite_signed(value: object) -> float:
    if type(value) not in {int, float} or isinstance(value, bool):
        raise ValueError("metric value must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("metric value is not finite")
    return result


def _lookup_metric(metrics: Mapping[str, Any], name: str) -> float:
    if name == "conditional_generative_perplexity":
        value = metrics.get("conditional_generative_perplexity")
        if not isinstance(value, Mapping) or "perplexity" not in value:
            raise ValueError("missing conditional generative perplexity")
        return _finite(value["perplexity"], positive=True)
    if name == "mauve_suffix":
        value = metrics.get("mauve_suffix")
        if not isinstance(value, Mapping) or "score" not in value:
            raise ValueError("missing MAUVE suffix score")
        score = _finite(value["score"])
        if score > 1:
            raise ValueError("MAUVE suffix score must be in [0, 1]")
        return score
    if name == "sample_entropy":
        value = metrics.get("sample_entropy")
        generated = value.get("generated") if isinstance(value, Mapping) else None
        if not isinstance(generated, Mapping) or "mean_entropy" not in generated:
            raise ValueError("missing generated sample entropy")
        return _finite(generated["mean_entropy"])
    if name == "sample_entropy_delta":
        value = metrics.get("sample_entropy")
        if not isinstance(value, Mapping) or "generated_minus_reference" not in value:
            raise ValueError("missing sample entropy delta")
        return _finite_signed(value["generated_minus_reference"])
    if name == "self_bleu":
        value = metrics.get("self_bleu")
        if not isinstance(value, Mapping) or "score" not in value:
            raise ValueError("missing Self-BLEU score")
        score = _finite(value["score"])
        if score > 1:
            raise ValueError("Self-BLEU must be in [0, 1]")
        return score
    if name == "prefix_exact_match":
        value = metrics.get("prefix_exact_match")
        if not isinstance(value, Mapping) or "rate" not in value:
            raise ValueError("missing prefix exact-match rate")
        rate = _finite(value["rate"])
        if rate != 1.0:
            raise ValueError("prefix exact-match rate must be 100%")
        return rate
    raise ValueError(f"unknown metric {name}")


def _validate_task(task: ConditionalMatrixTask) -> dict[str, Any]:
    contract = DATASET_CONTRACTS[task.dataset]
    sample_dir = Path(task.sample_dir)
    samples = sample_dir / "samples.jsonl"
    metadata_path = sample_dir / "run_metadata.json"
    metrics_path = Path(task.metrics_path)
    timing_path = Path(task.timing_path)
    validate_conditional_samples(
        samples,
        expected=task.sample_count,
        schedule=expected_conditional_schedule(),
        sequence_length=int(contract["sequence_length"]),
        vocab_size=int(contract["vocab_size"]),
    )
    metadata = _json(metadata_path)
    if metadata.get("status") != "succeeded":
        raise ValueError("run metadata is not marked succeeded")
    identity = metadata.get("identity")
    if not isinstance(identity, Mapping) or identity.get("generation_mode") != "conditional_prefix":
        raise ValueError("run metadata is not a conditional-prefix run")
    provenance = _provenance(metadata, task)

    metrics_document = _json(metrics_path)
    if metrics_document.get("sample_count") != task.sample_count:
        raise ValueError("metrics sample_count does not match matrix task")
    if metrics_document.get("protocol") != "c64_zs_v1":
        raise ValueError("metrics document has the wrong conditional protocol")
    metric_document = metrics_document.get("metrics")
    if not isinstance(metric_document, Mapping):
        raise ValueError("metrics document has no metrics object")
    quality = {name: _lookup_metric(metric_document, name) for name in QUALITY_METRICS}

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
    seconds_value = _finite(timing.get("seconds_per_sample"))
    timing_metadata = timing_document.get("metadata")
    if not isinstance(timing_metadata, Mapping):
        raise ValueError("timing document has no metadata object")
    _provenance(timing_metadata, task)
    if timing_metadata.get("generation_mode") != "conditional_prefix":
        raise ValueError("timing metadata is not bound to conditional-prefix generation")

    return {
        "task_id": task.task_id,
        "model": task.model,
        "dataset": task.dataset,
        "category": task.category,
        "steps": task.steps,
        "sample_count": task.sample_count,
        "seed": task.seed,
        "protocol": task.protocol,
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
        "conditioning_manifest_sha256": task.conditioning_manifest_sha256,
    }


def _failure(task: ConditionalMatrixTask, error: Exception) -> dict[str, Any]:
    return {
        "status": "failed",
        "task_id": task.task_id,
        "model": task.model,
        "dataset": task.dataset,
        "category": task.category,
        "steps": task.steps,
        "reason": str(error),
    }


def _write_outputs(output_dir: Path, report: ConditionalAggregateReport) -> None:
    output_dir = output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(report.rows)
    long_rows = [
        {
            "task_id": row["task_id"],
            "model": row["model"],
            "dataset": row["dataset"],
            "category": row["category"],
            "steps": row["steps"],
            "sample_count": row["sample_count"],
            "seed": row["seed"],
            "protocol": row["protocol"],
            "provenance": row["provenance"],
            "metric": metric,
            "value": row[metric],
            "source_sha256": row["source_sha256"],
            "config_sha256": row["config_sha256"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "checkpoint_lock_id": row["checkpoint_lock_id"],
            "adapter_identity": row["adapter_identity"],
            "environment": row["environment"],
            "conditioning_manifest_sha256": row["conditioning_manifest_sha256"],
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
        "protocol",
        "provenance",
        "metric",
        "value",
        "source_sha256",
        "config_sha256",
        "checkpoint_sha256",
        "checkpoint_lock_id",
        "adapter_identity",
        "environment",
        "conditioning_manifest_sha256",
    )
    wide_columns = (
        "task_id",
        "model",
        "dataset",
        "category",
        "steps",
        "sample_count",
        "seed",
        "protocol",
        *ALL_METRICS,
        "provenance",
        "source_sha256",
        "config_sha256",
        "checkpoint_sha256",
        "checkpoint_lock_id",
        "checkpoint_teacher_family",
        "adapter_identity",
        "environment",
        "conditioning_manifest_sha256",
    )
    _write_csv(output_dir / "conditional_results_long.csv", long_rows, long_columns)
    _write_csv(output_dir / "conditional_results_wide.csv", rows, wide_columns)
    failures = list(report.failures) + [
        {**item, "status": "unsupported"} for item in report.unsupported
    ]
    _write_csv(
        output_dir / "conditional_failures.csv",
        failures,
        ("status", "task_id", "model", "dataset", "category", "steps", "reason"),
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# Conditional Baseline Results ({CONDITIONAL_AGGREGATE_SCHEMA})",
                "",
                f"- complete: `{str(report.complete).lower()}`",
                f"- expected supported tasks: `{report.expected_tasks}`",
                f"- valid result rows: `{len(report.rows)}`",
                f"- failures: `{len(report.failures)}`",
                f"- unsupported cells: `{len(report.unsupported)}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def aggregate_conditional(
    root: Path,
    *,
    strict: bool = True,
    partial: bool = False,
    output_dir: Path | None = None,
    registry: ExperimentRegistry | None = None,
) -> ConditionalAggregateReport:
    """Validate all conditional tasks and optionally publish summary tables."""

    root = Path(root).resolve()
    registry = registry or load_registry(root / "configs" / "experiments.yaml")
    tasks = build_conditional_matrix(registry, root=root)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for task in tasks:
        try:
            rows.append(_validate_task(task))
        except (OSError, ValueError, TypeError) as error:
            failures.append(_failure(task, error))
    unsupported = conditional_unsupported_inventory(registry)
    complete = not failures and len(rows) == len(tasks)
    report = ConditionalAggregateReport(
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
    "ALL_METRICS",
    "CONDITIONAL_AGGREGATE_SCHEMA",
    "ConditionalAggregateReport",
    "QUALITY_METRICS",
    "aggregate_conditional",
]

"""Read-only diagnostics for suspicious RDLM quality rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
import sys
from typing import Any, Iterable, Mapping, Sequence

from dlb.io import sha256_file


SCHEMA = "dlb-rdlm-diagnostics-v1"
DEFAULT_STEPS = (1, 2, 4, 8, 16, 32, 1024)
COLLAPSE_WARNINGS = frozenset(
    {
        "low_ppl_with_low_entropy",
        "high_self_bleu",
        "high_duplicate_texts",
        "short_decoded_texts",
        "high_padding_fraction",
    }
)


def _read_json(path: Path, warnings: list[str], label: str) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        warnings.append(f"invalid_{label}")
        return None
    if not isinstance(value, dict):
        warnings.append(f"invalid_{label}")
        return None
    return value


def _read_samples(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for expected_id, line in enumerate(source):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    warnings.append("invalid_sample_jsonl")
                    return samples
                if not isinstance(record, dict):
                    warnings.append("invalid_sample_record")
                    return samples
                if record.get("sample_id") != expected_id:
                    warnings.append("sample_id_mismatch")
                samples.append(record)
    except OSError:
        warnings.append("invalid_samples")
    return samples


def _metric(metrics: Mapping[str, Any], name: str) -> float | None:
    aliases = {
        "generative_perplexity": (
            ("generative_perplexity", "perplexity"),
            ("gen_ppl", "perplexity"),
            ("ppl", None),
        ),
        "unigram_entropy": (
            ("unigram_entropy", "mean_entropy"),
            ("entropy", "mean_entropy"),
            ("entropy", None),
        ),
        "self_bleu": (
            ("self_bleu", "score"),
            ("self_bleu_score", None),
        ),
    }[name]
    for key, field in aliases:
        if key not in metrics:
            continue
        value = metrics[key]
        if field is not None:
            if not isinstance(value, Mapping) or field not in value:
                continue
            value = value[field]
        if type(value) in {int, float} and not isinstance(value, bool):
            result = float(value)
            return result if math.isfinite(result) else None
    return None


def _command_override(command: object, key: str) -> str | None:
    if not isinstance(command, list):
        return None
    prefix = key + "="
    values = [
        argument[len(prefix) :]
        for argument in command
        if isinstance(argument, str) and argument.startswith(prefix)
    ]
    return values[0] if len(values) == 1 else None


def _sample_stats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    texts = [item.get("text") for item in samples if isinstance(item.get("text"), str)]
    token_rows = [
        item.get("token_ids")
        for item in samples
        if isinstance(item.get("token_ids"), list)
    ]
    text_lengths = [len(text) for text in texts]
    token_lengths = [len(tokens) for tokens in token_rows]
    flat_tokens = [
        token
        for tokens in token_rows
        for token in tokens
        if type(token) is int and token >= 0
    ]
    total_tokens = len(flat_tokens)
    return {
        "sample_count": len(samples),
        "unique_text_count": len(set(texts)),
        "unique_text_ratio": len(set(texts)) / len(texts) if texts else None,
        "mean_text_chars": mean(text_lengths) if text_lengths else None,
        "median_text_chars": median(text_lengths) if text_lengths else None,
        "mean_token_length": mean(token_lengths) if token_lengths else None,
        "min_token_length": min(token_lengths) if token_lengths else None,
        "max_token_length": max(token_lengths) if token_lengths else None,
        "pad_token_fraction": (
            sum(1 for token in flat_tokens if token == 0) / total_tokens
            if total_tokens
            else None
        ),
        "cls_text_fraction": (
            sum(1 for text in texts if "[CLS]" in text) / len(texts) if texts else None
        ),
        "sep_text_fraction": (
            sum(1 for text in texts if "[SEP]" in text) / len(texts) if texts else None
        ),
    }


def _add_quality_warnings(
    row: dict[str, Any],
    warnings: list[str],
    *,
    expected_sample_count: int,
    low_ppl_threshold: float,
    low_entropy_threshold: float,
    high_self_bleu_threshold: float,
    duplicate_unique_ratio_threshold: float,
    short_text_chars_threshold: float,
    high_padding_fraction_threshold: float,
) -> None:
    if row.get("sample_count") not in {None, expected_sample_count}:
        warnings.append("sample_count_mismatch")
    if row.get("metrics_sample_count") not in {None, expected_sample_count}:
        warnings.append("metrics_sample_count_mismatch")
    if (
        row.get("generative_perplexity") is not None
        and row.get("unigram_entropy") is not None
        and row["generative_perplexity"] < low_ppl_threshold
        and row["unigram_entropy"] < low_entropy_threshold
    ):
        warnings.append("low_ppl_with_low_entropy")
    if (
        row.get("self_bleu") is not None
        and row["self_bleu"] > high_self_bleu_threshold
    ):
        warnings.append("high_self_bleu")
    if (
        row.get("unique_text_ratio") is not None
        and row["unique_text_ratio"] < duplicate_unique_ratio_threshold
    ):
        warnings.append("high_duplicate_texts")
    if (
        row.get("median_text_chars") is not None
        and row["median_text_chars"] < short_text_chars_threshold
    ):
        warnings.append("short_decoded_texts")
    if (
        row.get("pad_token_fraction") is not None
        and row["pad_token_fraction"] > high_padding_fraction_threshold
    ):
        warnings.append("high_padding_fraction")
    if row.get("token_ids_source") == "retokenized":
        warnings.append("retokenized_from_text")


def _row_for_step(
    root: Path,
    *,
    dataset: str,
    model: str,
    step: int,
    expected_sample_count: int,
    low_ppl_threshold: float,
    low_entropy_threshold: float,
    high_self_bleu_threshold: float,
    duplicate_unique_ratio_threshold: float,
    short_text_chars_threshold: float,
    high_padding_fraction_threshold: float,
) -> dict[str, Any]:
    sample_dir = root / "results" / "samples" / dataset / model / f"steps_{step}"
    samples_path = sample_dir / "samples.jsonl"
    metrics_path = root / "results" / "metrics" / dataset / model / f"steps_{step}" / "metrics.json"
    conversion_path = sample_dir / "conversion_metadata.json"
    metadata_path = sample_dir / "run_metadata.json"
    warnings: list[str] = []
    row: dict[str, Any] = {
        "dataset": dataset,
        "model": model,
        "steps": step,
        "sample_dir": str(sample_dir),
        "metrics_path": str(metrics_path),
        "sample_count": None,
        "metrics_sample_count": None,
        "generative_perplexity": None,
        "unigram_entropy": None,
        "self_bleu": None,
        "token_ids_source": None,
        "token_ids_reason": None,
        "command_sampling_steps": None,
        "command_batch_per_gpu": None,
    }

    if samples_path.is_file() and not samples_path.is_symlink():
        samples = _read_samples(samples_path, warnings)
        row.update(_sample_stats(samples))
    else:
        warnings.append("missing_samples")

    if metrics_path.is_file() and not metrics_path.is_symlink():
        metrics_document = _read_json(metrics_path, warnings, "metrics")
        if metrics_document is not None:
            metric_values = metrics_document.get("metrics")
            metric_values = metric_values if isinstance(metric_values, Mapping) else {}
            row["metrics_sample_count"] = metrics_document.get("sample_count")
            row["metrics_partial"] = metrics_document.get("partial")
            row["generative_perplexity"] = _metric(
                metric_values, "generative_perplexity"
            )
            row["unigram_entropy"] = _metric(metric_values, "unigram_entropy")
            row["self_bleu"] = _metric(metric_values, "self_bleu")
            recorded_sha = metrics_document.get("samples_sha256")
            if (
                isinstance(recorded_sha, str)
                and samples_path.is_file()
                and sha256_file(samples_path) != recorded_sha
            ):
                warnings.append("metrics_sample_sha_mismatch")
    else:
        warnings.append("missing_metrics")

    if conversion_path.is_file() and not conversion_path.is_symlink():
        conversion = _read_json(conversion_path, warnings, "conversion_metadata")
        if conversion is not None:
            row["token_ids_source"] = conversion.get("token_ids_source")
            transformation = conversion.get("token_ids_transformation")
            if isinstance(transformation, Mapping):
                row["token_ids_reason"] = transformation.get("reason")
    elif samples_path.is_file():
        warnings.append("missing_conversion_metadata")

    if metadata_path.is_file() and not metadata_path.is_symlink():
        metadata = _read_json(metadata_path, warnings, "run_metadata")
        if metadata is not None:
            row["run_status"] = metadata.get("status")
            command = metadata.get("command")
            row["command_sampling_steps"] = _command_override(
                command, "sampling.steps"
            )
            row["command_batch_per_gpu"] = _command_override(
                command, "sampling.batch_per_gpu"
            )
            if row["command_sampling_steps"] not in {None, str(step)}:
                warnings.append("command_step_mismatch")
    elif samples_path.is_file():
        warnings.append("missing_run_metadata")

    _add_quality_warnings(
        row,
        warnings,
        expected_sample_count=expected_sample_count,
        low_ppl_threshold=low_ppl_threshold,
        low_entropy_threshold=low_entropy_threshold,
        high_self_bleu_threshold=high_self_bleu_threshold,
        duplicate_unique_ratio_threshold=duplicate_unique_ratio_threshold,
        short_text_chars_threshold=short_text_chars_threshold,
        high_padding_fraction_threshold=high_padding_fraction_threshold,
    )
    row["warnings"] = list(dict.fromkeys(warnings))
    return row


def _ppl_nonmonotonic(rows: Sequence[Mapping[str, Any]]) -> bool:
    values = [
        row.get("generative_perplexity")
        for row in rows
        if type(row.get("generative_perplexity")) in {int, float}
    ]
    if len(values) < 3:
        return False
    nondecreasing = all(a <= b for a, b in zip(values, values[1:]))
    nonincreasing = all(a >= b for a, b in zip(values, values[1:]))
    return not nondecreasing and not nonincreasing


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    suspect_steps = [
        row["steps"]
        for row in rows
        if COLLAPSE_WARNINGS.intersection(row["warnings"])
        or (
            "retokenized_from_text" in row["warnings"]
            and row.get("generative_perplexity") is not None
            and row["generative_perplexity"] < 100.0
        )
    ]
    missing_steps = [
        row["steps"]
        for row in rows
        if "missing_samples" in row["warnings"] or "missing_metrics" in row["warnings"]
    ]
    retokenized_steps = [
        row["steps"] for row in rows if row.get("token_ids_source") == "retokenized"
    ]
    nonmonotonic = _ppl_nonmonotonic(rows)
    if suspect_steps:
        verdict = "low_ppl_points_show_collapse_signals"
    elif missing_steps:
        verdict = "missing_server_artifacts"
    elif nonmonotonic:
        verdict = "ppl_curve_is_nonmonotonic"
    else:
        verdict = "no_obvious_artifact"
    return {
        "verdict": verdict,
        "suspect_steps": suspect_steps,
        "missing_steps": missing_steps,
        "retokenized_steps": retokenized_steps,
        "ppl_curve_nonmonotonic": nonmonotonic,
    }


def diagnose_rdlm(
    root: Path | str,
    *,
    dataset: str = "lm1b",
    model: str = "rdlm",
    steps: Iterable[int] = DEFAULT_STEPS,
    expected_sample_count: int = 1024,
    low_ppl_threshold: float = 100.0,
    low_entropy_threshold: float = 3.8,
    high_self_bleu_threshold: float = 0.8,
    duplicate_unique_ratio_threshold: float = 0.9,
    short_text_chars_threshold: float = 80.0,
    high_padding_fraction_threshold: float = 0.25,
) -> dict[str, Any]:
    """Inspect already-produced RDLM artifacts and summarize suspicious signals."""

    root = Path(root).resolve()
    rows = [
        _row_for_step(
            root,
            dataset=dataset,
            model=model,
            step=step,
            expected_sample_count=expected_sample_count,
            low_ppl_threshold=low_ppl_threshold,
            low_entropy_threshold=low_entropy_threshold,
            high_self_bleu_threshold=high_self_bleu_threshold,
            duplicate_unique_ratio_threshold=duplicate_unique_ratio_threshold,
            short_text_chars_threshold=short_text_chars_threshold,
            high_padding_fraction_threshold=high_padding_fraction_threshold,
        )
        for step in steps
    ]
    return {
        "schema": SCHEMA,
        "root": str(root),
        "dataset": dataset,
        "model": model,
        "expected_sample_count": expected_sample_count,
        "summary": _summary(rows),
        "rows": rows,
    }


def _parse_steps(value: str) -> tuple[int, ...]:
    try:
        steps = tuple(int(part) for part in value.split(",") if part)
    except ValueError as error:
        raise argparse.ArgumentTypeError("steps must be comma-separated integers") from error
    if not steps or any(step <= 0 for step in steps):
        raise argparse.ArgumentTypeError("steps must contain positive integers")
    return steps


def _format(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def print_report(report: Mapping[str, Any]) -> None:
    rows = report["rows"]
    print(f"schema\t{report['schema']}")
    print(f"verdict\t{report['summary']['verdict']}")
    print(
        "steps\tsamples\tppl\tentropy\tself_bleu\tunique\tmedian_chars\t"
        "pad_frac\ttoken_source\twarnings"
    )
    for row in rows:
        print(
            "\t".join(
                [
                    str(row["steps"]),
                    _format(row.get("sample_count")),
                    _format(row.get("generative_perplexity")),
                    _format(row.get("unigram_entropy")),
                    _format(row.get("self_bleu")),
                    _format(row.get("unique_text_ratio")),
                    _format(row.get("median_text_chars")),
                    _format(row.get("pad_token_fraction")),
                    _format(row.get("token_ids_source")),
                    ",".join(row["warnings"]),
                ]
            )
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", default="lm1b")
    parser.add_argument("--model", default="rdlm")
    parser.add_argument("--steps", type=_parse_steps, default=DEFAULT_STEPS)
    parser.add_argument("--expected-sample-count", type=int, default=1024)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = diagnose_rdlm(
        args.root,
        dataset=args.dataset,
        model=args.model,
        steps=args.steps,
        expected_sample_count=args.expected_sample_count,
    )
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

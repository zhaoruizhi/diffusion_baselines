#!/usr/bin/env python
"""Print a compact LM1B Di4C summary from aggregate CSV outputs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable


MODELS = ("duo_di4c", "mdlm_di4c")
DATASET = "lm1b"
STEPS = (1, 2, 4, 8, 16, 32)
OUTPUT_COLUMNS = (
    "model",
    "dataset",
    "steps",
    "ppl",
    "entropy",
    "self_bleu",
    "seconds_per_sample",
)
FAILURE_COLUMNS = ("task_id", "model", "dataset", "steps", "reason")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: str, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def _matching(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    wanted = {(model, DATASET, str(step)) for model in MODELS for step in STEPS}
    return [
        row
        for row in rows
        if (row.get("model"), row.get("dataset"), row.get("steps")) in wanted
    ]


def _sort_key(row: dict[str, str]) -> tuple[int, int]:
    return MODELS.index(row["model"]), STEPS.index(int(row["steps"]))


def _result_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "model": row["model"],
        "dataset": row["dataset"],
        "steps": row["steps"],
        "ppl": _float(row["generative_perplexity"], 2),
        "entropy": _float(row["unigram_entropy"], 4),
        "self_bleu": _float(row["self_bleu"], 4),
        "seconds_per_sample": _float(row["generation_seconds_per_sample"], 6),
    }


def _write_rows(rows: list[dict[str, str]]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in sorted(rows, key=_sort_key):
        writer.writerow(_result_row(row))


def _write_failures(rows: list[dict[str, str]]) -> None:
    failures = sorted(_matching(rows), key=_sort_key)
    if not failures:
        return
    sys.stdout.write("\n# failures\n")
    writer = csv.DictWriter(sys.stdout, fieldnames=FAILURE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in failures:
        writer.writerow({column: row.get(column, "") for column in FAILURE_COLUMNS})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--summary-dir", type=Path)
    args = parser.parse_args(argv)

    root = args.root.resolve()
    summary = (args.summary_dir or root / "results" / "summary").resolve()
    result_rows = _matching(_read_csv(summary / "results_wide.csv"))
    failure_path = summary / "failures.csv"
    failure_rows = _read_csv(failure_path) if failure_path.exists() else []

    _write_rows(result_rows)
    _write_failures(failure_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

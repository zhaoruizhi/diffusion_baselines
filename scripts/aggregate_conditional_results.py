#!/usr/bin/env python
"""Publish strict or partial C64 conditional baseline summary tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from dlb.aggregate import IncompleteMatrixError
from dlb.conditional_aggregate import aggregate_conditional


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="defaults to ROOT/results/conditional/summary",
    )
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = (args.output_dir or root / "results/conditional/summary").resolve()
    try:
        report = aggregate_conditional(
            root,
            strict=not args.partial,
            partial=args.partial,
            output_dir=output,
        )
    except IncompleteMatrixError as error:
        print(str(error))
        print(f"failures={len(error.failures)}")
        return 1
    print(
        f"complete={str(report.complete).lower()} "
        f"rows={len(report.rows)} failures={len(report.failures)} "
        f"unsupported={len(report.unsupported)} output={output}"
    )
    return 0 if report.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())

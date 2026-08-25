#!/usr/bin/env python3
"""Verify deterministic C64 conditional prompt artifacts before GPU work."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", choices=("all", "lm1b", "owt"), default="all")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from dlb.conditional_prompts import load_protocol, verify_prompts

    protocol = load_protocol(args.config or root / "configs" / "conditional.yaml")
    datasets = ("lm1b", "owt") if args.dataset == "all" else (args.dataset,)
    for dataset_id in datasets:
        manifest = verify_prompts(root, dataset_id, protocol)
        print(f"OK {dataset_id} prompts={manifest.prompt_count} sha256={manifest.prompt_file_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

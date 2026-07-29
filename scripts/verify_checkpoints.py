#!/usr/bin/env python3
"""Verify downloaded checkpoint files against the observed SHA-256 lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from dlb.checkpoints import verify_checkpoint_lock

    lock_path = args.lock or root / "artifacts" / "checkpoint_lock.json"
    if not lock_path.is_file():
        print(f"ERROR missing checkpoint lock: {lock_path}", file=sys.stderr)
        return 2
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    manifest_path = args.manifest or root / "artifacts" / "checkpoints.yaml"
    report = verify_checkpoint_lock(root, lock, manifest_path=manifest_path)
    print(f"{str(report['manifest_status']).upper()} manifest")
    print(f"{str(report['resource_set_status']).upper()} resource-set")
    for resource_id, record in report["resources"].items():
        print(f"{str(record['status']).upper()} {resource_id}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

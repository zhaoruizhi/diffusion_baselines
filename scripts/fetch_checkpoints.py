#!/usr/bin/env python3
"""Enumerate or fetch pinned public checkpoint resources."""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--all-public", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from dlb.checkpoints import (
        fetch_all_resources,
        load_checkpoint_manifest,
        require_server_platform,
        validate_checkpoint_coverage,
    )
    from dlb.registry import load_registry

    manifest_path = (args.config or root / "artifacts" / "checkpoints.yaml").resolve()
    registry_path = (args.registry or root / "configs" / "experiments.yaml").resolve()
    manifest = load_checkpoint_manifest(manifest_path)
    registry = load_registry(registry_path)
    validate_checkpoint_coverage(registry, manifest)
    if args.dry_run:
        for resource_id, resource in manifest.resources.items():
            print(
                f"RESOURCE {resource_id} {resource.backend} {resource.provenance} "
                f"checkpoints/{resource.destination} teacher={resource.teacher_family}"
            )
        for recipe_id, recipe in manifest.recipes.items():
            print(
                f"RECIPE {recipe_id} {recipe.model}/{recipe.dataset} {recipe.provenance} "
                f"{recipe.output} teacher={recipe.teacher_family}"
            )
        return 0
    if not args.all_public:
        print("ERROR: pass --all-public to acknowledge all public checkpoint downloads", file=sys.stderr)
        return 2
    try:
        require_server_platform(platform.system())
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    lock = fetch_all_resources(root, manifest_path, manifest)
    for resource_id, record in lock["resources"].items():
        print(f"{str(record['status']).upper()} {resource_id} {record['destination']}")
    return 1 if any(record["status"] != "downloaded" for record in lock["resources"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())

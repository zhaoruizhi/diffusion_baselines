#!/usr/bin/env python3
"""Verify the locally checked-out official source repositories."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from dlb.io import atomic_json_write, sha256_file
from dlb.registry import load_registry


def git_output(source_dir: Path, *args: str) -> str:
    """Return stripped stdout for a Git command run inside *source_dir*."""

    return subprocess.run(
        ["git", "-C", str(source_dir), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def load_manifest(path: Path, required_sources: set[str]) -> dict[str, dict[str, str]]:
    """Load and validate the source manifest's shape and registry coverage."""

    with path.open(encoding="utf-8") as manifest_file:
        manifest: Any = yaml.safe_load(manifest_file)
    if not isinstance(manifest, dict) or set(manifest) != required_sources:
        raise ValueError("source manifest IDs must match registry sources")
    for name, source in manifest.items():
        if (
            not isinstance(source, dict)
            or set(source) != {"url", "commit"}
            or not isinstance(source["url"], str)
            or not isinstance(source["commit"], str)
            or len(source["commit"]) != 40
        ):
            raise ValueError(f"invalid manifest entry for {name}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "artifacts/sources.yaml"

    try:
        registry = load_registry(root / "configs/experiments.yaml")
        required_sources = {entry.source for entry in registry.models.values()}
        manifest = load_manifest(manifest_path, required_sources)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"ERROR manifest {error}", file=sys.stderr)
        return 1

    verified: dict[str, dict[str, str]] = {}
    failed = False
    for name in sorted(required_sources):
        source = manifest[name]
        source_dir = root / "upstreams" / name
        try:
            if not (source_dir / ".git").is_dir():
                raise ValueError("repository is missing")
            if git_output(source_dir, "remote", "get-url", "origin") != source["url"]:
                raise ValueError("origin does not match manifest")
            if git_output(source_dir, "rev-parse", "--abbrev-ref", "HEAD") != "HEAD":
                raise ValueError("HEAD is not detached")
            if git_output(source_dir, "rev-parse", "HEAD") != source["commit"]:
                raise ValueError("HEAD does not match manifest")
            if git_output(source_dir, "status", "--porcelain"):
                raise ValueError("repository has uncommitted changes")
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            print(f"ERROR {name} {error}", file=sys.stderr)
            failed = True
            continue

        verified[name] = {"url": source["url"], "commit": source["commit"]}
        print(f"OK {name} {source['commit']}")

    if failed:
        return 1

    atomic_json_write(
        root / "artifacts/source_lock.json",
        {
            "manifest_sha256": sha256_file(manifest_path),
            "sources": verified,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

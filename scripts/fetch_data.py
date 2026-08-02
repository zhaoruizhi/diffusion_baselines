#!/usr/bin/env python3
"""Download pinned datasets and model/tokenizer snapshots with cache resume."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-low-disk", action="store_true")
    return parser.parse_args()


def load_configuration(root: Path, config_path: Path | None) -> dict[str, object]:
    path = config_path or root / "artifacts" / "data.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def target_lines(configuration: dict[str, object]) -> list[str]:
    datasets = configuration["datasets"]
    models = configuration["models"]
    return [
        *[
            f"DATASET {name} {datasets[name]['repo_id']}@{datasets[name]['revision']}"
            for name in ("lm1b", "owt")
        ],
        *[
            f"MODEL {name}@{models[name]}"
            for name in ("bert-base-uncased", "gpt2", "gpt2-large")
        ],
    ]


def directory_size(path: Path) -> int:
    return sum(candidate.stat().st_size for candidate in path.rglob("*") if candidate.is_file())


def _relative_cache_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    configuration = load_configuration(root, args.config)
    for line in target_lines(configuration):
        print(line)
    if args.dry_run:
        return 0

    hf_home = root / "data" / "raw" / "huggingface"
    datasets_cache = hf_home / "datasets"
    hub_cache = hf_home / "hub"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)

    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    sys.path.insert(0, str(root / "src"))
    from dlb.data import disk_preflight_evidence, validate_download_manifest
    from dlb.io import atomic_json_write

    dataset_records: dict[str, object] = {}
    owt_disk_preflight = None
    for name in ("lm1b", "owt"):
        specification = configuration["datasets"][name]
        if name == "owt":
            free_bytes = shutil.disk_usage(root).free
            owt_disk_preflight = disk_preflight_evidence(
                free_bytes, allow_low_disk=args.allow_low_disk
            )
            if owt_disk_preflight["override_used"]:
                print(
                    f"WARNING low disk override: {free_bytes} bytes free, required 55 GiB",
                    file=sys.stderr,
                )
        if name == "lm1b":
            source_metadata_snapshot = snapshot_download(
                repo_id=specification["repo_id"],
                repo_type="dataset",
                revision=specification["revision"],
                cache_dir=hub_cache,
            )
            dataset_snapshot = snapshot_download(
                repo_id=specification["repo_id"],
                repo_type="dataset",
                revision=specification["parquet_revision"],
                cache_dir=hub_cache,
                allow_patterns="plain_text/**/*.parquet",
            )
            snapshot_path = Path(dataset_snapshot)
            loaded = load_dataset(
                "parquet",
                data_files={
                    "train": [
                        str(path)
                        for path in sorted(
                            (snapshot_path / "plain_text" / "train").glob("*.parquet")
                        )
                    ],
                    "test": [
                        str(path)
                        for path in sorted(
                            (snapshot_path / "plain_text" / "test").glob("*.parquet")
                        )
                    ],
                },
                cache_dir=str(datasets_cache),
                download_mode="reuse_dataset_if_exists",
            )
            archive_specification = configuration["lm1b_archive"]
            if len(loaded["train"]) != int(archive_specification["train_documents"]):
                raise RuntimeError("LM1B training document count differs from pinned metadata")
            if len(loaded["test"]) != int(archive_specification["test_documents"]):
                raise RuntimeError("LM1B test document count differs from pinned metadata")
        else:
            source_metadata_snapshot = None
            dataset_snapshot = None
            loaded = load_dataset(
                specification["repo_id"],
                specification["config"],
                revision=specification["revision"],
                cache_dir=str(datasets_cache),
                download_mode="reuse_dataset_if_exists",
            )
            if len(loaded["train"]) != int(specification["documents"]):
                raise RuntimeError("OpenWebText document count differs from pinned metadata")
        dataset_records[name] = {
            "repo_id": specification["repo_id"],
            "source_revision": specification["revision"],
            "cache_files": {
                split: [
                    _relative_cache_path(Path(entry["filename"]), root)
                    for entry in value.cache_files
                ]
                for split, value in loaded.items()
            },
            "split_rows": {split: len(value) for split, value in loaded.items()},
            "dataset_snapshot": (
                _relative_cache_path(Path(dataset_snapshot), root) if dataset_snapshot else None
            ),
            "source_metadata_snapshot": (
                _relative_cache_path(Path(source_metadata_snapshot), root)
                if source_metadata_snapshot
                else None
            ),
            "materialization_revision": specification.get("parquet_revision"),
        }
        print(f"DOWNLOADED DATASET {name} {dataset_records[name]['split_rows']}")

    model_records: dict[str, object] = {}
    for name in ("bert-base-uncased", "gpt2", "gpt2-large"):
        revision = configuration["models"][name]
        snapshot_path = Path(
            snapshot_download(
                repo_id=name,
                revision=revision,
                cache_dir=hub_cache,
            )
        )
        model_records[name] = {
            "repo_id": name,
            "revision": revision,
            "snapshot_path": snapshot_path.relative_to(root).as_posix(),
            "size_bytes": directory_size(snapshot_path),
        }
        print(f"DOWNLOADED MODEL {name} {snapshot_path}")

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "hf_home": hf_home.relative_to(root).as_posix(),
        "owt_disk_preflight": owt_disk_preflight,
        "datasets": dataset_records,
        "models": model_records,
    }
    for dataset_name in ("lm1b", "owt"):
        validate_download_manifest(configuration, manifest, dataset_name)
    atomic_json_write(root / "data" / "manifests" / "downloads.json", manifest)
    print(f"RAW CACHE BYTES {directory_size(hf_home)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

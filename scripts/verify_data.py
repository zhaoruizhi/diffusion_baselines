#!/usr/bin/env python3
"""Verify processed datasets against pinned integrity manifests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import yaml


OUTPUT_NAMES = {"lm1b": "lm1b-bert-128", "owt": "owt-gpt2-1024"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", choices=("all", "lm1b", "owt"), default="all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from dlb.data import build_processing_contract, verify_processed_dataset
    from dlb.io import atomic_json_write

    config_path = args.config or root / "artifacts" / "data.yaml"
    configuration = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    hf_home = root / "data" / "raw" / "huggingface"
    hub_cache = hf_home / "hub"
    os.environ["HF_HOME"] = str(hf_home)

    from datasets import load_from_disk
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    names = ("lm1b", "owt") if args.dataset == "all" else (args.dataset,)
    for name in names:
        contract = build_processing_contract(configuration, name)
        tokenizer_name = contract["tokenizer_id"]
        tokenizer_revision = contract["tokenizer_revision"]
        manifest_path = root / "data" / "manifests" / f"{name}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        snapshot_path = snapshot_download(
            repo_id=tokenizer_name,
            revision=tokenizer_revision,
            cache_dir=hub_cache,
            local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)
        output_dir = root / "data" / "processed" / OUTPUT_NAMES[name]
        dataset = load_from_disk(str(output_dir))
        verified = verify_processed_dataset(
            dataset=dataset,
            manifest=manifest,
            output_dir=output_dir,
            tokenizer=tokenizer,
            expected_contract=contract,
        )
        atomic_json_write(manifest_path, verified)
        print(
            f"OK {name} documents={verified['document_counts']} "
            f"packed={verified['packed_sequence_counts']} "
            f"checked={verified['verification']['checked_sequences']} "
            f"files={verified['verification']['files_checked']} verified=true"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

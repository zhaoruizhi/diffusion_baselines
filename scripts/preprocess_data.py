#!/usr/bin/env python3
"""Deterministically preprocess complete pinned LM1B and OpenWebText caches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys

import yaml


OUTPUT_NAMES = {"lm1b": "lm1b-bert-128", "owt": "owt-gpt2-1024"}
EXPECTED_VOCAB_SIZES = {"lm1b": 30_522, "owt": 50_257}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", choices=("all", "lm1b", "owt"), default="all")
    parser.add_argument("--batch-documents", type=int, default=1_000)
    return parser.parse_args()


def lm1b_detokenizer(text: str) -> str:
    """Match the normalization used by the FLM/MDLM LM1B loaders."""

    text = text.replace("http : / / ", "http://")
    text = text.replace("https : / / ", "https://")
    text = re.sub(r" '(\w+)", r"'\1", text)
    text = re.sub(r" (\w+) \. ", r" \1. ", text)
    text = re.sub(r" (\w+) \.$", r" \1.", text)
    text = text.replace(" ? ", "? ")
    text = re.sub(r" \?$", "?", text)
    text = text.replace(" ! ", "! ")
    text = re.sub(r" !$", "!", text)
    text = text.replace(" , ", ", ")
    text = text.replace(" : ", ": ")
    text = text.replace(" ; ", "; ")
    text = text.replace(" / ", "/")
    text = re.sub(r'" ([^"]+) "', r'"\1"', text)
    text = re.sub(r"' ([^']+) '", r"'\1'", text)
    text = re.sub(r"\( ([^()]+) \)", r"(\1)", text)
    text = re.sub(r"\[ ([^\[\]]+) \]", r"[\1]", text)
    text = text.replace("$ ", "$")
    return text.replace("£ ", "£")


def token_bounds(dataset_dict: object) -> tuple[int, int]:
    import pyarrow.compute as pc

    minimum = None
    maximum = None
    for split in dataset_dict.values():
        bounds = pc.min_max(pc.list_flatten(split.data.column("input_ids"))).as_py()
        minimum = bounds["min"] if minimum is None else min(minimum, bounds["min"])
        maximum = bounds["max"] if maximum is None else max(maximum, bounds["max"])
    return int(minimum), int(maximum)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from dlb.data import build_data_manifest, build_owt_split, preprocess_split
    from dlb.io import atomic_json_write

    config_path = args.config or root / "artifacts" / "data.yaml"
    configuration = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    downloads_path = root / "data" / "manifests" / "downloads.json"
    if not downloads_path.is_file():
        raise FileNotFoundError("run scripts/fetch_data.py before preprocessing")
    downloads = json.loads(downloads_path.read_text(encoding="utf-8"))

    hf_home = root / "data" / "raw" / "huggingface"
    datasets_cache = hf_home / "datasets"
    hub_cache = hf_home / "hub"
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)

    from datasets import DatasetDict, load_dataset, load_from_disk
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    names = ("lm1b", "owt") if args.dataset == "all" else (args.dataset,)
    for name in names:
        specification = configuration["datasets"][name]
        tokenizer_name = specification["tokenizer"]
        tokenizer_revision = configuration["models"][tokenizer_name]
        manifest_path = root / "data" / "manifests" / f"{name}.json"
        output_dir = root / "data" / "processed" / OUTPUT_NAMES[name]
        if output_dir.is_dir() and manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                existing.get("source_revision") == specification["revision"]
                and existing.get("tokenizer_revision") == tokenizer_revision
            ):
                print(f"SKIP {name} complete {len(load_from_disk(str(output_dir))['train'])} train sequences")
                continue
            raise RuntimeError(f"{name} output exists with a different pinned revision")

        snapshot_path = Path(
            snapshot_download(
                repo_id=tokenizer_name,
                revision=tokenizer_revision,
                cache_dir=hub_cache,
                local_files_only=True,
            )
        )
        tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)
        if tokenizer.vocab_size != EXPECTED_VOCAB_SIZES[name]:
            raise RuntimeError(
                f"{tokenizer_name} vocabulary is {tokenizer.vocab_size}, "
                f"expected {EXPECTED_VOCAB_SIZES[name]}"
            )
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        if name == "lm1b":
            bos_id = tokenizer.cls_token_id
            eos_id = tokenizer.sep_token_id
        if bos_id is None or eos_id is None:
            raise RuntimeError(f"{tokenizer_name} does not define required BOS/EOS semantics")

        if name == "lm1b":
            snapshot_path = root / downloads["datasets"]["lm1b"]["dataset_snapshot"]
            raw = load_dataset(
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
            sources = {"train": raw["train"], "validation": raw["test"]}
        else:
            raw = load_dataset(
                specification["repo_id"],
                specification["config"],
                revision=specification["revision"],
                cache_dir=str(datasets_cache),
                download_mode="reuse_dataset_if_exists",
            )
            source = raw["train"]
            if len(source) != int(specification["documents"]):
                raise RuntimeError("OpenWebText document count differs from pinned metadata")
            expressions = build_owt_split(len(source))
            if expressions.train != specification["splits"]["train"]:
                raise RuntimeError("OpenWebText training split expression changed")
            boundary = len(source) - 100_000
            sources = {
                "train": source.select(range(boundary)),
                "validation": source.select(range(boundary, len(source))),
            }

        builder_cache = root / "data" / "processed" / ".builder-cache" / name
        processed = {}
        for split_name in ("train", "validation"):
            print(f"PREPROCESS {name} {split_name} documents={len(sources[split_name])}")
            processed[split_name] = preprocess_split(
                sources[split_name],
                tokenizer=tokenizer,
                length=int(specification["sequence_length"]),
                bos_id=int(bos_id),
                eos_id=int(eos_id),
                cache_dir=builder_cache / split_name,
                batch_documents=args.batch_documents,
                detokenizer=lm1b_detokenizer if name == "lm1b" else None,
            )
        processed_dict = DatasetDict(processed)

        partial_dir = output_dir.with_name(output_dir.name + ".partial")
        if partial_dir.exists():
            shutil.rmtree(partial_dir)
        partial_dir.parent.mkdir(parents=True, exist_ok=True)
        processed_dict.save_to_disk(str(partial_dir))
        os.replace(partial_dir, output_dir)

        minimum, maximum = token_bounds(processed_dict)
        manifest = build_data_manifest(
            dataset=name,
            dataset_id=specification["repo_id"],
            source_revision=specification["revision"],
            tokenizer_id=tokenizer_name,
            tokenizer_revision=tokenizer_revision,
            sequence_length=int(specification["sequence_length"]),
            split_expression=specification["splits"],
            document_counts={split: len(value) for split, value in sources.items()},
            packed_sequence_counts={split: len(value) for split, value in processed.items()},
            vocab_size=int(tokenizer.vocab_size),
            min_token_id=minimum,
            max_token_id=maximum,
            output_dir=output_dir,
            root=root,
        )
        if name == "lm1b":
            manifest["source_materialization_revision"] = specification["parquet_revision"]
        atomic_json_write(manifest_path, manifest)
        print(
            f"WROTE {name} documents={manifest['document_counts']} "
            f"packed={manifest['packed_sequence_counts']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

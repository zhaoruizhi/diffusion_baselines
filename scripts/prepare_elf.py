#!/usr/bin/env python3
"""Server-only pinned downloads and text-only evaluation inputs for ELF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from elf_common import ROOT, SOURCE_COMMIT, asset, read_jsonl, require_server, sha256, source_path, write_json


def export_rows(path, rows, provenance):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    count = 0
    with temporary.open("w") as handle:
        for count, row in enumerate(rows, 1):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
    write_json(path.with_suffix(".manifest.json"), {
        "schema": "elf-input-v1", "count": count, "sha256": sha256(path), **provenance})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["source", "assets", "data", "prefix"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    require_server()
    root = args.root.resolve()
    lock = json.loads((root / "artifacts/elf_lock.json").read_text())
    if args.stage == "source":
        dst = root / "upstreams/elf"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--single-branch", "--branch", "pytorch_elf",
                            lock["source"]["url"], str(dst)], check=True)
            subprocess.run(["git", "-C", str(dst), "checkout", "--detach", SOURCE_COMMIT], check=True)
        print(source_path(root))
        return
    if args.stage == "assets":
        from huggingface_hub import snapshot_download
        records = {}
        for name, record in lock["assets"].items():
            dst = root / "data/elf/snapshots" / name / record["revision"]
            snapshot_download(repo_id=record["repo_id"], revision=record["revision"],
                              repo_type=record["repo_type"], allow_patterns=record["patterns"], local_dir=dst)
            files = {str(p.relative_to(dst)): sha256(p) for p in sorted(dst.rglob("*"))
                     if p.is_file() and ".cache" not in p.parts}
            if not files:
                raise ValueError(f"No downloaded files for {name}")
            records[name] = {**record, "path": str(dst.relative_to(root)), "sha256": files}
        write_json(root / "data/elf/assets.json", {"assets": records,
                   "source_commit": SOURCE_COMMIT, "lock_sha256": sha256(root / "artifacts/elf_lock.json")})
        return
    from datasets import load_dataset, load_from_disk
    from transformers import AutoTokenizer
    if args.stage == "data":
        for task in ("wmt14", "xsum"):
            official_name = f"{task}-official-validation"
            official = load_from_disk(str(asset(root, official_name)))
            # Author Arrow release includes raw input/target. Fail instead of
            # inventing references by decoding truncated model tokens.
            if not {"input", "target"}.issubset(official.column_names):
                raise ValueError(f"{official_name}: missing raw input/target columns")
            export_rows(root / f"data/elf/{official_name}.jsonl",
                        ({"id": i, "input": r["input"], "output": r["target"],
                          "condition_input_ids": list(r["condition_input_ids"])}
                         for i, r in enumerate(official)),
                        {"task": task, "split": "official-validation", "asset": lock["assets"][official_name]})
            directory = asset(root, f"{task}-raw")
            for split in ("validation", "test"):
                files = sorted(str(p) for p in directory.rglob(f"{split}-*.parquet"))
                if not files:
                    raise FileNotFoundError(f"Missing {task} {split} parquet")
                ds = load_dataset("parquet", data_files={split: files}, split=split)
                def rows():
                    for i, r in enumerate(ds):
                        src, target = ((r["translation"]["de"], r["translation"]["en"])
                                       if task == "wmt14" else (r["document"], r["summary"]))
                        yield {"id": i, "input": src, "output": target}
                export_rows(root / f"data/elf/{task}-{split}.jsonl", rows(),
                            {"task": task, "split": split, "asset": lock["assets"][f"{task}-raw"]})
        return
    # Same held-out rows and literal prefix/reference text as the C64 benchmark;
    # T5 re-encoding is a separate protocol, never reinterpret GPT-2 IDs as T5.
    sys.path.insert(0, str(root / "src"))
    from dlb.conditional_prompts import load_protocol, verify_prompts
    verify_prompts(root, "owt", load_protocol(root / "configs/conditional.yaml"))
    prompts = root / "data/conditional/owt-c64/prompts.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(str(asset(root, "gpt2")), local_files_only=True)
    export_rows(root / "data/elf/owt-prefix.jsonl",
                ({"id": r["prompt_id"], "input": tokenizer.decode(r["prefix_token_ids"], skip_special_tokens=False),
                  "output": tokenizer.decode(r["reference_token_ids"], skip_special_tokens=False),
                  "source_index": r["source_index"]} for r in read_jsonl(prompts)),
                {"task": "owt-prefix", "protocol": "c64_text_t5_v1", "source_sha256": sha256(prompts)})


if __name__ == "__main__":
    main()

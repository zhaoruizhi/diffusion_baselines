#!/usr/bin/env python3
"""Server-only pinned downloads and text-only evaluation inputs for ELF."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from email.utils import parsedate_to_datetime

from elf_common import ROOT, SOURCE_COMMIT, asset, read_jsonl, require_server, sha256, source_path, write_json


def retryable_response(error):
    """Find HTTP failures even when Hub 0.36 wraps them in a cache error."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        response = getattr(error, "response", None)
        # requests.Response is falsey for 4xx/5xx, so do not use `if response`.
        if response is not None:
            if response.status_code in (429, 500, 502, 503, 504):
                return response
            return None  # Authentication/revision errors need a fix, not retries.
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return None


def retry_delay(response, attempt):
    """Honor server cooldown headers, falling back to bounded exponential wait."""
    delay = float(min(60 * 2 ** min(attempt - 1, 3), 300))
    headers = {str(k).lower(): v for k, v in response.headers.items()}
    retry_after = headers.get("retry-after", "")
    try:
        requested = float(retry_after)
    except (TypeError, ValueError):
        try:
            requested = parsedate_to_datetime(retry_after).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            requested = 0
    if math.isfinite(requested):
        delay = max(delay, requested)
    for seconds in re.findall(r'(?:^|[;,])\s*t=(\d+)', headers.get("ratelimit", "")):
        delay = max(delay, float(seconds) + 1)
    return delay


def download_snapshot(download, *, attempts=6, sleep=time.sleep, **kwargs):
    """Retry only temporary Hub HTTP failures; keep pinned revision and cache."""
    if attempts < 1:
        raise ValueError("download attempts must be positive")
    for attempt in range(1, attempts + 1):
        try:
            return download(**kwargs)
        except Exception as error:
            response = retryable_response(error)
            if response is None:
                raise
            if attempt == attempts:
                raise RuntimeError(
                    f"HF download failed after {attempts} attempts (HTTP {response.status_code}): "
                    f"{kwargs['repo_id']}. Local download files were retained. "
                    "Retry `python scripts/prepare_elf.py assets` later; do not run the data stage yet."
                ) from error
            seconds = retry_delay(response, attempt)
            print(f"HF HTTP {response.status_code}: {kwargs['repo_id']}; "
                  f"waiting {seconds:.0f}s before attempt {attempt + 1}/{attempts}", flush=True)
            sleep(seconds)


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
    parser.add_argument("--download-workers", type=int, default=1,
                        help="Concurrent snapshot file downloads (default: 1)")
    parser.add_argument("--download-attempts", type=int, default=6,
                        help="Total attempts per snapshot for temporary Hub HTTP errors")
    args = parser.parse_args()
    if args.download_workers < 1 or args.download_attempts < 1:
        parser.error("download-workers and download-attempts must be positive")
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
            print(f"Preparing {name}: {record['repo_id']} @ {record['revision']}", flush=True)
            download_snapshot(snapshot_download, attempts=args.download_attempts,
                              repo_id=record["repo_id"], revision=record["revision"],
                              repo_type=record["repo_type"], allow_patterns=record["patterns"],
                              local_dir=dst, max_workers=args.download_workers)
            files = {str(p.relative_to(dst)): sha256(p) for p in sorted(dst.rglob("*"))
                     if p.is_file() and ".cache" not in p.parts}
            if not files:
                raise ValueError(f"No downloaded files for {name}")
            records[name] = {**record, "path": str(dst.relative_to(root)), "sha256": files}
        write_json(root / "data/elf/assets.json", {"assets": records,
                   "source_commit": SOURCE_COMMIT, "lock_sha256": sha256(root / "artifacts/elf_lock.json")})
        print(f"OK: all {len(records)} ELF assets verified; wrote data/elf/assets.json", flush=True)
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

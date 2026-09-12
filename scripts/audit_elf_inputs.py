#!/usr/bin/env python3
"""Offline CPU audit of raw versus released ELF conditional validation inputs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from elf_common import (ROOT, asset, condition_token_ids, offline, read_jsonl,
                        require_server, sha256, write_json)


def audit_rows(raw, official, tokenizer, limit):
    def inspect(rows):
        token_rows = [condition_token_ids(row, tokenizer) for row in rows]
        return token_rows, {
            "rows": len(rows),
            "blank_text_ids": [row["id"] for row in rows if not row["input"].strip()],
            "empty_condition_ids": [row["id"] for row, ids in zip(rows, token_rows) if not ids],
            "truncated_sources": sum(len(ids) > limit for ids in token_rows),
            "condition_length_mean_before_cap": sum(map(len, token_rows)) / len(rows) if rows else None,
        }

    raw_ids, raw_stats = inspect(raw)
    official_ids, official_stats = inspect(official)
    by_pair = defaultdict(list)
    for row, ids in zip(official, official_ids):
        by_pair[(row["input"], row["output"])].append((row, ids))
    matches = Counter()
    examples = []
    for row, ids in zip(raw, raw_ids):
        candidates = by_pair.get((row["input"], row["output"]), [])
        if len(candidates) != 1:
            matches["unmatched_text_pair" if not candidates else "ambiguous_text_pair"] += 1
            continue
        author, expected = candidates[0]
        matches["unique_text_pair"] += 1
        matches["exact_ids"] += int(ids == expected)
        matches["exact_ids_after_source_cap"] += int(ids[:limit] == expected[:limit])
        matches["raw_plus_eos_equals_official"] += int(ids + [tokenizer.eos_token_id] == expected)
        if ids[:limit] != expected[:limit] and len(examples) < 5:
            examples.append({
                "raw_id": row["id"], "official_id": author["id"],
                "source_excerpt": row["input"][:160],
                "raw_length": len(ids), "official_length": len(expected),
                "raw_first_ids": ids[:16], "raw_last_ids": ids[-16:],
                "official_first_ids": expected[:16], "official_last_ids": expected[-16:],
                "official_condition_decoded_excerpt": tokenizer.decode(expected[:limit], skip_special_tokens=False)[:240],
            })
    return {"source_cap": limit, "raw": raw_stats, "official": official_stats,
            "comparison_by_exact_source_and_reference_text_not_row_number": dict(matches),
            "mismatch_examples": examples,
            "official_conditions_nonempty": bool(official) and not official_stats["empty_condition_ids"]}


def load_checked(path, task):
    meta = json.loads(path.with_suffix(".manifest.json").read_text())
    rows = read_jsonl(path)
    if (meta["sha256"] != sha256(path) or meta["task"] != task or meta["count"] != len(rows)
            or len({row["id"] for row in rows}) != len(rows)):
        raise ValueError(f"Input manifest/count/IDs mismatch: {path}")
    return rows, {"path": str(path), "manifest": meta}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_server()
    offline()
    from transformers import AutoTokenizer
    root = args.root.resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(asset(root, "t5")), local_files_only=True)
    report = {"schema": "elf-condition-audit-v1", "tasks": {},
              "auditor_sha256": sha256(Path(__file__)),
              "assets_manifest_sha256": sha256(root / "data/elf/assets.json")}
    for task, limit in (("wmt14", 64), ("xsum", 1024)):
        raw, raw_binding = load_checked(root / f"data/elf/{task}-validation.jsonl", task)
        official, official_binding = load_checked(root / f"data/elf/{task}-official-validation.jsonl", task)
        print(f"Auditing {task}: raw={len(raw)}, official={len(official)}; no GPU inference", flush=True)
        result = audit_rows(raw, official, tokenizer, limit)
        result["inputs"] = {"raw": raw_binding, "official": official_binding}
        report["tasks"][task] = result
        print(json.dumps({k: v for k, v in result.items() if k != "inputs"}, indent=2, ensure_ascii=False), flush=True)
    write_json(args.output, report)
    print(f"Saved {args.output}. Inputs unchanged; no rows removed or replacement tokens inserted.")


if __name__ == "__main__":
    main()

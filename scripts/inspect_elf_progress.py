#!/usr/bin/env python3
"""Read-only ELF quality inventory; verify saved artifacts before reporting completion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from elf_common import ROOT, read_jsonl, sha256, write_json
from evaluate_elf import validate_records

OWT_MAIN_STEPS = (1, 2, 4, 8, 16, 32, 1024)


def inspect_run(path):
    row = {"run": str(path), "status": "incomplete_generation"}
    try:
        request = json.loads((path / "request.json").read_text())
        row.update(task=request["task"], steps=request["steps"], seed=request["seed"],
                   compile=request["compile"],
                   split=(request.get("input") or {}).get("split", "unconditional"),
                   protocol=request.get("protocol"),
                   condition_policy=request.get("condition_tokenization_policy"))
        gp = path / "generation.json"
        if not gp.exists():
            return row
        gen = json.loads(gp.read_text())
        for key in ("task", "steps", "seed", "compile"):
            if gen[key] != request[key]:
                raise ValueError(f"request/generation mismatch: {key}")
        sample_hash = sha256(path / "samples.jsonl")
        if gen["samples_sha256"] != sample_hash:
            raise ValueError("samples hash mismatch")
        validate_records(read_jsonl(path / "samples.jsonl"), gen)
        row.update(sample_count=gen["sample_count"], prompt_count=gen["prompt_count"],
                   status="needs_evaluation")
        mp = path / "metrics.json"
        if mp.exists():
            metric = json.loads(mp.read_text())
            if (metric["generation_manifest_sha256"] != sha256(gp)
                    or metric["samples_sha256"] != sample_hash
                    or metric["task"] != gen["task"]
                    or metric["sample_count"] != gen["prompt_count"]):
                raise ValueError("metrics provenance/count mismatch")
            if type(metric.get("valid")) is not bool:
                raise ValueError("missing boolean metric validity")
            row.update(status="quality_valid" if metric["valid"] else "quality_invalid",
                       metrics=metric["metrics"], reason=metric.get("reason"))
        row["owt_main_grid"] = (
            row["task"] == "owt" and row["steps"] in OWT_MAIN_STEPS
            and row["prompt_count"] == 1024 and not gen.get("diversity")
            and row["seed"] == 42 and not row["compile"]
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        row.update(status="invalid_artifact", reason=str(error))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted((args.root / "results").glob("elf*/quality/*/*/steps_*/seed_*/compile_*"))
    rows = [inspect_run(path) for path in paths if path.is_dir()]
    for row in rows:
        metrics = row.get("metrics", {})
        scores = {key: metrics[key] for key in ("bleu", "rouge1", "rouge2", "rougeL", "entropy_gpt2_nats") if key in metrics}
        for key in ("generative_ppl", "conditional_ppl"):
            if key in metrics:
                scores[key] = metrics[key].get("perplexity")
        print(json.dumps({k: v for k, v in row.items() if k != "metrics"} | {"scores": scores}, ensure_ascii=False))
    present = {row["steps"] for row in rows if row.get("owt_main_grid") and row["status"] == "quality_valid"}
    missing = [step for step in OWT_MAIN_STEPS if step not in present]
    print(f"OWT primary grid {list(OWT_MAIN_STEPS)}; steps without a verified valid result: {missing}")
    print("Old/no-EOS conditional runs and auxiliary steps remain listed; do not merge protocols. No GPU work performed.")
    if args.output:
        write_json(args.output, {"schema": "elf-progress-v1", "runs": rows,
                                "owt_steps_without_valid_result": missing})


if __name__ == "__main__":
    main()

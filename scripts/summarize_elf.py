#!/usr/bin/env python3
"""Inventory ELF results without treating missing metrics/timing as successes."""
import argparse
import csv
import json
from pathlib import Path

from elf_common import ROOT, sha256

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--results", type=Path, default=ROOT / "results/elf")
p.add_argument("--output", type=Path, default=ROOT / "results/elf/summary.csv")
args = p.parse_args()
records = []
for path in sorted(args.results.rglob("request.json")):
    request = json.loads(path.read_text())
    row = {"run": str(path.parent), "task": request["task"], "steps": request["steps"],
           "seed": request["seed"], "compile": request["compile"],
           "sampling": json.dumps(request["sampling"], sort_keys=True),
           "split": (request.get("input") or {}).get("split", ""),
           "status": "incomplete"}
    for key in ("checkpoint_sha256", "source_commit", "gpu_name", "precision", "model_canvas_t5_tokens",
                "elf_forward_calls_per_sample", "protocol", "batch_size", "condition_tokenization_policy"):
        row[key] = request.get(key, "")
    row["input_sha256"] = (request.get("input") or {}).get("sha256", "")
    metrics_path, timing_path = path.parent / "metrics.json", path.parent / "timing.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        row["status"] = "quality_valid" if m.get("valid") else "quality_invalid"
        if m["samples_sha256"] != sha256(path.parent / "samples.jsonl"):
            raise ValueError(f"Samples changed: {path.parent}")
        if m["generation_manifest_sha256"] != sha256(path.parent / "generation.json"):
            raise ValueError(f"Generation manifest changed: {path.parent}")
        for k, v in m["metrics"].items():
            if isinstance(v, (float, int)):
                row[k] = v
            elif isinstance(v, dict) and "perplexity" in v:
                row[k] = v["perplexity"]
    if timing_path.exists():
        t = json.loads(timing_path.read_text())
        row["status"] = "timing_only"
        for k, v in t["results"].items():
            row[k + "_seconds"] = v["seconds_per_sample"]
            row[k + "_std_seconds"] = v["standard_deviation_seconds"]
    records.append(row)
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w") as f:
    writer = csv.DictWriter(f, fieldnames=sorted({k for row in records for k in row}))
    writer.writeheader()
    writer.writerows(records)
print(f"Wrote {len(records)} run records to {args.output}; inventory only, not a completeness certificate")

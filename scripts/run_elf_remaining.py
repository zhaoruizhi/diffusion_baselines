#!/usr/bin/env python3
"""Resume the ELF quality matrix on independent single-GPU workers (no timing)."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

from elf_common import ROOT, SOURCE_COMMIT, default_sampling, manifest, read_jsonl, sha256, write_json
from inspect_elf_progress import inspect_run, OWT_MAIN_STEPS

EOS_POLICY = "released_ids_or_raw_plus_eos_then_source_cap_v2"
TASKS = ("owt", "owt-prefix", "wmt14", "xsum")


@dataclass(frozen=True)
class Cell:
    task: str
    split: str
    steps: int
    count: int
    input_path: str | None = None
    input_sha256: str | None = None

    @property
    def key(self):
        return f"{self.task}/{self.split}/steps_{self.steps}/seed_42/compile_0"


def build_matrix(root, tasks):
    cells, blocked = [], []
    for task in tasks:
        if task == "owt":
            cells.extend(Cell(task, "unconditional", s, 1024) for s in OWT_MAIN_STEPS)
            continue
        splits = (("c64-text-t5", OWT_MAIN_STEPS, 1024),) if task == "owt-prefix" else (
            ("validation", (1, 2, 4, 8, 16, 32, 64), 3000 if task == "wmt14" else 11332),
            ("test", (32, 64), 3003 if task == "wmt14" else 11334),
        )
        for split, steps, expected_count in splits:
            path = root / "data/elf" / ("owt-prefix.jsonl" if task == "owt-prefix" else f"{task}-{split}.jsonl")
            try:
                meta = json.loads(path.with_suffix(".manifest.json").read_text())
                digest = sha256(path)
                rows = read_jsonl(path)
                if (meta["task"] != task or meta["sha256"] != digest or meta["count"] != expected_count
                        or len(rows) != expected_count or len({r["id"] for r in rows}) != expected_count):
                    raise ValueError("input task/count/hash/IDs mismatch")
                if task == "owt-prefix":
                    if meta.get("protocol") != "c64_text_t5_v1":
                        raise ValueError("prefix protocol mismatch")
                    source = root / "data/conditional/owt-c64/prompts.jsonl"
                    if meta.get("source_sha256") != sha256(source):
                        raise ValueError("prefix source changed; prepare prefix again")
                elif meta.get("split") != split:
                    raise ValueError("input split mismatch")
            except (OSError, KeyError, TypeError, ValueError) as error:
                blocked.append({"task": task, "split": split, "status": "blocked_input", "reason": str(error)})
                continue
            cells.extend(Cell(task, split, s, expected_count, str(path), digest) for s in steps)
    return sorted(cells, key=lambda c: (c.steps, c.task, c.split)), blocked


def matches_request(request, cell, asset_hash, assets):
    checkpoint_task = "owt" if cell.task == "owt-prefix" else cell.task
    record = assets[checkpoint_task]
    expected = {
        "task": cell.task, "steps": cell.steps, "seed": 42, "compile": False, "batch_size": 8,
        "source_commit": SOURCE_COMMIT, "assets_manifest_sha256": asset_hash,
        "checkpoint_task": checkpoint_task, "checkpoint_selection": "ema_params1",
        "checkpoint_sha256": record["sha256"][record["checkpoint"]],
        "sampling": default_sampling(cell.task, cell.steps),
        "model_canvas_t5_tokens": 1088 if cell.task == "xsum" else 128 if cell.task == "wmt14" else 1024,
        "protocol": "c64_text_t5_v1" if cell.task == "owt-prefix" else "elf_native_v1",
    }
    if any(request.get(key) != value for key, value in expected.items()):
        return False
    if cell.task in ("wmt14", "xsum") and request.get("condition_tokenization_policy") != EOS_POLICY:
        return False
    return (request.get("input") or {}).get("sha256") == cell.input_sha256


def reusable_result(root, cell, asset_hash, assets):
    candidates, rejected = [], []
    for path in sorted((root / "results").glob(f"elf*/quality/{cell.key}")):
        try:
            request = json.loads((path / "request.json").read_text())
            if not matches_request(request, cell, asset_hash, assets):
                continue
            row = inspect_run(path)
            if row["status"] not in ("quality_valid", "quality_invalid", "needs_evaluation"):
                rejected.append({"run": str(path), "reason": row.get("reason", row["status"])})
                continue
            gen = json.loads((path / "generation.json").read_text())
            keys = ("sampling", "input", "assets_manifest_sha256", "checkpoint_sha256", "source_commit",
                    "condition_tokenization_policy", "batch_size", "protocol")
            if any(gen.get(k) != request.get(k) for k in keys):
                raise ValueError("request/generation provenance differs")
            diversity = cell.task == "owt-prefix"
            total = cell.count + (min(256, cell.count) * 4 if diversity else 0)
            if gen["prompt_count"] != cell.count or gen["sample_count"] != total or bool(gen.get("diversity")) != diversity:
                continue  # e.g. 1000-row sanity is not a 1024-row primary run.
            if row["status"] == "quality_invalid":
                reason = row.get("reason") or ""
                if not reason.startswith("At least one completion has fewer than two GPT-2 tokens;"):
                    rejected.append({"run": str(path), "reason": "unrecognized invalid-quality reason"})
                    continue
            candidates.append(row)
        except (OSError, ValueError, KeyError, TypeError) as error:
            rejected.append({"run": str(path), "reason": str(error)})
    # A valid score and a recorded invalid score for one exact cell conflict;
    # never silently choose the better run.
    completed = [r for r in candidates if r["status"] != "needs_evaluation"]
    if len({r["status"] for r in completed}) > 1:
        raise ValueError(f"Conflicting completed results for {cell.key}; inspect runs manually")
    return (completed or candidates or [None])[0], rejected


def fresh_path(output_root, cell):
    for attempt in range(10000):
        base = output_root if attempt == 0 else output_root.with_name(output_root.name + f"-retry{attempt}")
        path = base / "quality" / cell.key
        if not path.exists():
            return path
    raise RuntimeError("Too many existing retry directories")


def commands(root, python, cell, path, generate):
    result = []
    if generate:
        cmd = [python, "-B", str(root / "scripts/run_elf.py"), "generate", "--root", str(root),
               "--task", cell.task, "--steps", str(cell.steps), "--seed", "42", "--batch-size", "8",
               "--num-samples", str(cell.count), "--output", str(path)]
        if cell.input_path:
            cmd += ["--input", cell.input_path]
        if cell.task == "owt-prefix":
            cmd += ["--diversity"]
        result.append(("generate", cmd))
    result.append(("evaluate", [python, "-B", str(root / "scripts/evaluate_elf.py"), "--root", str(root),
                                "--run", str(path), "--batch-size", "8"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--plan", action="store_true", help="Read-only preview; no GPU queries, writes or inference")
    args = parser.parse_args()
    root = args.root.resolve()
    output_root = (args.output_root or root / "results/elf-gpu01").resolve()
    if output_root.parent != root / "results" or not output_root.name.startswith("elf"):
        parser.error("output-root must be an elf* directory directly under ROOT/results")
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("duplicate tasks")
    assets = manifest(root)["assets"]
    asset_hash = sha256(root / "data/elf/assets.json")
    cells, blocked = build_matrix(root, args.tasks)
    jobs, reports = [], []
    for cell in cells:
        existing, rejected = reusable_result(root, cell, asset_hash, assets)
        action = ("skip" if existing["status"] != "needs_evaluation" else "evaluate") if existing else "generate+evaluate"
        path = Path(existing["run"]) if existing else fresh_path(output_root, cell)
        row = {"cell": asdict(cell), "action": action, "run": str(path), "rejected_candidates": rejected,
               "status": existing["status"] if action == "skip" else "pending"}
        reports.append(row)
        print(f"{action:18} {cell.key} -> {path}", flush=True)
        if action != "skip":
            jobs.append((cell, row))
    for row in blocked:
        print(f"BLOCKED {row['task']}/{row['split']}: {row['reason']}", flush=True)
    print(f"Plan: {len(cells)} cells, {len(jobs)} queued, {len(blocked)} blocked input groups; no timing.", flush=True)
    if args.plan:
        return 2 if blocked else 0
    if sys.platform != "linux":
        parser.error("GPU execution is server-only; use --plan for a read-only preview")
    if not jobs:
        print("Nothing to run. Existing invalid-quality results remain invalid, not successful PPL points.")
        return 2 if blocked else 0

    import fcntl
    from contextlib import ExitStack
    with ExitStack() as stack:
        gpu_records = []
        lock_dir = root / "results/elf-queue-locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        for gpu in args.gpus:
            line = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=uuid,name", "--format=csv,noheader"], text=True).strip()
            uuid, name = [s.strip() for s in line.split(",", 1)]
            if not uuid.startswith("GPU-") or any(r["uuid"] == uuid for r in gpu_records):
                raise ValueError("GPU selections must resolve to distinct physical GPUs")
            handle = stack.enter_context((lock_dir / f"{uuid}.lock").open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            gpu_records.append({"physical_selection": gpu, "uuid": uuid, "name": name})
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"
        log_dir = output_root / "logs" / stamp
        log_dir.mkdir(parents=True)
        report_path = log_dir / "queue.json"
        report = {"schema": "elf-quality-queue-v1", "gpus": gpu_records, "runs": reports, "blocked": blocked,
                  "assets_manifest_sha256": asset_hash, "timing": False}
        mutex, stop = threading.Lock(), threading.Event()
        processes = {}
        work = queue.Queue()
        for job in jobs:
            work.put(job)

        def publish():
            write_json(report_path, report)

        def worker(gpu):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"], PYTHONDONTWRITEBYTECODE="1",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled")
            while not stop.is_set():
                try:
                    cell, row = work.get_nowait()
                except queue.Empty:
                    break
                path = Path(row["run"])
                with mutex:
                    row.update(status="running", gpu=gpu["uuid"])
                    publish()
                    print(f"START GPU {gpu['physical_selection']} {cell.key}", flush=True)
                try:
                    for stage, command in commands(root, sys.executable, cell, path, row["action"] == "generate+evaluate"):
                        if stop.is_set():
                            raise RuntimeError("queue interrupted")
                        log = log_dir / (cell.key.replace("/", "__") + f"__{stage}.log")
                        with log.open("w") as stream:
                            stream.write(json.dumps(command) + "\n")
                            stream.flush()
                            with mutex:
                                if stop.is_set():
                                    raise RuntimeError("queue interrupted")
                                proc = subprocess.Popen(command, cwd=root, env=env, stdout=stream,
                                                        stderr=subprocess.STDOUT, start_new_session=True)
                                processes[gpu["uuid"]] = proc
                            code = proc.wait()
                            with mutex:
                                processes.pop(gpu["uuid"], None)
                        with mutex:
                            row[stage + "_log"] = str(log)
                        if code != 0:
                            checked, _ = reusable_result(root, cell, asset_hash, assets)
                            if stage == "evaluate" and checked and checked["run"] == str(path) and checked["status"] == "quality_invalid":
                                break
                            raise RuntimeError(f"{stage} exit={code}; see {log}")
                    checked, _ = reusable_result(root, cell, asset_hash, assets)
                    if not checked or checked["run"] != str(path) or checked["status"] not in ("quality_valid", "quality_invalid"):
                        raise RuntimeError("finished process did not publish a verified quality result")
                    with mutex:
                        row.update(status=checked["status"], reason=checked.get("reason"))
                except Exception as error:
                    with mutex:
                        row.update(status="failed", reason=str(error))
                finally:
                    with mutex:
                        publish()
                        print(f"DONE  GPU {gpu['physical_selection']} {cell.key}: {row['status']}", flush=True)
                    work.task_done()

        publish()
        print(f"Logs and live queue report: {log_dir}", flush=True)
        pool = ThreadPoolExecutor(max_workers=len(gpu_records))
        futures = [pool.submit(worker, gpu) for gpu in gpu_records]
        interrupted = False
        try:
            pending = futures
            while pending:
                _, pending = wait(pending, timeout=30)
                if pending:
                    with mutex:
                        running = [r['cell'] for r in reports if r['status'] == 'running']
                    print(f"RUNNING {running}; remaining queued={work.qsize()}", flush=True)
            for future in futures:
                future.result()
        except KeyboardInterrupt:
            interrupted = True
            stop.set()
            with mutex:
                active = list(processes.values())
                for proc in active:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for proc in active:
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        finally:
            pool.shutdown(wait=True)
            with mutex:
                publish()
        counts = {s: sum(r["status"] == s for r in reports) for s in sorted({r["status"] for r in reports})}
        print(f"Final statuses: {counts}; blocked input groups={len(blocked)}; report={report_path}", flush=True)
        return 130 if interrupted else 2 if blocked or any(r["status"] in ("failed", "pending", "running") for r in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())

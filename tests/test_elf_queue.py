"""CPU-only resume/provenance and two-worker scheduling checks."""
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_elf_remaining as q
from elf_common import SOURCE_COMMIT, default_sampling, sha256, write_json


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assets = {t: {"checkpoint": "model", "sha256": {"model": "locked"}}
                       for t in ("owt", "wmt14", "xsum")}
        write_json(self.root / "artifacts/elf_lock.json", {})
        write_json(self.root / "data/elf/assets.json", {
            "assets": self.assets, "lock_sha256": sha256(self.root / "artifacts/elf_lock.json")})
        self.asset_hash = sha256(self.root / "data/elf/assets.json")

    def input(self, task, split, count):
        prefix = task == "owt-prefix"
        path = self.root / "data/elf" / ("owt-prefix.jsonl" if prefix else f"{task}-{split}.jsonl")
        path.write_text(''.join(json.dumps({"id": i}) + '\n' for i in range(count)))
        meta = {"task": task, "split": split, "count": count, "sha256": sha256(path)}
        if prefix:
            source = self.root / "data/conditional/owt-c64/prompts.jsonl"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("source\n")
            meta.update(protocol="c64_text_t5_v1", source_sha256=sha256(source))
        write_json(path.with_suffix(".manifest.json"), meta)
        return path

    def request(self, cell):
        return {
            "task": cell.task, "steps": cell.steps, "seed": 42, "compile": False, "batch_size": 8,
            "source_commit": SOURCE_COMMIT, "assets_manifest_sha256": self.asset_hash,
            "checkpoint_task": "owt" if cell.task == "owt-prefix" else cell.task,
            "checkpoint_selection": "ema_params1", "checkpoint_sha256": "locked",
            "sampling": default_sampling(cell.task, cell.steps),
            "model_canvas_t5_tokens": 1088 if cell.task == "xsum" else 128 if cell.task == "wmt14" else 1024,
            "protocol": "c64_text_t5_v1" if cell.task == "owt-prefix" else "elf_native_v1",
            "condition_tokenization_policy": q.EOS_POLICY if cell.task in ("wmt14", "xsum") else "none",
            "input": {"sha256": cell.input_sha256} if cell.input_sha256 else None,
        }

    def result(self, path, cell, status="quality_valid"):
        request = self.request(cell)
        write_json(path / "request.json", request)
        if status == "partial":
            return
        pairs = [(i, 0) for i in range(cell.count)]
        if cell.task == "owt-prefix":
            pairs += [(i, c) for c in range(1, 5) for i in range(min(256, cell.count))]
        samples = path / "samples.jsonl"
        samples.write_text(''.join(json.dumps({"id": j, "prompt_id": i, "completion_id": c}) + '\n'
                                   for j, (i, c) in enumerate(pairs)))
        gen = {**request, "sample_count": len(pairs), "prompt_count": cell.count,
               "diversity": cell.task == "owt-prefix", "samples_sha256": sha256(samples)}
        write_json(path / "generation.json", gen)
        if status == "needs_evaluation":
            return
        metric = {"task": cell.task, "sample_count": cell.count, "samples_sha256": sha256(samples),
                  "generation_manifest_sha256": sha256(path / "generation.json"),
                  "valid": status == "quality_valid", "metrics": {"entropy_gpt2_nats": 5.1}}
        if not metric["valid"]:
            metric["reason"] = "At least one completion has fewer than two GPT-2 tokens; no samples were dropped"
        write_json(path / "metrics.json", metric)

    def reusable(self, cell):
        return q.reusable_result(self.root, cell, self.asset_hash, self.assets)

    def test_full_matrix_and_input_block_isolation(self):
        cells, blocked = q.build_matrix(self.root, q.TASKS)
        self.assertEqual(len(cells), 7)
        self.assertEqual(len(blocked), 5)
        for task, split, count in (("owt-prefix", "c64-text-t5", 1024),
                                  ("wmt14", "validation", 3000), ("wmt14", "test", 3003),
                                  ("xsum", "validation", 11332), ("xsum", "test", 11334)):
            self.input(task, split, count)
        cells, blocked = q.build_matrix(self.root, q.TASKS)
        self.assertFalse(blocked)
        self.assertEqual(len(cells), 32)
        self.assertEqual(len({c.key for c in cells}), 32)
        self.assertEqual([c.steps for c in cells if c.task == "owt"], [1, 2, 4, 8, 16, 32, 1024])
        source = self.root / "data/conditional/owt-c64/prompts.jsonl"
        source.write_text("changed\n")
        cells, blocked = q.build_matrix(self.root, q.TASKS)
        self.assertEqual(len(cells), 25)
        self.assertEqual(blocked[0]["task"], "owt-prefix")

    def test_reuse_scores_or_generation_but_not_sanity_or_tampered_samples(self):
        cell = q.Cell("owt", "unconditional", 8, 1024)
        sanity = self.root / "results/elf-sanity/quality" / cell.key
        self.result(sanity, replace(cell, count=1000))
        self.assertIsNone(self.reusable(cell)[0])
        path = self.root / "results/elf-old/quality" / cell.key
        self.result(path, cell, "needs_evaluation")
        self.assertEqual(self.reusable(cell)[0]["status"], "needs_evaluation")
        self.result(path, cell)
        self.assertEqual(self.reusable(cell)[0]["status"], "quality_valid")
        with (path / "samples.jsonl").open("a") as f:
            f.write('{}\n')
        found, rejected = self.reusable(cell)
        self.assertIsNone(found)
        self.assertTrue(rejected)

    def test_sampling_batch_and_no_eos_mismatch_rejected(self):
        cell = q.Cell("wmt14", "validation", 8, 3000, "data", "hash")
        for key, value in (("batch_size", 16), ("condition_tokenization_policy", "old"),
                           ("sampling", {}), ("checkpoint_sha256", "different")):
            request = self.request(cell)
            request[key] = value
            self.assertFalse(q.matches_request(request, cell, self.asset_hash, self.assets))

    def test_invalid_quality_not_retried_and_conflicts_rejected(self):
        cell = q.Cell("owt", "unconditional", 1, 2)
        path = self.root / "results/elf-old/quality" / cell.key
        self.result(path, cell, "quality_invalid")
        self.assertEqual(self.reusable(cell)[0]["status"], "quality_invalid")
        self.result(self.root / "results/elf-other/quality" / cell.key, cell)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.reusable(cell)

    def test_partial_runs_preserved_and_prefix_requests_diversity(self):
        cell = q.Cell("owt-prefix", "c64-text-t5", 32, 1024, "inputs", "digest")
        root = self.root / "results/elf-gpu01"
        path = root / "quality" / cell.key
        self.result(path, cell, "partial")
        retry = q.fresh_path(root, cell)
        self.assertIn("elf-gpu01-retry1", str(retry))
        self.assertTrue((path / "request.json").exists())
        cmds = q.commands(self.root, sys.executable, cell, retry, True)
        self.assertEqual([stage for stage, _ in cmds], ["generate", "evaluate"])
        self.assertIn("--diversity", cmds[0][1])
        self.assertNotIn("timing", str(cmds))
        self.assertEqual([s for s, _ in q.commands(self.root, sys.executable, cell, path, False)], ["evaluate"])
        self.result(path, cell)
        self.assertEqual(self.reusable(cell)[0]["split"], "c64-text-t5")

    def test_plan_does_not_query_gpu_or_launch_processes(self):
        with patch.object(sys, "argv", ["queue", "--root", str(self.root), "--plan"]), \
                patch.object(q.subprocess, "check_output") as query, \
                patch.object(q.subprocess, "Popen") as popen, redirect_stdout(io.StringIO()):
            self.assertEqual(q.main(), 2)  # Missing conditional inputs, OWT still planned.
        query.assert_not_called()
        popen.assert_not_called()
        self.assertFalse((self.root / "results").exists())

    def test_two_workers_isolated_and_invalid_quality_does_not_stop_queue(self):
        cells = [q.Cell("owt", "unconditional", s, 2) for s in (1, 2, 4)]
        barrier = threading.Barrier(2, timeout=10)
        launches, lock = [], threading.Lock()

        def fake_popen(command, **kwargs):
            stage = "generate" if "generate" in command else "evaluate"
            path = Path(command[command.index("--output" if stage == "generate" else "--run") + 1])
            cell = next(c for c in cells if str(path).endswith(c.key))
            with lock:
                launches.append((cell.steps, stage, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))

            class Process:
                def wait(inner):
                    if stage == "generate" and cell.steps in (1, 2):
                        barrier.wait()
                    status = "needs_evaluation" if stage == "generate" else "quality_invalid" if cell.steps == 1 else "quality_valid"
                    self.result(path, cell, status)
                    return int(status == "quality_invalid")
            return Process()

        with patch.object(sys, "argv", ["queue", "--root", str(self.root)]), \
                patch.object(sys, "platform", "linux"), patch.object(q, "build_matrix", return_value=(cells, [])), \
                patch.object(q.subprocess, "check_output", side_effect=lambda cmd, **kw: f"GPU-{cmd[2]}, H200 NVL"), \
                patch.object(q.subprocess, "Popen", side_effect=fake_popen), redirect_stdout(io.StringIO()):
            self.assertEqual(q.main(), 0)
        self.assertEqual(len(launches), 6)
        self.assertEqual({gpu for _, _, gpu in launches}, {"GPU-0", "GPU-1"})
        for cell in cells:
            self.assertEqual(len({gpu for step, _, gpu in launches if step == cell.steps}), 1)
        report = json.loads(next((self.root / "results/elf-gpu01/logs").glob("*/queue.json")).read_text())
        self.assertEqual([r["status"] for r in report["runs"]], ["quality_invalid", "quality_valid", "quality_valid"])


if __name__ == "__main__":
    unittest.main()

"""CPU-only contract checks; no model/data download and no torch dependency."""
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from elf_common import SOURCE_COMMIT, default_sampling, manifest, sha256, write_json
from evaluate_elf import validate_records
from run_elf import parse_args


class ELFContracts(unittest.TestCase):
    def test_release_bindings(self):
        root = Path(__file__).resolve().parents[1]
        lock = json.loads((root / "artifacts/elf_lock.json").read_text())
        self.assertEqual(lock["source"]["commit"], SOURCE_COMMIT)
        for task in ("owt", "wmt14", "xsum"):
            self.assertTrue(lock["assets"][task]["repo_id"].endswith("-torch"))
            self.assertIn(lock["assets"][task]["checkpoint"], lock["assets"][task]["patterns"])

    def test_source_paper_budget_settings(self):
        self.assertEqual(default_sampling("owt", 8)["gamma"], 2)
        self.assertEqual(default_sampling("owt", 32)["gamma"], 1.5)
        self.assertEqual(default_sampling("owt", 64)["gamma"], 1)
        self.assertEqual(default_sampling("wmt14", 64)["cfg"], 2)
        self.assertEqual(default_sampling("xsum", 64)["sampling_method"], "ode")

    def test_reject_throughput_mislabeled_as_latency(self):
        with self.assertRaises(SystemExit):
            parse_args(["timing", "--task", "owt", "--steps", "32", "--output", "/tmp/unused", "--batch-size", "8"])

    def test_reject_wrong_checkpoint_task_interface(self):
        with self.assertRaises(SystemExit):
            parse_args(["generate", "--task", "wmt14", "--steps", "64", "--output", "/tmp/unused"])

    def test_missing_or_duplicated_completions_fail(self):
        pairs = [(i, 0) for i in range(2)] + [(i, c) for c in range(1, 5) for i in range(2)]
        rows = [{"id": j, "prompt_id": i, "completion_id": c} for j, (i, c) in enumerate(pairs)]
        meta = {"sample_count": 10, "prompt_count": 2, "diversity": True}
        validate_records(rows, meta)
        with self.assertRaises(ValueError):
            validate_records(rows[:-1], meta)
        rows[-1] = rows[-2]
        with self.assertRaises(ValueError):
            validate_records(rows, meta)

    def test_asset_lock_tamper_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "artifacts/elf_lock.json", {"version": 1})
            write_json(root / "data/elf/assets.json", {"lock_sha256": sha256(root / "artifacts/elf_lock.json")})
            manifest(root)
            write_json(root / "artifacts/elf_lock.json", {"version": 2})
            with self.assertRaises(ValueError):
                manifest(root)

    def test_suite_routes_test_full_split_and_single_gpu_timing(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            stub = Path(directory) / "fake-python"
            stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            stub.chmod(0o755)
            env = dict(os.environ, ELF_PYTHON=str(stub), ELF_STEPS="64", ELF_SPLIT="test", DLB_ROOT=str(root))
            for key in ("ELF_COUNT", "ELF_BATCH_SIZE", "ELF_COMPILE"):
                env.pop(key, None)
            generated = subprocess.check_output(["bash", str(root / "scripts/run_elf_suite.sh"), "generate", "wmt14"], env=env, text=True).splitlines()
            self.assertEqual(generated[generated.index("--num-samples") + 1], "0")
            self.assertTrue(generated[generated.index("--input") + 1].endswith("wmt14-test.jsonl"))
            timed = subprocess.check_output(["bash", str(root / "scripts/run_elf_suite.sh"), "timing", "xsum"], env=env, text=True).splitlines()
            self.assertEqual(timed[timed.index("--batch-size") + 1], "1")


if __name__ == "__main__":
    unittest.main()

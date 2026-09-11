"""CPU-only contract checks; no model/data download and no torch dependency."""
import json
import io
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from elf_common import SOURCE_COMMIT, default_sampling, manifest, sha256, write_json
from evaluate_elf import validate_records
from run_elf import parse_args
from prepare_elf import download_snapshot, retry_delay


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
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            parse_args(["timing", "--task", "owt", "--steps", "32", "--output", "/tmp/unused", "--batch-size", "8"])

    def test_reject_wrong_checkpoint_task_interface(self):
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
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


class ELFDownloadRecovery(unittest.TestCase):
    @staticmethod
    def hub_error(status, headers=None):
        class FailedResponse:
            # Match requests.Response.__bool__ on failed status codes.
            def __bool__(self):
                return False
        response = FailedResponse()
        response.status_code = status
        response.headers = headers or {}
        http = RuntimeError("HTTP failure")
        http.response = response
        wrapped = RuntimeError("LocalEntryNotFoundError: no snapshot on disk")
        wrapped.__cause__ = http
        return wrapped

    def test_wrapped_429_retries_same_snapshot_and_then_succeeds(self):
        download = Mock(side_effect=[self.hub_error(429, {"Retry-After": "90"}), "/cache/snapshot"])
        sleep = Mock()
        with redirect_stdout(io.StringIO()):
            result = download_snapshot(download, attempts=2, sleep=sleep,
                                       repo_id="owner/model", revision="locked", max_workers=1)
        self.assertEqual(result, "/cache/snapshot")
        self.assertEqual(download.call_args_list[0], download.call_args_list[1])
        sleep.assert_called_once_with(90.0)

    def test_retries_are_bounded_without_sleep_after_final_failure(self):
        download = Mock(side_effect=self.hub_error(503))
        sleep = Mock()
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
            download_snapshot(download, attempts=3, sleep=sleep, repo_id="owner/model")
        self.assertEqual(download.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [60.0, 120.0])

    def test_auth_and_missing_revision_do_not_retry(self):
        for status in (401, 403, 404):
            with self.subTest(status=status):
                original = self.hub_error(status)
                download, sleep = Mock(side_effect=original), Mock()
                with self.assertRaises(RuntimeError) as caught:
                    download_snapshot(download, sleep=sleep, repo_id="owner/model")
                self.assertIs(caught.exception, original)
                self.assertEqual(download.call_count, 1)
                sleep.assert_not_called()

    def test_local_failure_does_not_retry(self):
        download, sleep = Mock(side_effect=OSError("disk full")), Mock()
        with self.assertRaises(OSError):
            download_snapshot(download, sleep=sleep, repo_id="owner/model")
        sleep.assert_not_called()

    def test_ratelimit_reset_is_respected(self):
        response = SimpleNamespace(headers={"RateLimit": '"api";r=0;t=420'})
        self.assertEqual(retry_delay(response, 1), 421)

    def test_missing_manifest_explains_prerequisite(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(FileNotFoundError, "prepare_elf.py assets"):
                manifest(root)


if __name__ == "__main__":
    unittest.main()

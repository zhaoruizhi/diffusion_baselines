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
from elf_common import SOURCE_COMMIT, default_sampling, manifest, preflight_conditions, sha256, write_json
from audit_elf_inputs import audit_rows
from evaluate_elf import validate_records
from run_elf import parse_args
from prepare_elf import download_snapshot, retry_delay, official_arrow_rows, raw_parquet_rows, export_rows

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None


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


class ELFConditionAudit(unittest.TestCase):
    class Tokenizer:
        eos_token_id = 1

        def __len__(self):
            return 100

        def __call__(self, text, **kwargs):
            return {"input_ids": [2 + len(word) for word in text.split()]}

        def decode(self, ids, **kwargs):
            return str(ids)

    def test_preflight_finds_later_empty_row_without_dropping_or_changing_input(self):
        rows = [{"id": 0, "input": "source"}, {"id": 217, "input": "  ", "condition_input_ids": []}]
        before = json.dumps(rows)
        with self.assertRaisesRegex(ValueError, "source_id.*217"):
            preflight_conditions(rows, self.Tokenizer(), "xsum", 1024, 1088)
        self.assertEqual(json.dumps(rows), before)

    def test_official_ids_preserved_even_when_text_is_blank_and_prefix_not_truncated(self):
        row = {"id": 217, "input": "", "condition_input_ids": [1]}
        self.assertEqual(preflight_conditions([row], self.Tokenizer(), "xsum", 1024, 1088), [[1]])
        long = {"id": 0, "input": "", "condition_input_ids": [3] * 65}
        self.assertEqual(len(preflight_conditions([long], self.Tokenizer(), "wmt14", 64, 128)[0]), 64)
        with self.assertRaises(ValueError):
            preflight_conditions([long], self.Tokenizer(), "owt-prefix", 64, 128)

    def test_audit_matches_text_pairs_across_reordering_and_exposes_eos_difference(self):
        raw = [{"id": 0, "input": "abc", "output": "A"},
               {"id": 1, "input": "", "output": "B"}]
        official = [{"id": 10, "input": "", "output": "B", "condition_input_ids": [1]},
                    {"id": 11, "input": "abc", "output": "A", "condition_input_ids": [5, 1]}]
        report = audit_rows(raw, official, self.Tokenizer(), 64, "wmt14")
        self.assertEqual(report["raw"]["empty_condition_ids"], [1])
        self.assertTrue(report["official_conditions_nonempty"])
        comparison = report["comparison_by_exact_source_and_reference_text_not_row_number"]
        self.assertEqual(comparison["unique_text_pair"], 2)
        self.assertEqual(comparison["raw_plus_eos_equals_official"], 2)
        self.assertEqual(comparison["exact_ids"], 0)
        corrected = report["corrected_runner_comparison_including_duplicates"]
        self.assertTrue(corrected["ready_for_raw_validation"])
        self.assertTrue(corrected["full_token_multisets_equal"])
        self.assertFalse(corrected["same_order_and_full_tokens"])

    def test_task_trained_eos_handles_blank_and_is_added_before_source_cap(self):
        rows = [{"id": 0, "input": " "}, {"id": 1, "input": "word " * 63},
                {"id": 2, "input": "word " * 64}]
        before = json.dumps(rows)
        ids = preflight_conditions(rows, self.Tokenizer(), "wmt14", 64, 128)
        self.assertEqual(ids[0], [1])
        self.assertEqual(ids[1], [6] * 63 + [1])
        self.assertEqual(ids[2], [6] * 64)
        self.assertEqual(json.dumps(rows), before)
        self.assertEqual(preflight_conditions(rows[:1], self.Tokenizer(), "xsum", 1024, 1088), [[1]])
        self.assertEqual(preflight_conditions([{"id": 0, "input": "word"}], self.Tokenizer(),
                                             "owt-prefix", 64, 1024), [[6]])
        with self.assertRaises(ValueError):
            preflight_conditions(rows[:1], self.Tokenizer(), "owt-prefix", 64, 1024)

    def test_audit_checks_duplicate_multiplicity_and_disagreement(self):
        raw = [{"id": i, "input": "abc", "output": "A"} for i in range(2)]
        official = [{**r, "condition_input_ids": [5, 1]} for r in raw]
        report = audit_rows(raw, official, self.Tokenizer(), 64, "wmt14")
        self.assertEqual(report["comparison_by_exact_source_and_reference_text_not_row_number"]["ambiguous_text_pair"], 2)
        self.assertTrue(report["corrected_runner_comparison_including_duplicates"]["ready_for_raw_validation"])
        official[1]["condition_input_ids"] = [5, 2]
        report = audit_rows(raw, official, self.Tokenizer(), 64, "wmt14")
        self.assertFalse(report["corrected_runner_comparison_including_duplicates"]["ready_for_raw_validation"])


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


@unittest.skipIf(pa is None, "Arrow export checks require pyarrow (installed in dlb-elf)")
class ELFArrowCompatibility(unittest.TestCase):
    def write_shard(self, root, name, rows):
        table = pa.Table.from_pylist(rows)
        # The same HF feature annotation that datasets 3.6 cannot deserialize.
        features = {"condition_input_ids": {"_type": "List", "feature": {"_type": "Value", "dtype": "int64"}}}
        info = {"features": features}
        table = table.replace_schema_metadata({b"huggingface": json.dumps({"info": info}).encode()})
        with pa.OSFile(str(root / name), "wb") as sink:
            with pa.ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)
        write_json(root / "dataset_info.json", info)

    def test_list_metadata_exports_exact_rows_in_state_order_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_shard(root, "a.arrow", [{"input": "Zwei", "target": "Two", "condition_input_ids": [3]}])
            self.write_shard(root, "z.arrow", [{"input": "Eins", "target": "One", "condition_input_ids": [7, 8]}])
            write_json(root / "state.json", {"_data_files": [{"filename": "z.arrow"}, {"filename": "a.arrow"}]})
            hashes = {p: sha256(p) for p in root.iterdir()}
            rows = list(official_arrow_rows(root))
            self.assertEqual(rows, [{"id": 0, "input": "Eins", "output": "One", "condition_input_ids": [7, 8]},
                                    {"id": 1, "input": "Zwei", "output": "Two", "condition_input_ids": [3]}])
            self.assertEqual(hashes, {p: sha256(p) for p in root.iterdir()})
            output = root / "export.jsonl"
            with redirect_stdout(io.StringIO()):
                export_rows(output, rows, {"task": "wmt14"})
            sidecar = json.loads(output.with_suffix(".manifest.json").read_text())
            self.assertEqual(sidecar["count"], 2)
            self.assertEqual(sidecar["sha256"], sha256(output))

    def test_missing_reference_fails_instead_of_reconstructing_from_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_shard(root, "data.arrow", [{"input": "Source", "condition_input_ids": [1]}])
            write_json(root / "state.json", {"_data_files": [{"filename": "data.arrow"}]})
            with self.assertRaisesRegex(ValueError, "missing raw source/reference"):
                list(official_arrow_rows(root))

    def test_duplicate_shards_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_shard(root, "data.arrow", [{"input": "Source", "target": "Target", "condition_input_ids": [1]}])
            write_json(root / "state.json", {"_data_files": [{"filename": "data.arrow"}] * 2})
            with self.assertRaisesRegex(ValueError, "duplicated Arrow shard"):
                list(official_arrow_rows(root))

    def test_parquet_preserves_translation_direction_and_full_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            for task, raw, expected in [
                ("wmt14", {"translation": {"de": "Hallo", "en": "Hello"}}, {"id": 0, "input": "Hallo", "output": "Hello"}),
                ("xsum", {"document": "Full article", "summary": "Full reference"},
                 {"id": 0, "input": "Full article", "output": "Full reference"}),
            ]:
                path = Path(directory) / f"{task}.parquet"
                pq.write_table(pa.Table.from_pylist([raw]), path)
                self.assertEqual(list(raw_parquet_rows([path], task)), [expected])


if __name__ == "__main__":
    unittest.main()

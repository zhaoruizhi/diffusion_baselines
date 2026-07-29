import json
import copy
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from dlb.data import (
    build_data_manifest,
    build_processing_contract,
    disk_preflight_evidence,
    processing_contract_sha256,
    publication_is_reusable,
    publish_staged_output,
    recover_incomplete_publication,
    validate_download_manifest,
    validate_manifest_contract,
    verify_processed_dataset,
)
from dlb.io import atomic_json_write


class TinySplit:
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        return self._rows[index]


class TinyTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        assert skip_special_tokens is False
        return " ".join(str(token_id) for token_id in token_ids)


def _test_contract(
    *,
    dataset,
    dataset_id,
    source_revision,
    tokenizer_id,
    tokenizer_revision,
    sequence_length,
    split_expression,
    document_counts,
    vocab_size,
    materialization_revision=None,
):
    return {
        "schema_version": 1,
        "dataset": dataset,
        "dataset_id": dataset_id,
        "source_revision": source_revision,
        "source_materialization_revision": materialization_revision,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "sequence_length": sequence_length,
        "split_expression": split_expression,
        "expected_document_counts": document_counts,
        "vocab_size": vocab_size,
        "packing": {
            "algorithm": "document_eos_continuous_bos_eos",
            "version": 1,
        },
    }


def test_manifest_records_reproducibility_fields_and_file_digests(tmp_path):
    """Catch manifests that cannot identify and integrity-check their data cache."""

    output_dir = tmp_path / "processed"
    output_dir.mkdir()
    (output_dir / "train.arrow").write_bytes(b"train rows")
    (output_dir / "validation.arrow").write_bytes(b"validation rows")

    contract = _test_contract(
        dataset="owt",
        dataset_id="openwebtext",
        source_revision="a" * 40,
        tokenizer_id="gpt2",
        tokenizer_revision="b" * 40,
        sequence_length=1024,
        split_expression={
            "train": "train[:-100000]",
            "validation": "train[-100000:]",
        },
        document_counts={"train": 7_913_769, "validation": 100_000},
        vocab_size=50_257,
    )
    manifest = build_data_manifest(
        dataset="owt",
        dataset_id="openwebtext",
        source_revision="a" * 40,
        tokenizer_id="gpt2",
        tokenizer_revision="b" * 40,
        sequence_length=1024,
        split_expression={
            "train": "train[:-100000]",
            "validation": "train[-100000:]",
        },
        document_counts={"train": 7_913_769, "validation": 100_000},
        packed_sequence_counts={"train": 2, "validation": 1},
        vocab_size=50_257,
        min_token_id=1,
        max_token_id=50_256,
        output_dir=output_dir,
        root=tmp_path,
        processing_contract=contract,
    )

    assert manifest["source_revision"] == "a" * 40
    assert manifest["tokenizer_revision"] == "b" * 40
    assert manifest["split_expression"]["validation"] == "train[-100000:]"
    assert manifest["document_counts"] == {"train": 7_913_769, "validation": 100_000}
    assert manifest["packed_sequence_counts"] == {"train": 2, "validation": 1}
    assert manifest["vocabulary"] == {
        "size": 50_257,
        "min_token_id": 1,
        "max_token_id": 50_256,
    }
    assert manifest["created_at"].endswith("Z")
    assert manifest["processed_path"] == "processed"
    assert [entry["path"] for entry in manifest["files"]] == [
        "train.arrow",
        "validation.arrow",
    ]
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])
    assert manifest["verified"] is False
    assert manifest["processing_contract"] == contract
    assert manifest["processing_contract_sha256"] == processing_contract_sha256(contract)


def test_verifier_checks_lengths_bounds_counts_digests_and_decoding(tmp_path):
    """Catch a verifier that trusts manifest claims without reading processed rows."""

    output_dir = tmp_path / "processed"
    output_dir.mkdir()
    (output_dir / "state.json").write_text("complete\n", encoding="utf-8")
    dataset = {
        "train": TinySplit([{"input_ids": [101, 10, 11, 102]}]),
        "validation": TinySplit([{"input_ids": [101, 12, 13, 102]}]),
    }
    contract = _test_contract(
        dataset="lm1b",
        dataset_id="lm1b",
        source_revision="c" * 40,
        tokenizer_id="bert-base-uncased",
        tokenizer_revision="d" * 40,
        sequence_length=4,
        split_expression={"train": "train", "validation": "test"},
        document_counts={"train": 2, "validation": 1},
        vocab_size=30_522,
    )
    manifest = build_data_manifest(
        dataset="lm1b",
        dataset_id="lm1b",
        source_revision="c" * 40,
        tokenizer_id="bert-base-uncased",
        tokenizer_revision="d" * 40,
        sequence_length=4,
        split_expression={"train": "train", "validation": "test"},
        document_counts={"train": 2, "validation": 1},
        packed_sequence_counts={"train": 1, "validation": 1},
        vocab_size=30_522,
        min_token_id=10,
        max_token_id=102,
        output_dir=output_dir,
        root=tmp_path,
        processing_contract=contract,
    )

    result = verify_processed_dataset(
        dataset=dataset,
        manifest=manifest,
        output_dir=output_dir,
        tokenizer=TinyTokenizer(),
        expected_contract=contract,
    )

    assert result["verified"] is True
    assert result["verification"]["checked_sequences"] == 2
    assert result["verification"]["decoded_samples"] == 2


def test_verifier_rejects_out_of_range_tokens(tmp_path):
    """Catch processed IDs that exceed the pinned tokenizer vocabulary."""

    output_dir = tmp_path / "processed"
    output_dir.mkdir()
    (output_dir / "state.json").write_text("complete\n", encoding="utf-8")
    contract = _test_contract(
        dataset="tiny",
        dataset_id="tiny",
        source_revision="e" * 40,
        tokenizer_id="tiny",
        tokenizer_revision="f" * 40,
        sequence_length=4,
        split_expression={"train": "train", "validation": "validation"},
        document_counts={"train": 1, "validation": 1},
        vocab_size=20,
    )
    manifest = build_data_manifest(
        dataset="tiny",
        dataset_id="tiny",
        source_revision="e" * 40,
        tokenizer_id="tiny",
        tokenizer_revision="f" * 40,
        sequence_length=4,
        split_expression={"train": "train", "validation": "validation"},
        document_counts={"train": 1, "validation": 1},
        packed_sequence_counts={"train": 1, "validation": 1},
        vocab_size=20,
        min_token_id=1,
        max_token_id=19,
        output_dir=output_dir,
        root=tmp_path,
        processing_contract=contract,
    )
    dataset = {
        "train": TinySplit([{"input_ids": [1, 2, 20, 3]}]),
        "validation": TinySplit([{"input_ids": [1, 2, 3, 4]}]),
    }

    with pytest.raises(ValueError, match="outside vocabulary"):
        verify_processed_dataset(
            dataset=dataset,
            manifest=manifest,
            output_dir=output_dir,
            tokenizer=TinyTokenizer(),
            expected_contract=contract,
        )


def test_verified_manifest_is_atomically_persistable(tmp_path):
    """Catch verification results that cannot be stored as valid JSON metadata."""

    path = tmp_path / "manifest.json"
    value = {"dataset": "tiny", "verified": True, "verification": {"ok": True}}

    atomic_json_write(path, value)

    assert json.loads(path.read_text(encoding="utf-8")) == value


def test_data_configuration_pins_exact_revisions_and_semantics():
    """Catch moving Hub references or accidental changes to split/tokenizer semantics."""

    root = Path(__file__).parents[1]
    configuration = yaml.safe_load((root / "artifacts" / "data.yaml").read_text())

    assert configuration["datasets"]["lm1b"] == {
        "repo_id": "billion-word-benchmark/lm1b",
        "revision": "35161838ea9e05371a25a8db037f94fcae4c2064",
        "parquet_revision": "8d52bfd3cc2819fc1166dbb8144c328e2690de3e",
        "config": "plain_text",
        "splits": {"train": "train", "validation": "test"},
        "sequence_length": 128,
        "tokenizer": "bert-base-uncased",
    }
    assert configuration["datasets"]["owt"] == {
        "repo_id": "Skylion007/openwebtext",
        "revision": "79d93d786212f7344586290adb811d4ae6a1762c",
        "config": "plain_text",
        "documents": 8_013_769,
        "splits": {
            "train": "train[:-100000]",
            "validation": "train[-100000:]",
        },
        "sequence_length": 1024,
        "tokenizer": "gpt2",
    }
    assert configuration["models"] == {
        "bert-base-uncased": "86b5e0934494bd15c9632b12f734a8a67f723594",
        "gpt2": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "gpt2-large": "32b71b12589c2f8d625668d2335a01cac3249519",
    }
    assert configuration["lm1b_archive"] == {
        "url": "http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz",
        "size_bytes": 1_792_209_805,
        "sha256": "01ba60381110baf7f189dfd2b8374de371e8c9a340835793f190bdae9e90a34e",
        "train_documents": 30_301_028,
        "test_documents": 306_688,
    }


def test_fetch_dry_run_lists_every_target_without_writing_data(tmp_path):
    """Catch dry-run mode that omits a required snapshot or creates cache files."""

    root = Path(__file__).parents[1]
    data_path = tmp_path / "data"

    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "fetch_data.py"),
            "--root",
            str(tmp_path),
            "--config",
            str(root / "artifacts" / "data.yaml"),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "DATASET lm1b billion-word-benchmark/lm1b@35161838ea9e05371a25a8db037f94fcae4c2064",
        "DATASET owt Skylion007/openwebtext@79d93d786212f7344586290adb811d4ae6a1762c",
        "MODEL bert-base-uncased@86b5e0934494bd15c9632b12f734a8a67f723594",
        "MODEL gpt2@607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "MODEL gpt2-large@32b71b12589c2f8d625668d2335a01cac3249519",
    ]
    assert not data_path.exists()


def test_owt_disk_preflight_requires_55_gib_unless_explicitly_overridden():
    """Catch starting the large OWT fetch on unsafe disk without explicit consent."""

    required = 55 * 1024**3

    with pytest.raises(RuntimeError, match="at least 55 GiB"):
        disk_preflight_evidence(required - 1, allow_low_disk=False)


def test_owt_disk_preflight_records_requested_and_actual_override_separately():
    """Catch metadata claiming a bypass when the flag was unnecessary."""

    required = 55 * 1024**3

    sufficient = disk_preflight_evidence(required, allow_low_disk=True)
    low = disk_preflight_evidence(required - 1, allow_low_disk=True)

    assert sufficient == {
        "observed_free_bytes": required,
        "required_free_bytes": required,
        "below_threshold": False,
        "override_requested": True,
        "override_used": False,
    }
    assert low == {
        "observed_free_bytes": required - 1,
        "required_free_bytes": required,
        "below_threshold": True,
        "override_requested": True,
        "override_used": True,
    }


def _configuration():
    root = Path(__file__).parents[1]
    return yaml.safe_load((root / "artifacts" / "data.yaml").read_text())


def _valid_downloads(configuration):
    lm1b = configuration["datasets"]["lm1b"]
    owt = configuration["datasets"]["owt"]
    return {
        "schema_version": 1,
        "datasets": {
            "lm1b": {
                "repo_id": lm1b["repo_id"],
                "source_revision": lm1b["revision"],
                "materialization_revision": lm1b["parquet_revision"],
                "split_rows": {
                    "train": configuration["lm1b_archive"]["train_documents"],
                    "test": configuration["lm1b_archive"]["test_documents"],
                },
                "dataset_snapshot": (
                    "data/raw/huggingface/hub/datasets--lm1b/snapshots/"
                    + lm1b["parquet_revision"]
                ),
                "source_metadata_snapshot": (
                    "data/raw/huggingface/hub/datasets--lm1b/snapshots/"
                    + lm1b["revision"]
                ),
                "cache_files": {"train": ["train.arrow"], "test": ["test.arrow"]},
            },
            "owt": {
                "repo_id": owt["repo_id"],
                "source_revision": owt["revision"],
                "materialization_revision": None,
                "split_rows": {"train": owt["documents"]},
                "dataset_snapshot": None,
                "cache_files": {"train": ["owt.arrow"]},
            },
        },
        "models": {
            name: {
                "repo_id": name,
                "revision": revision,
                "snapshot_path": (
                    f"data/raw/huggingface/hub/models--{name}/snapshots/{revision}"
                ),
            }
            for name, revision in configuration["models"].items()
        },
    }


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("repo_id", "wrong/lm1b"),
        ("source_revision", "0" * 40),
        ("materialization_revision", "1" * 40),
        ("split_rows", {"train": 1, "test": 1}),
    ],
)
def test_preprocessing_rejects_stale_lm1b_download_provenance(field, bad_value):
    """Catch preprocessing data that does not match its claimed immutable source."""

    configuration = _configuration()
    downloads = _valid_downloads(configuration)
    downloads["datasets"]["lm1b"][field] = bad_value

    with pytest.raises(ValueError, match=field):
        validate_download_manifest(configuration, downloads, "lm1b")


def test_preprocessing_validates_owt_rows_and_tokenizer_snapshot_revision():
    """Catch truncated OWT caches or a tokenizer snapshot from another revision."""

    configuration = _configuration()
    downloads = _valid_downloads(configuration)
    downloads["datasets"]["owt"]["split_rows"] = {"train": 8_000_000}
    with pytest.raises(ValueError, match="split_rows"):
        validate_download_manifest(configuration, downloads, "owt")

    downloads = _valid_downloads(configuration)
    downloads["models"]["gpt2"]["revision"] = "2" * 40
    with pytest.raises(ValueError, match="tokenizer revision"):
        validate_download_manifest(configuration, downloads, "owt")


def test_preprocessing_rejects_snapshot_paths_for_other_immutable_revisions():
    """Catch correct labels pointing preprocessing at stale snapshot content."""

    configuration = _configuration()
    downloads = _valid_downloads(configuration)
    downloads["datasets"]["lm1b"]["dataset_snapshot"] = (
        "data/raw/huggingface/hub/datasets--lm1b/snapshots/" + "4" * 40
    )

    with pytest.raises(ValueError, match="dataset_snapshot"):
        validate_download_manifest(configuration, downloads, "lm1b")


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("sequence_length",), 999),
        (("split_expression", "validation"), "train[:100000]"),
        (("tokenizer_id",), "wrong-tokenizer"),
        (("source_materialization_revision",), "3" * 40),
        (("packing", "version"), 999),
    ],
)
def test_manifest_contract_rejects_processing_semantic_mutations(path, bad_value):
    """Catch reuse or verification under stale processing semantics."""

    expected = build_processing_contract(_configuration(), "lm1b")
    manifest_contract = copy.deepcopy(expected)
    target = manifest_contract
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = bad_value
    manifest = {
        "processing_contract": manifest_contract,
        "processing_contract_sha256": processing_contract_sha256(manifest_contract),
    }

    with pytest.raises(ValueError, match="processing contract"):
        validate_manifest_contract(manifest, expected)


def test_restart_recovers_final_output_missing_manifest_and_publishes_cleanly(tmp_path):
    """Catch a crash-after-rename state requiring manual deletion before rerun."""

    configuration = _configuration()
    contract = build_processing_contract(configuration, "lm1b")
    final = tmp_path / "lm1b-bert-128"
    final.mkdir()
    (final / "interrupted.arrow").write_bytes(b"incomplete")
    manifest_path = tmp_path / "lm1b.json"

    assert recover_incomplete_publication(final, manifest_path, contract) is False
    assert not final.exists()

    staged = tmp_path / "lm1b-bert-128.staging"
    staged.mkdir()
    (staged / "train.arrow").write_bytes(b"complete")
    manifest = build_data_manifest(
        dataset="lm1b",
        dataset_id=contract["dataset_id"],
        source_revision=contract["source_revision"],
        tokenizer_id=contract["tokenizer_id"],
        tokenizer_revision=contract["tokenizer_revision"],
        sequence_length=contract["sequence_length"],
        split_expression=contract["split_expression"],
        document_counts=contract["expected_document_counts"],
        packed_sequence_counts={"train": 1, "validation": 1},
        vocab_size=contract["vocab_size"],
        min_token_id=1,
        max_token_id=102,
        output_dir=staged,
        root=tmp_path,
        processing_contract=contract,
        published_output_dir=final,
    )

    publish_staged_output(staged, final, manifest_path, manifest)

    assert publication_is_reusable(final, manifest_path, contract)
    assert json.loads(manifest_path.read_text())["processed_path"] == "lm1b-bert-128"

from pathlib import Path

import hashlib
import json

import pytest

from dlb.conditional_prompts import (
    build_prompts,
    load_protocol,
    select_source_indices,
    verify_prompts,
)
from scripts.build_conditional_prompts import main as build_main


PRODUCTION_CONFIG = """\
schema_version: 1
protocol: c64_zs_v1
selection_seed: 42
sampling_seed: 42
prompt_count: 1024
prefix_length: 64
evaluation_continuation_length: 64
diversity_prompt_count: 256
completions_per_diversity_prompt: 5
datasets:
  lm1b:
    model_length: 128
  owt:
    model_length: 1024
"""


def test_protocol_has_exact_c64_production_contract(tmp_path: Path) -> None:
    """Catch protocol drift that would change the published C64 benchmark."""

    path = tmp_path / "conditional.yaml"
    path.write_text(PRODUCTION_CONFIG, encoding="utf-8")

    protocol = load_protocol(path)

    assert protocol.protocol == "c64_zs_v1"
    assert (protocol.prompt_count, protocol.prefix_length) == (1024, 64)
    assert protocol.datasets["owt"].model_length == 1024


def test_selection_is_unique_stable_and_seed_bound() -> None:
    """Catch selection that is non-deterministic, duplicated, or seed-insensitive."""

    first = select_source_indices(4096, 1024, 42)

    assert first == select_source_indices(4096, 1024, 42)
    assert first != select_source_indices(4096, 1024, 43)
    assert len(first) == len(set(first)) == 1024


class TinyValidation:
    def __init__(self, rows: list[dict[str, list[int]]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


def production_protocol():
    return load_protocol(Path(__file__).parents[1] / "configs" / "conditional.yaml")


def install_fake_load_from_disk(monkeypatch, *, rows: int) -> list[Path]:
    observed: list[Path] = []

    def fake_load_from_disk(path: Path) -> TinyValidation:
        observed.append(path)
        return TinyValidation(
            [{"input_ids": [index % 30_522] * 128} for index in range(rows)]
        )

    monkeypatch.setattr("dlb.conditional_prompts.load_from_disk", fake_load_from_disk)
    return observed


def make_prompt_root(tmp_path: Path) -> Path:
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "data.yaml").write_text(
        """\
schema_version: 1
datasets:
  lm1b:
    repo_id: billion-word-benchmark/lm1b
    revision: source-revision
    splits: {validation: test}
    sequence_length: 128
    tokenizer: bert-base-uncased
models: {bert-base-uncased: tokenizer-revision}
""",
        encoding="utf-8",
    )
    manifests = tmp_path / "data" / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "lm1b.json").write_text(
        json.dumps(
            {
                "dataset": "lm1b",
                "dataset_id": "billion-word-benchmark/lm1b",
                "tokenizer_id": "bert-base-uncased",
                "tokenizer_revision": "tokenizer-revision",
                "sequence_length": 128,
                "vocabulary": {"size": 30_522},
                "packed_sequence_counts": {"validation": 2048},
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def built_prompt_tree(monkeypatch, tmp_path: Path) -> Path:
    root = make_prompt_root(tmp_path)
    install_fake_load_from_disk(monkeypatch, rows=2048)
    build_prompts(root, "lm1b", production_protocol())
    return root


def test_build_writes_sha_bound_prompt_records(built_prompt_tree: Path) -> None:
    """Catch prompt publication that loses source alignment or its integrity bindings."""

    manifest_path = built_prompt_tree / "data/manifests/conditional-lm1b-c64.json"
    prompts_path = built_prompt_tree / "data/conditional/lm1b-c64/prompts.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in prompts_path.read_text(encoding="utf-8").splitlines()]

    assert len(records) == 1024
    assert [record["prompt_id"] for record in records] == list(range(1024))
    assert all(len(record["prefix_token_ids"]) == len(record["reference_token_ids"]) == 64 for record in records)
    assert manifest["source_manifest_sha256"] == hashlib.sha256(
        (built_prompt_tree / "data/manifests/lm1b.json").read_bytes()
    ).hexdigest()
    assert manifest["source_row_count"] == 2048
    assert manifest["prompt_file_sha256"] == hashlib.sha256(prompts_path.read_bytes()).hexdigest()


def test_verify_rejects_prompt_file_tampering(built_prompt_tree: Path) -> None:
    """Catch a verifier that trusts the prompt file without checking its digest."""

    prompts = built_prompt_tree / "data/conditional/lm1b-c64/prompts.jsonl"
    prompts.write_bytes(prompts.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="prompt file SHA-256"):
        verify_prompts(built_prompt_tree, "lm1b", production_protocol())


def test_verify_rejects_duplicate_json_keys_even_with_a_recomputed_digest(
    built_prompt_tree: Path,
) -> None:
    """Catch ambiguous prompt JSON that could otherwise hide a changed field."""

    prompts = built_prompt_tree / "data/conditional/lm1b-c64/prompts.jsonl"
    lines = prompts.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace('"prompt_id":0,', '"prompt_id":0,"prompt_id":0,', 1)
    prompts.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_path = built_prompt_tree / "data/manifests/conditional-lm1b-c64.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prompt_file_sha256"] = hashlib.sha256(prompts.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        verify_prompts(built_prompt_tree, "lm1b", production_protocol())


def test_build_rejects_tokens_outside_source_vocabulary(monkeypatch, tmp_path: Path) -> None:
    """Catch invalid processed tokens before they enter a prompt artifact."""

    root = make_prompt_root(tmp_path)

    def invalid_rows(path: Path) -> TinyValidation:
        return TinyValidation([{"input_ids": [30_522] * 128} for _ in range(2048)])

    monkeypatch.setattr("dlb.conditional_prompts.load_from_disk", invalid_rows)

    with pytest.raises(ValueError, match="outside vocabulary"):
        build_prompts(root, "lm1b", production_protocol())


def test_build_rejects_source_manifest_validation_count_drift(monkeypatch, tmp_path: Path) -> None:
    """Catch building against rows that no longer match their data-manifest count."""

    root = make_prompt_root(tmp_path)
    install_fake_load_from_disk(monkeypatch, rows=2048)
    manifest_path = root / "data/manifests/lm1b.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["packed_sequence_counts"]["validation"] = 2047
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="validation row count"):
        build_prompts(root, "lm1b", production_protocol())


def test_prompt_cli_uses_processed_validation_without_repacking(monkeypatch, tmp_path: Path) -> None:
    """Catch the build CLI reading a different source or attempting preprocessing."""

    make_prompt_root(tmp_path)
    config = tmp_path / "configs" / "conditional.yaml"
    config.parent.mkdir()
    config.write_text(PRODUCTION_CONFIG, encoding="utf-8")
    observed = install_fake_load_from_disk(monkeypatch, rows=2048)

    assert build_main(["--root", str(tmp_path), "--dataset", "lm1b"]) == 0
    assert observed == [tmp_path / "data/processed/lm1b-bert-128/validation"]

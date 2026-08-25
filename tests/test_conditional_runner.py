import hashlib
import json
import sys
from pathlib import Path

import pytest

import dlb.runner as runner_module
from dlb.conditional_prompts import PromptManifest
from dlb.io import expected_conditional_schedule
from dlb.runner import RunRequest, _identity, run_experiment
from tests.test_runner import FakeAdapter, prepare_canonical_root


ROOT = Path(__file__).parents[1]
SCHEDULE_TEXT = "c0:p0-1023;c1-4:p0-255"


def _base_request() -> RunRequest:
    return RunRequest(
        run_id="flm-lm1b-steps-2",
        model_id="flm",
        dataset_id="lm1b",
        step_count=2,
        seed=42,
        sample_count=2,
    )


def _conditional_request(manifest: Path) -> RunRequest:
    return RunRequest(
        **{
            **_base_request().__dict__,
            "sample_count": 2048,
            "generation_mode": "conditional_prefix",
            "conditioning_manifest": str(manifest),
            "conditioning_manifest_sha256": "a" * 64,
            "conditioning_config_sha256": "b" * 64,
            "prefix_length": 64,
            "evaluation_continuation_length": 64,
            "prompt_count": 1024,
            "diversity_prompt_count": 256,
            "completions_per_diversity_prompt": 5,
            "completion_schedule": SCHEDULE_TEXT,
        }
    )


def test_unconditional_identity_has_no_conditional_keys():
    """Catch a change to unconditional cache identity caused by new conditional fields."""

    identity = _identity(_base_request(), ["python", "sample.py"])

    assert "generation_mode" not in identity
    assert "conditioning_manifest_sha256" not in identity


def test_conditional_identity_binds_manifest_and_schedule(tmp_path: Path):
    """Catch a conditional cache identity that can reuse a different prompt schedule."""

    identity = _identity(_conditional_request(tmp_path / "manifest.json"), ["python", "sample.py"])

    assert identity["generation_mode"] == "conditional_prefix"
    assert identity["conditioning_manifest_sha256"] == "a" * 64
    assert identity["prefix_length"] == 64
    assert identity["completion_schedule"] == SCHEDULE_TEXT


def test_unconditional_request_rejects_conditional_fields(tmp_path: Path):
    """Catch accidental conditional provenance being silently ignored by an unconditional run."""

    prepare_canonical_root(tmp_path)
    request = RunRequest(**{**_base_request().__dict__, "prefix_length": 64})
    command = [sys.executable, "-c", "print('ok')"]

    with pytest.raises(ValueError, match="conditional"):
        run_experiment(request, tmp_path, adapter=FakeAdapter(command, []))


def _conditional_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for sample_id, (prompt_id, completion_id) in enumerate(expected_conditional_schedule()):
        prefix = [1] * 64
        continuation = [2] * 64
        records.append(
            {
                "sample_id": sample_id,
                "prompt_id": prompt_id,
                "completion_id": completion_id,
                "source_index": sample_id,
                "prefix_token_ids": prefix,
                "continuation_token_ids": continuation,
                "reference_token_ids": [3] * 64,
                "full_token_ids": prefix + continuation,
                "prefix_text": "fixed prefix",
                "continuation_text": "generated continuation",
                "reference_text": "source continuation",
                "full_text": "fixed prefix generated continuation",
                "seed": 42,
                "generation_seconds": 0.25,
                "prefix_exact_match": True,
            }
        )
    return records


def test_conditional_runner_publishes_in_isolated_root_with_verified_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Catch conditional publication falling back to unconditional I/O or LM1B defaults."""

    prepare_canonical_root(tmp_path)
    config = tmp_path / "configs" / "conditional.yaml"
    config.write_text((ROOT / "configs" / "conditional.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    manifest_path = tmp_path / "data" / "conditional" / "lm1b-c64" / "prompts.jsonl"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}\n", encoding="utf-8")
    prompt_manifest = PromptManifest(
        schema_version=1,
        protocol="c64_zs_v1",
        dataset="lm1b",
        source_split="validation",
        source_processed_path="data/processed/lm1b-bert-128/validation",
        source_manifest_path="data/manifests/lm1b.json",
        source_manifest_sha256="c" * 64,
        tokenizer_id="bert-base-uncased",
        tokenizer_revision="revision",
        vocabulary_size=10,
        selection_algorithm="test-selection",
        selection_seed=42,
        source_row_count=1024,
        prompt_count=1024,
        prefix_length=64,
        evaluation_continuation_length=64,
        model_length=128,
        prompt_file="data/conditional/lm1b-c64/prompts.jsonl",
        prompt_file_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(runner_module, "verify_prompts", lambda root, dataset, config: prompt_manifest)
    observed: dict[str, object] = {}
    original_writer = runner_module.write_conditional_samples_atomic

    def writer(path, records, **kwargs):
        observed.update(kwargs)
        return original_writer(path, records, **kwargs)

    monkeypatch.setattr(runner_module, "write_conditional_samples_atomic", writer)
    expected_manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    expected_config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
    request = RunRequest(
        **{
            **_conditional_request(manifest_path).__dict__,
            "conditioning_manifest_sha256": expected_manifest_sha,
            "conditioning_config_sha256": expected_config_sha,
        }
    )
    command = [sys.executable, "-c", "print('ok')"]
    adapter = FakeAdapter(command, _conditional_records())

    result = run_experiment(request, tmp_path, adapter=adapter)

    assert result.status == "succeeded"
    assert result.run_dir == tmp_path / "results" / "conditional" / "samples" / "lm1b" / "flm" / "steps_2"
    assert observed == {
        "expected": 2048,
        "schedule": expected_conditional_schedule(),
        "sequence_length": 128,
        "vocab_size": 10,
    }


def test_conditional_runner_allows_descendant_of_isolated_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Catch conditional smoke output roots being rejected despite remaining isolated."""

    prepare_canonical_root(tmp_path)
    config = tmp_path / "configs" / "conditional.yaml"
    config.write_text((ROOT / "configs" / "conditional.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    manifest_path = tmp_path / "data" / "conditional" / "lm1b-c64" / "prompts.jsonl"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}\n", encoding="utf-8")
    prompt_manifest = PromptManifest(
        schema_version=1, protocol="c64_zs_v1", dataset="lm1b", source_split="validation",
        source_processed_path="data/processed/lm1b-bert-128/validation", source_manifest_path="data/manifests/lm1b.json",
        source_manifest_sha256="c" * 64, tokenizer_id="bert-base-uncased", tokenizer_revision="revision",
        vocabulary_size=10, selection_algorithm="test-selection", selection_seed=42, source_row_count=1024,
        prompt_count=1024, prefix_length=64, evaluation_continuation_length=64, model_length=128,
        prompt_file="data/conditional/lm1b-c64/prompts.jsonl", prompt_file_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(runner_module, "verify_prompts", lambda root, dataset, config: prompt_manifest)
    request = RunRequest(
        **{
            **_conditional_request(manifest_path).__dict__,
            "conditioning_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "conditioning_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "results_root": str(tmp_path / "results" / "conditional" / "smoke"),
        }
    )
    command = [sys.executable, "-c", "print('ok')"]

    result = run_experiment(request, tmp_path, adapter=FakeAdapter(command, _conditional_records()))

    assert result.run_dir == tmp_path / "results" / "conditional" / "smoke" / "samples" / "lm1b" / "flm" / "steps_2"


def test_conditional_runner_rejects_results_root_outside_isolated_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Catch a conditional request that can write into unconditional results."""

    prepare_canonical_root(tmp_path)
    config = tmp_path / "configs" / "conditional.yaml"
    config.write_text((ROOT / "configs" / "conditional.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    prompt = tmp_path / "data" / "conditional" / "lm1b-c64" / "prompts.jsonl"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("{}\n", encoding="utf-8")
    manifest = _conditional_request(prompt)
    monkeypatch.setattr(runner_module, "verify_prompts", lambda root, dataset, config: PromptManifest(
        schema_version=1, protocol="c64_zs_v1", dataset="lm1b", source_split="validation",
        source_processed_path="x", source_manifest_path="y", source_manifest_sha256="c" * 64,
        tokenizer_id="tokenizer", tokenizer_revision="revision", vocabulary_size=10, selection_algorithm="selection",
        selection_seed=42, source_row_count=1024, prompt_count=1024, prefix_length=64,
        evaluation_continuation_length=64, model_length=128, prompt_file="data/conditional/lm1b-c64/prompts.jsonl",
        prompt_file_sha256=hashlib.sha256(prompt.read_bytes()).hexdigest(),
    ))
    request = RunRequest(**{**manifest.__dict__, "conditioning_manifest_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(), "conditioning_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(), "results_root": str(tmp_path / "results")})
    command = [sys.executable, "-c", "raise SystemExit(99)"]

    with pytest.raises(ValueError, match="conditional.*results_root"):
        run_experiment(request, tmp_path, adapter=FakeAdapter(command, []))


def test_conditional_runner_rejects_noncanonical_sampling_seed_before_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Catch a conditional run whose seed diverges from the verified C64 protocol."""

    prepare_canonical_root(tmp_path)
    config = tmp_path / "configs" / "conditional.yaml"
    config.write_text((ROOT / "configs" / "conditional.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    prompt = tmp_path / "data" / "conditional" / "lm1b-c64" / "prompts.jsonl"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(runner_module, "verify_prompts", lambda root, dataset, config: PromptManifest(
        schema_version=1, protocol="c64_zs_v1", dataset="lm1b", source_split="validation",
        source_processed_path="x", source_manifest_path="y", source_manifest_sha256="c" * 64,
        tokenizer_id="tokenizer", tokenizer_revision="revision", vocabulary_size=10, selection_algorithm="selection",
        selection_seed=42, source_row_count=1024, prompt_count=1024, prefix_length=64,
        evaluation_continuation_length=64, model_length=128, prompt_file="data/conditional/lm1b-c64/prompts.jsonl",
        prompt_file_sha256=hashlib.sha256(prompt.read_bytes()).hexdigest(),
    ))
    request = RunRequest(**{**_conditional_request(prompt).__dict__, "seed": 7, "conditioning_manifest_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(), "conditioning_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()})
    command = [sys.executable, "-c", "raise SystemExit(99)"]

    with pytest.raises(ValueError, match="seed"):
        run_experiment(request, tmp_path, adapter=FakeAdapter(command, []))

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from dlb.io import atomic_json_write


def test_token_projector_clamps_every_call_without_mutating_input():
    from dlb.adapters.conditional_runtime import token_project_fn

    prefix = torch.tensor([[4, 5], [6, 7]])
    project = token_project_fn(prefix)
    state = torch.tensor([[99, 99, 8], [99, 99, 9]])

    result = project(state)

    assert result.tolist() == [[4, 5, 8], [6, 7, 9]]
    assert state.tolist() == [[99, 99, 8], [99, 99, 9]]
    state[:, :2] = 88
    assert project(state).tolist() == [[4, 5, 8], [6, 7, 9]]


def test_vocab_projector_writes_clean_one_hot_prefix_and_preserves_suffix():
    from dlb.adapters.conditional_runtime import vocab_project_fn

    prefix = torch.tensor([[1, 3]])
    state = torch.randn(1, 4, 5)
    suffix = state[:, 2:, :].clone()

    result = vocab_project_fn(prefix)(state)

    assert torch.equal(result[:, :2, :], torch.nn.functional.one_hot(prefix, 5).to(state))
    assert torch.equal(result[:, 2:, :], suffix)
    assert torch.equal(state[:, 2:, :], suffix)


def test_rdlm_projector_encodes_prefix_tokens_as_base_n_digits():
    from dlb.adapters.conditional_runtime import base_n_digits, rdlm_project_fn

    prefix = torch.tensor([[5, 8]])  # base-3 little-endian: 5 -> [2, 1], 8 -> [2, 2]
    state = torch.randn(1, 5, 4)
    suffix = state[:, 4:, :].clone()

    result = rdlm_project_fn(prefix, base=3, digits_per_token=2)(state)

    expected_digits = base_n_digits(prefix, base=3, digits_per_token=2)
    expected = torch.nn.functional.one_hot(expected_digits, 4).to(state)
    assert expected_digits.tolist() == [[2, 1, 2, 2]]
    assert torch.equal(result[:, :4, :], expected)
    assert torch.equal(result[:, 4:, :], suffix)
    assert torch.equal(state[:, 4:, :], suffix)


def test_embedding_projector_writes_clean_prefix_and_preserves_suffix():
    from dlb.adapters.conditional_runtime import embedding_project_fn

    clean = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    state = torch.randn(1, 5, 2)
    suffix = state[:, 2:, :].clone()

    result = embedding_project_fn(clean)(state)

    assert torch.equal(result[:, :2, :], clean)
    assert torch.equal(result[:, 2:, :], suffix)
    assert torch.equal(state[:, 2:, :], suffix)


def test_projectors_reject_incompatible_shapes():
    from dlb.adapters.conditional_runtime import (
        embedding_project_fn,
        rdlm_project_fn,
        token_project_fn,
        vocab_project_fn,
    )

    with pytest.raises(ValueError, match="rank two"):
        token_project_fn(torch.tensor([1, 2]))(torch.zeros(1, 3, dtype=torch.long))
    with pytest.raises(ValueError, match="vocabulary"):
        vocab_project_fn(torch.tensor([[6]]))(torch.zeros(1, 2, 5))
    with pytest.raises(ValueError, match="RDLM"):
        rdlm_project_fn(torch.tensor([[4]]), base=5, digits_per_token=1)(
            torch.zeros(1, 2, 3)
        )
    with pytest.raises(ValueError, match="embedding"):
        embedding_project_fn(torch.zeros(2, 4, 3))(torch.zeros(1, 4, 3))


class FakeOwner:
    def update(self, value):
        return value


def test_scoped_patch_restores_even_after_sampler_error():
    from dlb.adapters.conditional_runtime import patched_attribute

    owner = FakeOwner()
    original = owner.update

    with pytest.raises(RuntimeError):
        with patched_attribute(owner, "update", lambda *_: (_ for _ in ()).throw(RuntimeError())):
            owner.update(None)

    assert owner.update.__func__ is original.__func__


def _write_manifest_tree(root: Path) -> Path:
    prompts = root / "data" / "conditional" / "lm1b-c64" / "prompts.jsonl"
    prompts.parent.mkdir(parents=True)
    rows = []
    for prompt_id in range(4):
        rows.append(
            {
                "prompt_id": prompt_id,
                "source_index": 100 + prompt_id,
                "prefix_token_ids": [prompt_id] * 64,
                "reference_token_ids": [prompt_id + 10] * 64,
                "source_sequence_sha256": f"{prompt_id:064x}",
            }
        )
    prompts.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    manifest = root / "data" / "manifests" / "conditional-lm1b-c64.json"
    manifest.parent.mkdir(parents=True)
    atomic_json_write(
        manifest,
        {
            "schema_version": 1,
            "protocol": "c64_zs_v1",
            "dataset": "lm1b",
            "source_split": "validation",
            "source_processed_path": "data/processed/lm1b-bert-128/validation",
            "source_manifest_path": "data/manifests/lm1b.json",
            "source_manifest_sha256": "a" * 64,
            "tokenizer_id": "bert-base-uncased",
            "tokenizer_revision": "b" * 40,
            "vocabulary_size": 30522,
            "selection_algorithm": "sha256_seed_index_digest_ascending_v1",
            "selection_seed": 42,
            "source_row_count": 4,
            "prompt_count": 4,
            "prefix_length": 64,
            "evaluation_continuation_length": 64,
            "model_length": 128,
            "prompt_file": "data/conditional/lm1b-c64/prompts.jsonl",
            "prompt_file_sha256": __import__("hashlib").sha256(prompts.read_bytes()).hexdigest(),
        },
    )
    return manifest


def test_load_conditioning_batch_follows_completion_boundaries(tmp_path: Path):
    from dlb.adapters.conditional_runtime import load_conditioning_batch
    from dlb.io import sha256_file

    manifest = _write_manifest_tree(tmp_path)
    digest = sha256_file(manifest)

    batch = load_conditioning_batch(
        manifest,
        digest,
        completion_id=0,
        prompt_start=1,
        batch_size=2,
        device="cpu",
        vocab_size=30522,
    )

    assert batch.prompt_ids == (1, 2)
    assert batch.source_indices == (101, 102)
    assert batch.prefix_token_ids.tolist() == [[1] * 64, [2] * 64]
    assert batch.reference_token_ids.tolist() == [[11] * 64, [12] * 64]

    with pytest.raises(ValueError, match="schedule boundary"):
        load_conditioning_batch(
            manifest,
            digest,
            completion_id=1,
            prompt_start=0,
            batch_size=257,
            device="cpu",
            vocab_size=30522,
        )


def test_load_conditioning_batch_accepts_runtime_vocab_with_extra_mask_dimension(tmp_path: Path):
    from dlb.adapters.conditional_runtime import load_conditioning_batch
    from dlb.io import sha256_file

    manifest = _write_manifest_tree(tmp_path)
    digest = sha256_file(manifest)

    batch = load_conditioning_batch(
        manifest,
        digest,
        completion_id=0,
        prompt_start=0,
        batch_size=1,
        device="cpu",
        vocab_size=30523,
    )

    assert batch.prefix_token_ids.tolist() == [[0] * 64]

    with pytest.raises(ValueError, match="smaller than prompt manifest"):
        load_conditioning_batch(
            manifest,
            digest,
            completion_id=0,
            prompt_start=0,
            batch_size=1,
            device="cpu",
            vocab_size=30521,
        )

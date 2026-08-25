import hashlib
from pathlib import Path

import pytest

import dlb.conditional_matrix as conditional_matrix
from dlb.conditional_prompts import PromptManifest, load_protocol
from dlb.matrix import build_matrix
from dlb.registry import load_registry


ROOT = Path(__file__).parents[1]


@pytest.fixture
def registry():
    return load_registry(ROOT / "configs/experiments.yaml")


def _prompt_manifest(dataset: str) -> PromptManifest:
    return PromptManifest(
        schema_version=1,
        protocol="c64_zs_v1",
        dataset=dataset,
        source_split="validation",
        source_processed_path=f"data/processed/{dataset}/validation",
        source_manifest_path=f"data/manifests/{dataset}.json",
        source_manifest_sha256="a" * 64,
        tokenizer_id="test-tokenizer",
        tokenizer_revision="revision",
        vocabulary_size=10,
        selection_algorithm="test-selection",
        selection_seed=42,
        source_row_count=1024,
        prompt_count=1024,
        prefix_length=64,
        evaluation_continuation_length=64,
        model_length=128 if dataset == "lm1b" else 1024,
        prompt_file=f"data/conditional/{dataset}-c64/prompts.jsonl",
        prompt_file_sha256="b" * 64,
    )


def _install_verified_sidecars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    for dataset in ("lm1b", "owt"):
        sidecar = tmp_path / "data" / "manifests" / f"conditional-{dataset}-c64.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(f'{{"dataset":"{dataset}"}}\n', encoding="utf-8")

    def verify(root: Path, dataset: str, protocol):
        assert root == tmp_path
        assert protocol.protocol == "c64_zs_v1"
        calls.append(dataset)
        return _prompt_manifest(dataset)

    monkeypatch.setattr(conditional_matrix, "verify_prompts", verify)
    return calls


def test_conditional_matrix_matches_supported_unconditional_tasks(registry, tmp_path, monkeypatch):
    """Catch a conditional matrix that diverges from the supported registry cells."""

    _install_verified_sidecars(tmp_path, monkeypatch)
    ordinary = build_matrix(registry, root=tmp_path)
    conditional = conditional_matrix.build_conditional_matrix(
        registry, root=tmp_path, protocol=load_protocol(ROOT / "configs/conditional.yaml")
    )

    assert len(conditional) == len(ordinary) == 137
    assert [(task.model, task.dataset, task.steps) for task in conditional] == [
        (task.model, task.dataset, task.steps) for task in ordinary
    ]
    assert all("/results/conditional/" in task.sample_dir for task in conditional)
    assert all(task.sample_count == 2048 for task in conditional)


def test_conditional_matrix_verifies_both_prompts_and_binds_sidecar_digest(
    registry, tmp_path, monkeypatch
):
    """Catch tasks published from unverified prompts or a prompt-file digest."""

    calls = _install_verified_sidecars(tmp_path, monkeypatch)
    tasks = conditional_matrix.build_conditional_matrix(
        registry, root=tmp_path, protocol=load_protocol(ROOT / "configs/conditional.yaml")
    )

    assert calls == ["lm1b", "owt"]
    for dataset in ("lm1b", "owt"):
        sidecar = tmp_path / "data" / "manifests" / f"conditional-{dataset}-c64.json"
        dataset_tasks = [task for task in tasks if task.dataset == dataset]
        assert {task.conditioning_manifest for task in dataset_tasks} == {str(sidecar)}
        assert {task.conditioning_manifest_sha256 for task in dataset_tasks} == {
            hashlib.sha256(sidecar.read_bytes()).hexdigest()
        }


@pytest.mark.parametrize("failure", ["missing", "tampered"])
def test_conditional_matrix_rejects_missing_or_tampered_prompt_manifest(
    registry, tmp_path, monkeypatch, failure
):
    """Catch matrix construction that emits cells after prompt verification fails."""

    if failure == "tampered":
        monkeypatch.setattr(
            conditional_matrix,
            "verify_prompts",
            lambda root, dataset, protocol: (_ for _ in ()).throw(ValueError("prompt manifest tampered")),
        )
    else:
        monkeypatch.setattr(conditional_matrix, "verify_prompts", lambda root, dataset, protocol: _prompt_manifest(dataset))

    with pytest.raises(ValueError, match="manifest|tampered|No such file"):
        conditional_matrix.build_conditional_matrix(
            registry, root=tmp_path, protocol=load_protocol(ROOT / "configs/conditional.yaml")
        )


def test_conditional_unsupported_inventory_is_rdlm_owt(registry):
    """Catch conditional reporting that omits the explicit unsupported RDLM/OWT cell."""

    from dlb.conditional_matrix import conditional_unsupported_inventory

    assert conditional_unsupported_inventory(registry) == [
        {
            "status": "unsupported",
            "model": "rdlm",
            "dataset": "owt",
            "category": "few",
            "reason": registry.models["rdlm"].datasets["owt"].reason,
        }
    ]

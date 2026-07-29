import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml
from jsonschema.validators import Draft202012Validator

from dlb.checkpoints import (
    CheckpointResource,
    DigestSpec,
    GDriveSource,
    load_checkpoint_manifest,
    validate_checkpoint_coverage,
)
from dlb.registry import DatasetSupport, load_registry


ROOT = Path(__file__).parents[1]


@pytest.fixture
def registry():
    return load_registry(ROOT / "configs" / "experiments.yaml")


@pytest.fixture
def checkpoints():
    return load_checkpoint_manifest(ROOT / "artifacts" / "checkpoints.yaml")


def test_every_supported_cell_has_checkpoint_or_training_recipe(registry, checkpoints):
    """Catch supported experiments with no auditable acquisition path."""

    for model_name, model in registry.models.items():
        for dataset, support in model.datasets.items():
            if support.status == "supported":
                assert (model_name, dataset) in checkpoints.coverage or support.train_recipe


def test_runtime_coverage_validation_rejects_a_recipe_for_the_wrong_cell(
    registry, checkpoints
):
    """Catch a registry recipe reference that points at another model/dataset cell."""

    validate_checkpoint_coverage(registry, checkpoints)
    mutated = registry.model_copy(deep=True)
    mutated.models["duo_dcd"].datasets["lm1b"].train_recipe = "candi_owt"

    with pytest.raises(ValueError, match="does not describe duo_dcd/lm1b"):
        validate_checkpoint_coverage(mutated, checkpoints)


def test_checkpoint_sources_are_typed_and_public_resources_are_auditable(checkpoints):
    """Catch an unimplemented backend or a public resource missing terms/digest policy."""

    assert {resource.backend for resource in checkpoints.resources.values()} <= {
        "huggingface",
        "gdrive",
        "zenodo",
        "direct",
    }
    assert {resource.backend for resource in checkpoints.resources.values()} >= {
        "huggingface",
        "gdrive",
        "zenodo",
    }
    for resource in checkpoints.resources.values():
        assert resource.terms_url.startswith("https://")
        assert resource.digest.policy in {"sha256", "capture_after_download"}
        if resource.digest.policy == "sha256":
            assert len(resource.digest.sha256) == 64
        else:
            assert resource.digest.sha256 is None


def test_huggingface_resources_declare_required_payload_inventory(checkpoints):
    """Catch allow filters being mistaken for proof that model weights arrived."""

    for resource in checkpoints.resources.values():
        if resource.backend != "huggingface":
            continue
        assert resource.required_files
        assert "model.safetensors" in resource.required_files
        assert set(resource.required_files) <= set(resource.source.allow_patterns)


def test_multifile_digest_policy_is_typed_and_aggregate_sha_is_rejected():
    """Catch directory backends silently ignoring one aggregate SHA-256."""

    first = "1" * 64
    second = "2" * 64
    digest = DigestSpec(
        policy="sha256",
        per_file_sha256={"config.json": first, "model.safetensors": second},
    )
    assert digest.per_file_sha256 == {
        "config.json": first,
        "model.safetensors": second,
    }

    with pytest.raises(ValueError, match="aggregate SHA-256"):
        CheckpointResource.model_validate(
            {
                "id": "bad_hf",
                "backend": "huggingface",
                "provenance": "official",
                "teacher_family": "masked_mdlm",
                "destination": "official/bad",
                "license": "test",
                "terms_url": "https://example.test/terms",
                "digest": {"policy": "sha256", "sha256": first},
                "required_files": ["config.json", "model.safetensors"],
                "source": {
                    "repo_id": "owner/model",
                    "revision": "a" * 40,
                    "allow_patterns": ["config.json", "model.safetensors"],
                },
            }
        )


def test_gdrive_path_to_file_id_inventory_is_one_to_one():
    """Catch one mutable Drive object being relabeled as multiple declared files."""

    with pytest.raises(ValueError, match="unique"):
        GDriveSource(
            folder_id="folder123",
            expected_files={"model.bin": "sameID", "config.json": "sameID"},
        )


def test_known_primary_sources_and_immutable_hf_revisions_are_recorded(checkpoints):
    """Catch moving Hub references or omission of a required upstream release."""

    expected_hf = {
        "flm_lm1b_hf": ("david3684/FLM-B-LM1B", "cd0ca433d11d9392494de8122f3b63cc996fd4ce"),
        "fmlm_lm1b_hf": ("david3684/FMLM-B-LM1B", "356592ce2d9af4adbf9d7b55e8e00bbfb37eb4c8"),
        "flm_owt_hf": ("david3684/FLM-B-OWT", "624471b934fdd0421757d62290f7e639f32566d3"),
        "fmlm_owt_hf": ("david3684/FMLM-B-OWT", "483ea1b38bba56632cd40dc5a3c70a2340bb4946"),
        "langflow_owt_hf": (
            "Continuous-Rivals-Discrete/langflow-owt",
            "a08f933dd337d52762fec5ef7d60c131896cc341",
        ),
        "duo_owt_hf": ("s-sahoo/duo", "ef8ad7a25a4bf9624306230913dd7ae5cb2a00bd"),
        "duo_dcd_owt_hf": (
            "s-sahoo/duo-distilled",
            "08ee71faabfc0e2eaaa6cb1833ed3a486931ef2b",
        ),
        "mdlm_owt_hf": (
            "kuleshov-group/mdlm-owt",
            "d0958fa851335ece6c15260ce0025f030673c0fb",
        ),
        "sdtt_owt_hf": ("jdeschena/sdtt", "c84bdf45ce3a80f5a70bb09548093f1c8f2ac7a7"),
    }
    for resource_name, (repo_id, revision) in expected_hf.items():
        source = checkpoints.resources[resource_name].source
        assert source.repo_id == repo_id
        assert source.revision == revision
        assert source.allow_patterns

    flm_drive = checkpoints.resources["flm_lm1b_reproductions"].source
    assert flm_drive.folder_id == (
        "1TJO3aFWqI7ukbmjciZ6krAUFlAak1itl"
    )
    assert flm_drive.expected_files == {
        "lm1b_CANDI.ckpt": "19TbJu1VHWX5EHqE8Si4xWqBhJlTqaGjf",
        "lm1b_Duo.ckpt": "1yHtaYrLX34rS5tu9V1SKF5NvwTVxPWoF",
        "lm1b_MDLM_.ckpt": "1jp6Jka8tQJJ9zljihdLErokFp9TDvlsl",
    }
    rdlm_drive = checkpoints.resources["rdlm_lm1b_drive"].source
    assert rdlm_drive.folder_id == (
        "1aDTZtPIxAxQrkaRSahjuWbkxX1OYq9CC"
    )
    assert rdlm_drive.expected_files["LM1B/checkpoint.pth"] == (
        "1zAqP-DtZigoiChpNfgJgJobKIZ9VOaro"
    )
    assert checkpoints.resources["di4c_mdlm_owt_zenodo"].source.record_id == 15124163
    expected_reproductions = {
        ("duo", "lm1b"): ("lm1b_Duo.ckpt", "uniform_duo"),
        ("mdlm", "lm1b"): ("lm1b_MDLM_.ckpt", "masked_mdlm"),
        ("candi", "lm1b"): ("lm1b_CANDI.ckpt", "hybrid_candi"),
    }
    for cell, (path, teacher_family) in expected_reproductions.items():
        coverage = checkpoints.coverage[cell]
        assert coverage.resource == "flm_lm1b_reproductions"
        assert coverage.path == path
        assert coverage.teacher_family == teacher_family


def test_coverage_preserves_registry_provenance_and_teacher_family(registry, checkpoints):
    """Catch official/reproduction conflation or a distilled cell using the wrong teacher."""

    expected_teachers = {
        "duo": "uniform_duo",
        "duo_dcd": "uniform_duo",
        "duo_di4c": "uniform_duo",
        "mdlm": "masked_mdlm",
        "mdlm_sdtt": "masked_mdlm",
        "mdlm_di4c": "masked_mdlm",
        "flm": "continuous_flm",
        "fmlm": "continuous_flm",
        "langflow": "continuous_langflow",
        "candi": "hybrid_candi",
        "rdlm": "continuous_rdlm",
    }
    for cell, coverage in checkpoints.coverage.items():
        model, dataset = cell
        support = registry.models[model].datasets[dataset]
        resource = checkpoints.resources[coverage.resource]
        assert resource.provenance == support.provenance
        assert (coverage.teacher_family or resource.teacher_family) == expected_teachers[model]

    for model_name, model in registry.models.items():
        for support in model.datasets.values():
            if support.train_recipe:
                recipe = checkpoints.recipes[support.train_recipe]
                assert recipe.teacher_family == expected_teachers[model_name]

    masked_di4c = checkpoints.resources["di4c_mdlm_owt_zenodo"]
    assert masked_di4c.teacher_family == "masked_mdlm"
    assert checkpoints.coverage[("mdlm_di4c", "owt")].resource == masked_di4c.id
    assert all(
        coverage.resource != masked_di4c.id
        for cell, coverage in checkpoints.coverage.items()
        if cell[0] == "duo_di4c"
    )


def test_training_recipes_are_explicit_and_pinned(checkpoints, registry):
    """Catch a recipe fallback that cannot identify its source revision and command."""

    recipe_ids = {
        support.train_recipe
        for model in registry.models.values()
        for support in model.datasets.values()
        if support.train_recipe
    }
    assert recipe_ids
    assert recipe_ids <= set(checkpoints.recipes)
    for recipe_id in recipe_ids:
        recipe = checkpoints.recipes[recipe_id]
        assert len(recipe.source_commit) == 40
        assert recipe.command.strip()
        assert recipe.command.startswith(("bash scripts/train/", "bash scripts/distill/"))
        assert recipe.output.startswith("checkpoints/")


def test_candi_and_duo_dcd_use_project_owned_task12_wrappers(checkpoints):
    """Catch nonexistent clone paths and upstream jobs that ignore declared outputs."""

    candi = checkpoints.recipes["candi_owt"].command
    assert candi.startswith("bash scripts/train/candi.sh")
    assert "--source upstreams/candi" in candi
    assert "--dataset owt" in candi
    assert "--output checkpoints/reference_reproduction/candi/owt" in candi
    assert "slurm" not in candi.lower()

    dcd = checkpoints.recipes["duo_dcd_lm1b"].command
    assert dcd.startswith("bash scripts/distill/duo_dcd.sh")
    assert "--source upstreams/duo" in dcd
    assert "--dataset lm1b" in dcd
    assert "--teacher checkpoints/reference_reproduction/flm_baselines/lm1b/lm1b_Duo.ckpt" in dcd
    assert "--output checkpoints/reference_reproduction/duo_dcd/lm1b" in dcd
    assert "--rounds 8" in dcd
    assert "--steps-per-round 10000" in dcd
    assert "--global-batch-size 128" in dcd
    assert "--learning-rate 6e-5" in dcd


def test_distillation_recipes_pin_the_declared_teacher_family(checkpoints):
    """Catch recipes falling back to default MDLM teachers or non-Di4C training."""

    dcd = checkpoints.recipes["duo_dcd_lm1b"].command
    assert dcd.startswith("bash scripts/distill/duo_dcd.sh")
    assert "--teacher " in dcd
    assert "lm1b_Duo.ckpt" in dcd

    for recipe_id in ("duo_di4c_lm1b", "duo_di4c_owt"):
        recipe = checkpoints.recipes[recipe_id]
        command = recipe.command
        assert recipe.teacher_adapter == "uniform_to_absorbing"
        assert command.startswith("bash scripts/distill/di4c.sh")
        assert "is_di4c=true" in command
        assert "--teacher-family uniform_duo" in command
        assert "Duo.ckpt" in command or "duo.ckpt" in command
        assert "MDLM" not in command
        assert "python src/sdtt/main.py" not in command

    masked_di4c = checkpoints.recipes["mdlm_di4c_lm1b"]
    assert masked_di4c.teacher_adapter == "masked_to_absorbing"
    assert masked_di4c.command.startswith("bash scripts/distill/di4c.sh")
    assert "is_di4c=true" in masked_di4c.command
    assert "--teacher-family masked_mdlm" in masked_di4c.command
    assert "lm1b_MDLM_.ckpt" in masked_di4c.command
    assert "python src/sdtt/main.py" not in masked_di4c.command

    masked_sdtt = checkpoints.recipes["mdlm_sdtt_lm1b"]
    assert masked_sdtt.teacher_adapter == "masked_to_absorbing"
    assert masked_sdtt.command.startswith("bash scripts/distill/mdlm_sdtt.sh")
    assert "--teacher-family masked_mdlm" in masked_sdtt.command
    assert "lm1b_MDLM_.ckpt" in masked_sdtt.command
    assert "python src/sdtt/main.py" not in masked_sdtt.command

    for recipe_id in (
        "duo_dcd_lm1b",
        "duo_di4c_lm1b",
        "mdlm_sdtt_lm1b",
        "mdlm_di4c_lm1b",
    ):
        command = checkpoints.recipes[recipe_id].command
        assert "lm1b" in command
        assert "--output " in command


def test_train_recipe_is_typed_and_forbidden_on_unsupported_cells():
    """Catch recipes attached to unsupported cells or blank recipe identifiers."""

    supported = DatasetSupport(
        status="supported",
        provenance="reference_reproduction",
        train_recipe="duo_lm1b",
    )
    assert supported.train_recipe == "duo_lm1b"
    with pytest.raises(ValueError, match="unsupported datasets"):
        DatasetSupport(
            status="unsupported",
            reason="No public or reproducible experiment is defined.",
            train_recipe="not_allowed",
        )
    with pytest.raises(ValueError):
        DatasetSupport(
            status="supported",
            provenance="self_trained",
            train_recipe=" ",
        )


def test_json_schema_accepts_recipes_and_rejects_them_on_unsupported_cells():
    """Catch drift between the Pydantic and JSON Schema recipe contracts."""

    document = yaml.safe_load((ROOT / "configs" / "experiments.yaml").read_text())
    schema = json.loads((ROOT / "configs" / "schema.json").read_text())
    validator = Draft202012Validator(schema)
    assert list(validator.iter_errors(document)) == []

    invalid = copy.deepcopy(document)
    invalid["models"]["langflow"]["datasets"]["lm1b"]["train_recipe"] = "bad"
    assert list(validator.iter_errors(invalid))


def test_fetch_dry_run_enumerates_public_resources_and_recipes_without_writes(tmp_path):
    """Catch dry-run network/filesystem side effects or omitted acquisition fallbacks."""

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "fetch_checkpoints.py"),
            "--root",
            str(tmp_path),
            "--config",
            str(ROOT / "artifacts" / "checkpoints.yaml"),
            "--registry",
            str(ROOT / "configs" / "experiments.yaml"),
            "--dry-run",
            "--all-public",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    manifest = load_checkpoint_manifest(ROOT / "artifacts" / "checkpoints.yaml")
    assert sum(line.startswith("RESOURCE ") for line in lines) == len(manifest.resources)
    assert sum(line.startswith("RECIPE ") for line in lines) == len(manifest.recipes)
    assert any("huggingface" in line and "official/flm/lm1b" in line for line in lines)
    assert any("gdrive" in line and "official/rdlm/lm1b" in line for line in lines)
    assert any("zenodo" in line and "official/mdlm_di4c/owt" in line for line in lines)
    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "artifacts" / "checkpoint_lock.json").exists()

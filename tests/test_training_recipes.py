from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import dlb.recipes as recipes_module
from dlb.recipes import (
    BACKBONE_TENSOR_KEYS,
    RecipeError,
    build_launch,
    load_recipe,
    masked_to_absorbing,
    uniform_to_absorbing,
    validate_effective_batch,
)


ROOT = Path(__file__).parents[1]


def override(command: tuple[str, ...], key: str) -> str:
    prefix = key + "="
    values = [argument[len(prefix) :] for argument in command if argument.startswith(prefix)]
    assert len(values) == 1, (key, command)
    return values[0]


def test_flm_recipe_matches_paper_v3_reference() -> None:
    recipe = load_recipe("flm", "lm1b")

    assert recipe.max_steps == 1_000_000
    assert recipe.global_batch_size == 512
    assert recipe.learning_rate == pytest.approx(3e-4)
    assert recipe.warmup_steps == 2_500
    assert recipe.optimizer == "adam"
    assert recipe.sequence_length == 128
    assert recipe.evidence == "paper_v3_main"


def test_fmlm_recipe_uses_main_text_ce_schedule_not_release_drift() -> None:
    recipe = load_recipe("fmlm", "lm1b")

    assert recipe.max_steps == 100_000
    assert recipe.global_batch_size == 512
    assert recipe.learning_rate == pytest.approx(3e-4)
    assert recipe.warmup_steps == 2_500
    assert recipe.distillation_loss == "cross_entropy"
    assert recipe.release_script_max_steps == 1_000_000


def test_flm_release_script_drift_is_explicit() -> None:
    recipe = load_recipe("flm", "owt")

    assert recipe.max_steps == 1_000_000
    assert recipe.release_script_max_steps == 1_500_000
    assert recipe.release_evidence == "official_release_script"
    assert "trainer.max_steps=1500000" in (
        ROOT / "upstreams" / "flm" / "scripts" / "train_lm1b_flm.sh"
    ).read_text()
    assert "trainer.max_steps=1000000" in (
        ROOT / "upstreams" / "flm" / "scripts" / "train_lm1b_fmlm_denoiser.sh"
    ).read_text()


def test_sdtt_and_dcd_round_schedule() -> None:
    for name in ("mdlm_sdtt", "duo_dcd"):
        recipe = load_recipe(name, "lm1b")
        assert recipe.rounds == 8
        assert recipe.steps_per_round == 10_000
        assert recipe.max_steps == 80_000
        assert recipe.global_batch_size == 128
        assert recipe.learning_rate == pytest.approx(6e-5)
        assert recipe.warmup_steps == 2_500


def test_di4c_sampling_checkpoints_match_reference_selection() -> None:
    assert load_recipe("duo_di4c", "lm1b").sampling_step == 20_000
    assert load_recipe("mdlm_di4c", "lm1b").sampling_step == 20_000
    assert load_recipe("duo_di4c", "owt").sampling_step == 50_000


@pytest.mark.parametrize(
    ("model", "dataset"),
    [
        ("flm", "lm1b"),
        ("duo", "lm1b"),
        ("mdlm", "lm1b"),
        ("candi", "owt"),
        ("rdlm", "lm1b"),
        ("fmlm", "lm1b"),
        ("duo_dcd", "lm1b"),
        ("mdlm_sdtt", "lm1b"),
        ("duo_di4c", "lm1b"),
        ("mdlm_di4c", "lm1b"),
    ],
)
def test_rendered_commands_use_real_entrypoints_and_absolute_project_paths(
    model: str, dataset: str
) -> None:
    recipe = load_recipe(model, dataset)
    teacher = ROOT / "checkpoints" / "fixture" / "teacher.ckpt" if recipe.teacher_family else None
    devices = 2 if model.endswith("_di4c") else 8
    launch = build_launch(
        recipe,
        root=ROOT,
        source=ROOT / recipe.source_path,
        output=ROOT / "checkpoints" / "self_trained" / dataset / model,
        teacher=teacher,
        devices=devices,
        nodes=1,
        per_device_batch_size=recipe.default_per_device_batch_size,
        seed=42,
        resume=True,
    )

    assert Path(launch.command[0]).is_absolute()
    assert Path(launch.entrypoint).is_absolute()
    assert Path(launch.cwd).is_absolute()
    assert Path(launch.data_path).is_absolute()
    assert Path(launch.output).is_absolute()
    assert all("${" not in argument for argument in launch.command)
    assert override(launch.command, recipe.output_override) == str(launch.output)
    assert override(launch.command, recipe.max_steps_override) == str(recipe.max_steps)
    assert override(launch.command, recipe.learning_rate_override) == str(
        recipe.learning_rate
    )
    assert launch.effective_global_batch_size == recipe.global_batch_size


def test_rendered_hydra_overrides_exist_in_pinned_upstream_configs() -> None:
    flm = build_launch(
        load_recipe("flm", "lm1b"),
        root=ROOT,
        source=ROOT / "upstreams" / "flm",
        output=ROOT / "checkpoints" / "self_trained" / "lm1b" / "flm",
        teacher=None,
        devices=8,
        nodes=1,
        per_device_batch_size=64,
        seed=42,
        resume=True,
    )
    assert override(flm.command, "data") == "lm1b-wrap"
    assert override(flm.command, "algo") == "flm"
    assert override(flm.command, "model.length") == "128"
    assert override(flm.command, "lr_scheduler.num_warmup_steps") == "2500"
    assert override(flm.command, "trainer.accumulate_grad_batches") == "1"
    assert override(flm.command, "wandb") == "null"


def test_all_manual_recipes_render_unique_site_independent_overrides() -> None:
    models = (
        "flm",
        "duo",
        "mdlm",
        "candi",
        "fmlm",
        "duo_dcd",
        "mdlm_sdtt",
        "duo_di4c",
        "mdlm_di4c",
    )
    for model in models:
        for dataset in ("lm1b", "owt"):
            recipe = load_recipe(model, dataset)
            devices = 2 if model.endswith("_di4c") else 8
            teacher = (
                ROOT / "checkpoints" / "fixture" / f"{recipe.teacher_family}.ckpt"
                if recipe.teacher_family
                else None
            )
            launch = build_launch(
                recipe,
                root=ROOT,
                source=ROOT / recipe.source_path,
                output=ROOT / "checkpoints" / "self_trained" / dataset / model,
                teacher=teacher,
                devices=devices,
                nodes=1,
                per_device_batch_size=recipe.default_per_device_batch_size,
                seed=42,
                resume=True,
            )
            keys = [argument.split("=", 1)[0].lstrip("+") for argument in launch.command[3:]]
            assert len(keys) == len(set(keys)), (model, dataset, keys)
            rendered = "\n".join(launch.command)
            assert "/home/" not in rendered
            assert "/share/" not in rendered
            assert "REDACTED" not in rendered
            assert launch.entrypoint.is_file()

    rdlm = load_recipe("rdlm", "lm1b")
    launch = build_launch(
        rdlm,
        root=ROOT,
        source=ROOT / rdlm.source_path,
        output=ROOT / "checkpoints" / "self_trained" / "lm1b" / "rdlm",
        teacher=None,
        devices=8,
        nodes=1,
        per_device_batch_size=64,
        seed=42,
        resume=True,
    )
    keys = [argument.split("=", 1)[0].lstrip("+") for argument in launch.command[3:]]
    assert len(keys) == len(set(keys))
    assert "/home/" not in "\n".join(launch.command)


def test_effective_global_batch_must_be_exact_and_divisible() -> None:
    assert validate_effective_batch(512, devices=8, nodes=1, per_device_batch_size=16) == 4
    with pytest.raises(RecipeError, match="divisible"):
        validate_effective_batch(512, devices=6, nodes=1, per_device_batch_size=32)
    with pytest.raises(RecipeError, match="positive"):
        validate_effective_batch(512, devices=0, nodes=1, per_device_batch_size=32)


def tensor_state(vocab: int = 3, hidden: int = 2) -> dict[str, object]:
    return {
        "vocab_embed.embedding": [
            [10 * row + column + 1 for column in range(hidden)] for row in range(vocab)
        ],
        "output_layer.linear.weight": [
            [100 + 10 * row + column for column in range(hidden)] for row in range(vocab)
        ],
        "output_layer.linear.bias": [200 + row for row in range(vocab)],
    }


def test_uniform_teacher_appends_zero_absorbing_rows() -> None:
    state = tensor_state()
    adapted = uniform_to_absorbing(state, expected_keys=set(BACKBONE_TENSOR_KEYS))

    assert adapted["vocab_embed.embedding"] == [[1, 2], [11, 12], [21, 22], [0, 0]]
    assert adapted["output_layer.linear.weight"][-1] == [0, 0]
    assert adapted["output_layer.linear.bias"] == [200, 201, 202, 0]
    assert state["vocab_embed.embedding"] == [[1, 2], [11, 12], [21, 22]]


def test_masked_teacher_moves_existing_mask_row_to_final_absorbing_state() -> None:
    state = tensor_state()
    adapted = masked_to_absorbing(
        state,
        source_mask_index=1,
        target_vocab_size=4,
        expected_keys=set(BACKBONE_TENSOR_KEYS),
    )

    assert adapted["vocab_embed.embedding"] == [[1, 2], [0, 0], [21, 22], [11, 12]]
    assert adapted["output_layer.linear.weight"][1] == [0, 0]
    assert adapted["output_layer.linear.weight"][-1] == [110, 111]
    assert adapted["output_layer.linear.bias"] == [200, 0, 202, 201]


def test_masked_teacher_does_not_duplicate_an_already_final_mask() -> None:
    state = tensor_state(vocab=4)
    adapted = masked_to_absorbing(
        state,
        source_mask_index=3,
        target_vocab_size=4,
        expected_keys=set(BACKBONE_TENSOR_KEYS),
    )
    assert adapted == state
    assert adapted is not state


def test_teacher_adapters_allow_rotary_buffer_from_pinned_checkpoints() -> None:
    state = tensor_state()
    state["rotary_emb.inv_freq"] = [1, 2]

    adapted = masked_to_absorbing(
        state,
        source_mask_index=1,
        target_vocab_size=4,
        expected_keys=set(BACKBONE_TENSOR_KEYS),
    )

    assert adapted["rotary_emb.inv_freq"] == [1, 2]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda state: state.pop("output_layer.linear.bias"), "keys"),
        (lambda state: state.update({"unexpected": [1]}), "keys"),
        (
            lambda state: state.update({"output_layer.linear.weight": [[1], [2], [3]]}),
            "hidden",
        ),
    ],
)
def test_teacher_adapters_fail_closed_on_key_or_shape_mismatch(mutation, match: str) -> None:
    state = tensor_state()
    mutation(state)
    with pytest.raises(RecipeError, match=match):
        uniform_to_absorbing(state, expected_keys=set(BACKBONE_TENSOR_KEYS))


@pytest.mark.parametrize(
    "wrapper",
    [
        "scripts/train/flm.sh",
        "scripts/train/duo.sh",
        "scripts/train/mdlm.sh",
        "scripts/train/candi.sh",
        "scripts/train/rdlm.sh",
        "scripts/distill/fmlm.sh",
        "scripts/distill/duo_dcd.sh",
        "scripts/distill/mdlm_sdtt.sh",
        "scripts/distill/di4c.sh",
    ],
)
def test_every_wrapper_supports_help(wrapper: str) -> None:
    completed = subprocess.run(
        ["bash", str(ROOT / wrapper), "--help"],
        cwd=ROOT,
        env={**os.environ, "DLB_PYTHON": sys.executable},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--dry-run" in completed.stdout


def test_manifest_prerequisite_commands_dry_run_without_external_assets() -> None:
    commands = [
        [
            "bash",
            "scripts/train/candi.sh",
            "--source",
            "upstreams/candi",
            "--dataset",
            "owt",
            "--output",
            "checkpoints/reference_reproduction/candi/owt",
            "--dry-run",
        ],
        [
            "bash",
            "scripts/distill/duo_dcd.sh",
            "--source",
            "upstreams/duo",
            "--dataset",
            "lm1b",
            "--teacher",
            "checkpoints/reference_reproduction/flm_baselines/lm1b/lm1b_Duo.ckpt",
            "--output",
            "checkpoints/reference_reproduction/duo_dcd/lm1b",
            "--rounds",
            "8",
            "--steps-per-round",
            "10000",
            "--global-batch-size",
            "128",
            "--learning-rate",
            "6e-5",
            "--dry-run",
        ],
    ]
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "DLB_PYTHON": sys.executable},
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        record = json.loads(completed.stdout)
        assert record["dry_run"] is True
        assert record["command"]
        assert Path(record["output"]).is_absolute()


def test_fake_launch_publishes_before_run_and_resumes_exact_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = load_recipe("flm", "lm1b")
    launch = build_launch(
        recipe,
        root=tmp_path,
        source=tmp_path / "upstreams" / "flm",
        output=tmp_path / "checkpoints" / "self_trained" / "lm1b" / "flm",
        teacher=None,
        devices=8,
        nodes=1,
        per_device_batch_size=64,
        seed=42,
        resume=True,
    )
    monkeypatch.setattr(recipes_module, "_verify_source", lambda item: None)
    monkeypatch.setattr(
        recipes_module,
        "_prepare_training_cache",
        lambda item, root: {
            "mode": "fake",
            "processed_manifest_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(
        recipes_module, "_compose_config", lambda item, environment: "mode: train\n"
    )
    calls = 0

    def fake_run(command, **kwargs):
        nonlocal calls
        calls += 1
        assert (launch.output / "recipe.json").is_file()
        assert (launch.output / "launch_argv.json").is_file()
        assert (launch.output / "provenance.json").is_file()
        checkpoint = launch.output / "checkpoints" / "last.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"fake checkpoint")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(recipes_module.subprocess, "run", fake_run)

    assert recipes_module.execute_launch(launch, root=tmp_path) == "completed"
    assert recipes_module.execute_launch(launch, root=tmp_path) == "skipped"
    assert calls == 1
    completed = json.loads((launch.output / "completed.json").read_text())
    assert completed["checkpoint"] == "model.ckpt"
    assert (launch.output / "model.ckpt").read_bytes() == b"fake checkpoint"

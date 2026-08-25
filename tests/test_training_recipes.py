from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import types
import importlib.util

import pytest

import dlb.recipes as recipes_module
from dlb.recipes import (
    BACKBONE_TENSOR_KEYS,
    RecipeError,
    build_launch,
    expected_backbone_keys,
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


@pytest.mark.parametrize(
    ("dataset", "cache_name"),
    [
        ("lm1b", "bert-base-uncased.pkl"),
        ("owt", "gpt2.pkl"),
    ],
)
def test_duo_dcd_uses_repo_integral_cache_not_tokenizer_path(
    dataset: str, cache_name: str
) -> None:
    launch = build_launch(
        load_recipe("duo_dcd", dataset),
        root=ROOT,
        source=ROOT / "upstreams" / "duo",
        output=ROOT / "checkpoints" / "self_trained" / dataset / "duo_dcd",
        teacher=ROOT / "checkpoints" / "fixture" / "uniform_duo.ckpt",
        devices=8,
        nodes=1,
        per_device_batch_size=16,
        seed=42,
        resume=True,
    )

    integral_cache = override(launch.command, "algo.integral_cache_path")
    curriculum_integral_cache = override(
        launch.command, "+algo.curriculum.integral_cache_path"
    )
    assert integral_cache == str(
        (ROOT / "upstreams" / "duo" / "integral" / cache_name).absolute()
    )
    assert curriculum_integral_cache == integral_cache
    assert "snapshots" not in integral_cache
    assert override(launch.command, "+algo.curriculum.mode") == "simple"
    assert override(launch.command, "+algo.curriculum.gamma_min") == "-4"
    assert override(launch.command, "+algo.curriculum.gamma_max") == "-1"
    assert Path(launch.command[2]).name == "train_duo_dcd.py"


def test_duo_dcd_wrapper_accepts_trainer_base_nll_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "train_duo_dcd_for_test", ROOT / "adapters" / "train_duo_dcd.py"
    )
    assert spec is not None and spec.loader is not None
    train_duo_dcd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_duo_dcd)

    calls = []

    class Distillation:
        def nll(
            self,
            x0,
            output_tokens,
            current_accumulation_step=None,
            train_mode=None,
        ):
            calls.append(
                (x0, output_tokens, current_accumulation_step, train_mode)
            )
            return "loss"

    algo = type(sys)("algo")
    algo.Distillation = Distillation
    monkeypatch.setitem(sys.modules, "algo", algo)

    train_duo_dcd._patch_distillation_nll()

    assert Distillation().nll("x0", None, "output", "accum", True) == "loss"
    assert calls == [("x0", "output", "accum", True)]


def test_di4c_sampling_checkpoints_match_reference_selection() -> None:
    assert load_recipe("duo_di4c", "lm1b").sampling_step == 20_000
    assert load_recipe("mdlm_di4c", "lm1b").sampling_step == 20_000
    assert load_recipe("duo_di4c", "owt").sampling_step == 50_000


def test_lm1b_di4c_uses_project_wrapper_and_teacher_student_init() -> None:
    recipe = load_recipe("duo_di4c", "lm1b")
    launch = build_launch(
        recipe,
        root=ROOT,
        source=ROOT / "upstreams" / "di4c",
        output=ROOT / "checkpoints" / "reference_reproduction" / "duo_di4c" / "lm1b",
        teacher=ROOT / "checkpoints" / "fixture" / "uniform_duo.ckpt",
        devices=2,
        nodes=1,
        per_device_batch_size=1,
        seed=42,
        resume=True,
    )

    assert Path(launch.command[2]).name == "train_di4c.py"
    assert override(launch.command, "+dlb_student_init") == "teacher"


def test_di4c_student_init_can_be_overridden_to_scratch() -> None:
    recipe = load_recipe("mdlm_di4c", "lm1b")
    launch = build_launch(
        recipe,
        root=ROOT,
        source=ROOT / "upstreams" / "di4c",
        output=ROOT / "checkpoints" / "reference_reproduction" / "mdlm_di4c" / "lm1b",
        teacher=ROOT / "checkpoints" / "fixture" / "masked_mdlm.ckpt",
        devices=2,
        nodes=1,
        per_device_batch_size=1,
        seed=42,
        resume=True,
        di4c_student_init="scratch",
    )

    assert Path(launch.command[2]).name == "train_di4c.py"
    assert override(launch.command, "+dlb_student_init") == "scratch"


def test_di4c_wrapper_registers_hydra_resolvers_before_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adapters import train_di4c

    registered = {}

    class FakeOmegaConf:
        @staticmethod
        def register_new_resolver(name, resolver, *, replace=False):
            registered[name] = (resolver, replace)

    monkeypatch.setitem(
        sys.modules,
        "omegaconf",
        types.SimpleNamespace(OmegaConf=FakeOmegaConf),
    )
    train_di4c._register_omegaconf_resolvers()

    assert set(registered) == {"cwd", "device_count", "div_up", "eval"}
    assert all(replace for _, replace in registered.values())
    assert registered["div_up"][0](5, 2) == 3
    assert registered["eval"][0]("1 + 1") == 2
    assert registered["eval"][0]("len([1, 2, 3])") == 3
    assert registered["eval"][0]("__import__('math').ceil(1.2)") == 2
    assert isinstance(registered["cwd"][0](), str)
    assert isinstance(registered["device_count"][0](), int)


def test_di4c_wrapper_adds_original_script_directory_to_import_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adapters import train_di4c

    source = tmp_path / "upstreams" / "di4c"
    package_root = source / "sdtt" / "src" / "sdtt"
    package_root.mkdir(parents=True)
    (package_root / "main.py").write_text("# fixture\n", encoding="utf-8")
    monkeypatch.chdir(source)
    before = list(sys.path)

    try:
        train_di4c._install_upstream_path()

        assert str(package_root) in sys.path
        assert sys.path.index(str(package_root)) < sys.path.index(
            str(source / "sdtt" / "src")
        )
    finally:
        sys.path[:] = before


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


def test_uniform_teacher_zeroes_existing_absorbing_row() -> None:
    state = tensor_state(vocab=4)

    adapted = uniform_to_absorbing(
        state,
        target_vocab_size=4,
        expected_keys=set(BACKBONE_TENSOR_KEYS),
    )

    assert adapted["vocab_embed.embedding"] == [[1, 2], [11, 12], [21, 22], [0, 0]]
    assert adapted["output_layer.linear.weight"][-1] == [0, 0]
    assert adapted["output_layer.linear.bias"] == [200, 201, 202, 0]
    assert state["vocab_embed.embedding"][-1] == [31, 32]


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


def test_adapted_teacher_checkpoint_drops_lightning_metadata_for_weights_only_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = load_recipe("mdlm_di4c", "lm1b")
    launch = build_launch(
        recipe,
        root=tmp_path,
        source=tmp_path / recipe.source_path,
        output=tmp_path / "checkpoints" / "reference_reproduction" / "mdlm_di4c",
        teacher=tmp_path / "teacher.ckpt",
        devices=1,
        nodes=1,
        per_device_batch_size=1,
        seed=42,
        resume=True,
    )
    launch.teacher.write_bytes(b"source checkpoint")
    state = tensor_state()
    for key in expected_backbone_keys() - set(state):
        state[key] = [0]
    source_payload = {
        "state_dict": state,
        "hyper_parameters": {"config": object()},
        "callbacks": {"custom": object()},
    }
    saved_payloads: list[dict[str, object]] = []

    class FakeTorch:
        @staticmethod
        def load(path, **kwargs):
            return source_payload

        @staticmethod
        def save(payload, path):
            saved_payloads.append(payload)
            Path(path).write_bytes(b"adapted checkpoint")

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(recipes_module.BASE_VOCAB_SIZES, "lm1b", 3)

    recipes_module.adapt_teacher_checkpoint(launch, source_mask_index=1)

    assert len(saved_payloads) == 1
    assert set(saved_payloads[0]) == {"state_dict"}
    assert "hyper_parameters" not in saved_payloads[0]


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


def test_distilled_launch_canonicalizes_lightning_step_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = load_recipe("duo_di4c", "owt")
    teacher = tmp_path / "checkpoints" / "fixture" / "uniform_duo.ckpt"
    teacher.parent.mkdir(parents=True)
    teacher.write_bytes(b"fake teacher")
    launch = build_launch(
        recipe,
        root=tmp_path,
        source=tmp_path / "upstreams" / "di4c",
        output=tmp_path / "checkpoints" / "reference_reproduction" / "duo_di4c" / "owt",
        teacher=teacher,
        devices=2,
        nodes=1,
        per_device_batch_size=1,
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
    monkeypatch.setattr(
        recipes_module,
        "adapt_teacher_checkpoint",
        lambda item: {"mode": "fake_teacher_adapter"},
    )

    def fake_run(command, **kwargs):
        checkpoint = launch.output / "checkpoints" / "0-50000.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"fake distilled checkpoint")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(recipes_module.subprocess, "run", fake_run)

    assert recipes_module.execute_launch(launch, root=tmp_path) == "completed"
    completed = json.loads((launch.output / "completed.json").read_text())
    assert completed["checkpoint"] == "student_checkpoints/50000.ckpt"
    assert (launch.output / completed["checkpoint"]).read_bytes() == (
        b"fake distilled checkpoint"
    )


def test_training_cache_bridge_adds_attention_mask_for_duo_text_batches(
    tmp_path: Path,
) -> None:
    datasets = pytest.importorskip("datasets")
    Dataset = datasets.Dataset
    load_from_disk = datasets.load_from_disk

    processed = tmp_path / "data" / "processed" / "lm1b-bert-128"
    (tmp_path / "data" / "manifests").mkdir(parents=True)
    for split in ("train", "validation"):
        Dataset.from_dict(
            {"input_ids": [[101] + [7] * 126 + [102]]}
        ).save_to_disk(str(processed / split))
    (tmp_path / "data" / "manifests" / "lm1b.json").write_text(
        json.dumps({"dataset": "lm1b", "sequence_length": 128}),
        encoding="utf-8",
    )

    launch = build_launch(
        load_recipe("duo_dcd", "lm1b"),
        root=tmp_path,
        source=tmp_path / "upstreams" / "duo",
        output=tmp_path / "checkpoints" / "reference_reproduction" / "duo_dcd" / "lm1b",
        teacher=tmp_path / "checkpoints" / "fixture" / "uniform_duo.ckpt",
        devices=1,
        nodes=1,
        per_device_batch_size=128,
        seed=42,
        resume=True,
    )

    record = recipes_module._prepare_training_cache(launch, tmp_path)
    cached = load_from_disk(
        str(launch.data_path / "lm1b_train_bs128_wrapped.dat")
    )

    assert record["format"] == "hf_dataset_with_attention_mask_v1"
    assert set(cached.column_names) == {"input_ids", "attention_mask"}
    assert cached[0]["attention_mask"] == [1] * 128


def test_source_verification_ignores_python_bytecode_cache_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = load_recipe("mdlm_di4c", "lm1b")
    source = tmp_path / "upstreams" / "di4c"
    entrypoint = source / recipe.entrypoint
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("print('placeholder')\n", encoding="utf-8")
    launch = build_launch(
        recipe,
        root=tmp_path,
        source=source,
        output=tmp_path / "checkpoints" / "reference_reproduction" / "mdlm_di4c",
        teacher=tmp_path / "teacher.ckpt",
        devices=1,
        nodes=1,
        per_device_batch_size=1,
        seed=42,
        resume=True,
    )

    def fake_run(command, **kwargs):
        if command[3] == "rev-parse":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=recipe.source_commit + "\n",
                stderr="",
            )
        if command[3] == "status":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="?? sdtt/src/sdtt/__pycache__/\n?? sdtt/src/sdtt/core/__pycache__/\n",
                stderr="",
            )
        raise AssertionError(command)

    monkeypatch.setattr(recipes_module.subprocess, "run", fake_run)

    recipes_module._verify_source(launch)


def test_training_launch_disables_python_bytecode_writes(
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

    def fake_compose(item, environment):
        assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
        return "mode: train\n"

    def fake_run(command, **kwargs):
        assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        checkpoint = launch.output / "checkpoints" / "last.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"fake checkpoint")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(recipes_module, "_compose_config", fake_compose)
    monkeypatch.setattr(recipes_module.subprocess, "run", fake_run)

    assert recipes_module.execute_launch(launch, root=tmp_path) == "completed"

"""Contracts for the pinned SDTT and Di4C distilled language samplers."""

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

from dlb.adapters.base import AdapterError
from dlb.adapters.di4c import Di4CAdapter
from dlb.adapters.sdtt import SDTTAdapter
from dlb.command import main as command_main
from dlb.runner import RunRequest, _resolve_checkpoint_provenance


ROOT = Path(__file__).parents[1]


def load_runtime_module():
    path = ROOT / "adapters/_distilled_runtime.py"
    specification = importlib.util.spec_from_file_location("dlb_distilled_runtime", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def request(model: str, dataset: str, steps: int = 8, samples: int = 17) -> RunRequest:
    return RunRequest(
        run_id=f"{model}-{dataset}-steps-{steps}",
        model_id=model,
        dataset_id=dataset,
        step_count=steps,
        seed=42,
        sample_count=samples,
    )


def run_dir(item: RunRequest) -> Path:
    return (
        ROOT
        / "results"
        / "samples"
        / item.dataset_id
        / item.model_id
        / f"steps_{item.step_count}"
    )


def option(command: list[str], key: str) -> str:
    index = command.index(key)
    return command[index + 1]


def prepare_tokenizer_binding(
    root: Path, dataset: str
) -> tuple[Path, Path, Path]:
    """Create the minimal immutable tokenizer lock consumed by wrapper tests."""

    data_config = root / "artifacts" / "data.yaml"
    data_config.parent.mkdir(parents=True, exist_ok=True)
    data_config.write_text(
        (ROOT / "artifacts/data.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    document = yaml.safe_load(data_config.read_text(encoding="utf-8"))
    tokenizer_id = document["datasets"][dataset]["tokenizer"]
    revision = document["models"][tokenizer_id]
    snapshot = (
        root
        / "locked-tokenizers"
        / f"models--{tokenizer_id}"
        / "snapshots"
        / revision
    )
    snapshot.mkdir(parents=True)
    downloads = root / "data/manifests/downloads.json"
    downloads.parent.mkdir(parents=True)
    downloads.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": {
                    tokenizer_id: {
                        "repo_id": tokenizer_id,
                        "revision": revision,
                        "snapshot_path": snapshot.relative_to(root).as_posix(),
                        "size_bytes": 1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return data_config, downloads, snapshot


@pytest.mark.parametrize(
    ("model", "dataset", "adapter_type"),
    [
        ("mdlm_sdtt", "lm1b", SDTTAdapter),
        ("mdlm_sdtt", "owt", SDTTAdapter),
        ("mdlm_di4c", "lm1b", Di4CAdapter),
        ("mdlm_di4c", "owt", Di4CAdapter),
        ("duo_di4c", "lm1b", Di4CAdapter),
        ("duo_di4c", "owt", Di4CAdapter),
    ],
)
def test_all_six_distilled_cells_render_real_project_wrappers(
    model: str, dataset: str, adapter_type: type
) -> None:
    item = request(model, dataset)
    adapter = adapter_type() if adapter_type is SDTTAdapter else adapter_type(model.split("_", 1)[0])

    command = adapter.render_command(item, run_dir(item), dry_run=True)

    wrapper = ROOT / "adapters" / ("sample_sdtt.py" if model == "mdlm_sdtt" else "sample_di4c.py")
    assert command[:4] == [command[0], "-B", "-u", str(wrapper)]
    assert Path(option(command, "--upstream-root")).is_dir()
    assert option(command, "--sample-count") == "17"
    assert option(command, "--num-steps") == "8"
    assert option(command, "--seed") == "42"
    assert option(command, "--sampler") == "ancestral"
    assert option(command, "--offline") == "true"
    assert Path(option(command, "--data-config")) == ROOT / "artifacts/data.yaml"
    assert Path(option(command, "--downloads-manifest")) == (
        ROOT / "data/manifests/downloads.json"
    )
    assert option(command, "--dataset") == dataset
    assert option(command, "--checkpoint-sha256") == "dry-run-unverified"
    assert option(command, "--config-sha256") == "dry-run-unverified"
    assert Path(option(command, "--output")) == run_dir(item) / "upstream_token_ids.json"


def test_sdtt_uses_kld_round_seven_and_the_official_local_snapshot() -> None:
    item = request("mdlm_sdtt", "owt", steps=4)
    command = SDTTAdapter().render_command(item, run_dir(item), dry_run=True)

    assert option(command, "--loss") == "kld"
    assert option(command, "--round") == "7"
    assert Path(option(command, "--checkpoint")) == (
        ROOT / "checkpoints/official/mdlm_sdtt/owt/model.safetensors"
    )
    assert Path(option(command, "--config")) == (
        ROOT / "checkpoints/official/mdlm_sdtt/owt/config.json"
    )
    assert Path(option(command, "--tokenizer-snapshot")) == (
        ROOT
        / "data/raw/huggingface/hub/models--gpt2/snapshots"
        / "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    )


@pytest.mark.parametrize(
    ("model", "dataset", "family", "suffix"),
    [
        ("mdlm_di4c", "lm1b", "masked_mdlm", "student_checkpoints/20000.ckpt"),
        ("mdlm_di4c", "owt", "masked_mdlm", "sdtt7-di4c2.ckpt"),
        ("duo_di4c", "lm1b", "uniform_duo", "student_checkpoints/20000.ckpt"),
        ("duo_di4c", "owt", "uniform_duo", "student_checkpoints/50000.ckpt"),
    ],
)
def test_di4c_binds_teacher_family_and_dataset_intermediate_checkpoint(
    model: str, dataset: str, family: str, suffix: str
) -> None:
    item = request(model, dataset)
    command = Di4CAdapter(model.split("_", 1)[0]).render_command(
        item, run_dir(item), dry_run=True
    )

    assert option(command, "--teacher-family") == family
    assert Path(option(command, "--checkpoint")).as_posix().endswith(suffix)
    assert Path(option(command, "--config")).name in {
        "config.yaml",
        "di4c_mdlm_owt.yaml",
    }
    if family == "uniform_duo":
        assert "sdtt7-di4c2.ckpt" not in " ".join(command)
        assert "official/mdlm_di4c" not in " ".join(command)


def test_di4c_rejects_cross_family_model_identity_before_checkpoint_resolution() -> None:
    item = request("duo_di4c", "owt")
    with pytest.raises(AdapterError, match="does not support"):
        Di4CAdapter("mdlm").render_command(item, run_dir(item), dry_run=True)


def test_distilled_dry_run_matrix_has_six_supported_cells(capsys) -> None:
    exit_code = command_main(
        [
            "--root",
            str(ROOT),
            "--models",
            "mdlm_sdtt,mdlm_di4c,duo_di4c",
            "--datasets",
            "lm1b,owt",
            "--steps",
            "8",
            "--num-samples",
            "17",
            "--seed",
            "42",
            "--dry-run",
        ]
    )
    lines = capsys.readouterr().out.splitlines()

    assert exit_code == 0
    assert len(lines) == 6
    assert all('"status": "supported"' in line for line in lines)


def test_wrapper_files_are_source_auditable_and_upstreams_stay_unmodified() -> None:
    assert (ROOT / "adapters/sample_sdtt.py").is_file()
    assert (ROOT / "adapters/sample_di4c.py").is_file()
    assert (ROOT / "patches/sdtt/README.md").is_file()
    assert (ROOT / "patches/di4c/README.md").is_file()


class FakeTensor:
    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.rows


class FakeModel:
    def __init__(self, length: int, fail_on_call: int | None = None) -> None:
        self.length = length
        self.fail_on_call = fail_on_call
        self.calls: list[dict[str, object]] = []

    def sample(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("fake sampler failure")
        start = sum(int(call["n_samples"]) for call in self.calls[:-1])
        return FakeTensor(
            [[start + index + 1] * self.length for index in range(kwargs["n_samples"])]
        )


class FakeTokenizer:
    def batch_decode(self, rows):
        return [f"decoded {row[0]}" for row in rows]


def test_sampling_runtime_streams_exact_batches_and_atomically_publishes(tmp_path: Path) -> None:
    runtime = load_runtime_module()
    model = FakeModel(length=8)
    output = tmp_path / "capture.json"

    runtime.write_capture_atomic(
        output,
        model=model,
        tokenizer=FakeTokenizer(),
        sample_count=17,
        batch_size=16,
        num_steps=4,
        seq_len=8,
        sampler="ancestral",
    )

    document = json.loads(output.read_text(encoding="utf-8"))
    assert [call["n_samples"] for call in model.calls] == [16, 1]
    assert all(call["num_steps"] == 4 for call in model.calls)
    assert [item["sample_id"] for item in document["samples"]] == list(range(17))
    assert all(len(item["token_ids"]) == 8 for item in document["samples"])
    assert not any(path.name.endswith(".partial") for path in tmp_path.iterdir())


def test_sampling_runtime_failure_preserves_existing_output(tmp_path: Path) -> None:
    runtime = load_runtime_module()
    output = tmp_path / "capture.json"
    output.write_text("old-complete-output", encoding="utf-8")

    with pytest.raises(RuntimeError, match="fake sampler failure"):
        runtime.write_capture_atomic(
            output,
            model=FakeModel(length=8, fail_on_call=2),
            tokenizer=FakeTokenizer(),
            sample_count=17,
            batch_size=16,
            num_steps=4,
            seq_len=8,
            sampler="ancestral",
        )

    assert output.read_text(encoding="utf-8") == "old-complete-output"
    assert not any(path.name.endswith(".partial") for path in tmp_path.iterdir())


def test_sampling_runtime_rejects_a_symlink_output(tmp_path: Path) -> None:
    runtime = load_runtime_module()
    target = tmp_path / "target.json"
    target.write_text("do not overwrite", encoding="utf-8")
    output = tmp_path / "capture.json"
    output.symlink_to(target)

    with pytest.raises(ValueError, match="unsafe"):
        runtime.write_capture_atomic(
            output,
            model=FakeModel(length=8),
            tokenizer=FakeTokenizer(),
            sample_count=1,
            batch_size=1,
            num_steps=1,
            seq_len=8,
            sampler="ancestral",
        )

    assert target.read_text(encoding="utf-8") == "do not overwrite"


def test_lightning_dictconfig_checkpoint_uses_verified_full_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch PyTorch 2.5 weights-only rejecting Lightning's OmegaConf DictConfig."""

    runtime = load_runtime_module()
    checkpoint = tmp_path / "student.ckpt"
    checkpoint.write_bytes(b"manifest-verified-lightning-checkpoint")
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    embedded = {"model": {"type": "ddit-orig"}}
    calls: list[dict[str, object]] = []

    def load(path, **kwargs):
        calls.append({"path": path, **kwargs})
        if kwargs.get("weights_only") is not False:
            raise RuntimeError("Weights only load failed for omegaconf.dictconfig.DictConfig")
        return {
            "state_dict": {"backbone.weight": object()},
            "hyper_parameters": {"config": embedded},
        }

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=load))

    state, observed = runtime.checkpoint_state(checkpoint, expected)

    assert calls == [
        {"path": str(checkpoint), "map_location": "cpu", "weights_only": False}
    ]
    assert state == {"backbone.weight": state["backbone.weight"]}
    assert observed is embedded


def test_checkpoint_loader_rejects_unverified_pickle_before_torch_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = load_runtime_module()
    checkpoint = tmp_path / "student.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    called = False

    def load(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("must not deserialize unverified bytes")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=load))

    with pytest.raises(ValueError, match="SHA-256"):
        runtime.checkpoint_state(checkpoint, "0" * 64)
    assert not called


def test_real_lightning_omegaconf_fixture_when_runtime_is_available(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    runtime = load_runtime_module()
    checkpoint = tmp_path / "student.ckpt"
    config = OmegaConf.create({"model": {"type": "ddit-orig"}})
    torch.save(
        {
            "state_dict": {"backbone.weight": torch.tensor([1.0])},
            "hyper_parameters": {"config": config},
        },
        checkpoint,
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    state, embedded = runtime.checkpoint_state(checkpoint, digest)

    assert state["backbone.weight"].tolist() == [1.0]
    assert OmegaConf.to_container(embedded) == {"model": {"type": "ddit-orig"}}


def test_embedded_config_must_match_manifest_selected_architecture() -> None:
    runtime = load_runtime_module()
    authoritative = {
        "model": {"type": "ddit-orig", "hidden_size": 768, "n_blocks": 12},
        "parameterization": {"name": "multi-round-sdtt", "sampling_mode": "ancestral"},
        "training": {"sampling_eps": 1e-5},
        "T": 1024,
        "time_conditioning": False,
    }
    mismatched = json.loads(json.dumps(authoritative))
    mismatched["model"]["n_blocks"] = 24

    with pytest.raises(ValueError, match="model.n_blocks"):
        runtime.validate_embedded_config(authoritative, mismatched)


def test_tokenizer_binding_cross_checks_data_and_download_manifests(tmp_path: Path) -> None:
    runtime = load_runtime_module()
    data_config, downloads, snapshot = prepare_tokenizer_binding(tmp_path, "owt")

    binding = runtime.load_tokenizer_binding(
        data_config, downloads, "owt", snapshot
    )
    assert binding.tokenizer_id == "gpt2"
    assert binding.snapshot == snapshot

    document = json.loads(downloads.read_text(encoding="utf-8"))
    document["models"]["gpt2"]["revision"] = "0" * 40
    downloads.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        runtime.load_tokenizer_binding(data_config, downloads, "owt", snapshot)


def test_project_sampling_config_bytes_are_bound_to_checkpoint_provenance(
    tmp_path: Path,
) -> None:
    for relative in (
        Path("artifacts/checkpoints.yaml"),
        Path("configs/experiments.yaml"),
        Path("configs/sampling/di4c_mdlm_owt.yaml"),
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    manifest = tmp_path / "artifacts/checkpoints.yaml"
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    checkpoint_path = "checkpoints/official/mdlm_di4c/owt/sdtt7-di4c2.ckpt"
    (tmp_path / "artifacts/checkpoint_lock.json").write_text(
        json.dumps(
            {
                "manifest_sha256": manifest_sha256,
                "resources": {
                    "di4c_mdlm_owt_zenodo": {
                        "status": "downloaded",
                        "files": [
                            {
                                "path": checkpoint_path,
                                "size_bytes": 3,
                                "sha256": "a" * 64,
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    item = request("mdlm_di4c", "owt", samples=1)

    provenance = _resolve_checkpoint_provenance(tmp_path, item, None)
    assert provenance.selection["sampling_config_sha256"] == (
        "cf627f89b8dede9b25ac8e10407c3b7de203a76c4ba84100723287e571dcec34"
    )
    assert provenance.selection["sampling_config_source_commit"] == (
        "ac61ff9fe8e85120f9e1d2a8c5a332f8b8353dd3"
    )

    config = tmp_path / "configs/sampling/di4c_mdlm_owt.yaml"
    config.write_text(config.read_text(encoding="utf-8") + "\n# mutation\n", encoding="utf-8")
    with pytest.raises(ValueError, match="project sampling config differs"):
        _resolve_checkpoint_provenance(tmp_path, item, None)


@pytest.mark.parametrize(
    ("adapter", "model", "dataset"),
    [
        (SDTTAdapter(), "mdlm_sdtt", "lm1b"),
        (SDTTAdapter(), "mdlm_sdtt", "owt"),
        (Di4CAdapter("mdlm"), "mdlm_di4c", "lm1b"),
        (Di4CAdapter("mdlm"), "mdlm_di4c", "owt"),
        (Di4CAdapter("duo"), "duo_di4c", "lm1b"),
        (Di4CAdapter("duo"), "duo_di4c", "owt"),
    ],
)
def test_real_commands_reject_missing_or_unverified_checkpoint_bytes(
    adapter, model: str, dataset: str
) -> None:
    item = request(model, dataset, samples=1)
    with pytest.raises(AdapterError, match="checkpoint"):
        adapter.build_command(item, run_dir(item))


def prepare_official_sdtt_conversion(tmp_path: Path, item: RunRequest) -> RunRequest:
    for relative in (
        Path("artifacts/data.yaml"),
        Path("artifacts/checkpoints.yaml"),
        Path("configs/experiments.yaml"),
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((ROOT / relative).read_text(encoding="utf-8"), encoding="utf-8")
    manifest = tmp_path / "artifacts/checkpoints.yaml"
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    files = [
        {
            "path": f"checkpoints/official/mdlm_sdtt/owt/{name}",
            "size_bytes": 3,
            "sha256": character * 64,
        }
        for name, character in (("config.json", "a"), ("model.safetensors", "b"))
    ]
    (tmp_path / "artifacts/checkpoint_lock.json").write_text(
        json.dumps(
            {
                "manifest_sha256": manifest_sha256,
                "resources": {"sdtt_owt_hf": {"status": "downloaded", "files": files}},
            }
        ),
        encoding="utf-8",
    )
    selector = {
        "resource": "sdtt_owt_hf",
        "path": None,
        "teacher_family": None,
        "sampling_config": "config.json",
        "sampling_config_source": "resource",
        "sampling_config_sha256": None,
        "sampling_config_source_commit": None,
    }
    digest = hashlib.sha256(
        json.dumps(
            {
                "manifest_sha256": manifest_sha256,
                "selector": selector,
                "files": sorted(files, key=lambda value: value["path"]),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return replace(
        item,
        checkpoint_sha256=digest,
        checkpoint_lock_id=f"sdtt_owt_hf:{manifest_sha256}:all",
        checkpoint_selection=selector,
        checkpoint_teacher_family="masked_mdlm",
    )


def test_conversion_validates_family_before_reading_capture(tmp_path: Path) -> None:
    item = request("mdlm_sdtt", "owt", samples=1)
    item = prepare_official_sdtt_conversion(tmp_path, item)
    item = replace(item, checkpoint_teacher_family="uniform_duo")
    output = (
        tmp_path / "results/samples/owt/mdlm_sdtt" / f"steps_{item.step_count}"
    )
    output.mkdir(parents=True)

    with pytest.raises(AdapterError, match="teacher family"):
        list(SDTTAdapter().convert_outputs(item, output))

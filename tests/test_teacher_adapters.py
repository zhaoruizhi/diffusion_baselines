from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

import dlb.adapters.capture as capture_module
from dlb.adapters.base import AdapterError
from dlb.adapters.candi import CANDIAdapter
from dlb.adapters.duo import DuoAdapter
from dlb.adapters.flm import FLMAdapter
from dlb.adapters.mdlm import MDLMAdapter
from dlb.command import main as command_main
from dlb.checkpoints import load_checkpoint_manifest
from dlb.registry import load_registry
from dlb.runner import RunRequest, _resolve_checkpoint_provenance


ROOT = Path(__file__).parents[1]


def request(model: str, dataset: str, steps: int = 32, samples: int = 17) -> RunRequest:
    return RunRequest(
        run_id=f"{model}-{dataset}-steps-{steps}",
        model_id=model,
        dataset_id=dataset,
        step_count=steps,
        seed=7,
        sample_count=samples,
    )


def run_dir(root: Path, item: RunRequest) -> Path:
    return (
        root
        / "results"
        / "samples"
        / item.dataset_id
        / item.model_id
        / f"steps_{item.step_count}"
    )


def override(command: list[str], key: str) -> str:
    prefix = key + "="
    values = [argument[len(prefix) :] for argument in command if argument.startswith(prefix)]
    assert len(values) == 1, (key, command)
    return values[0]


def test_hf_masked_lm_adapter_only_truncates_duo_extra_vocab_class() -> None:
    """Catch Duo's HF checkpoint leaking its non-tokenizer runtime class into sampling."""

    class FakeLogits:
        def __init__(self, last_dim: int):
            self.shape = (2, 3, last_dim)

        def __getitem__(self, key):
            assert key == (Ellipsis, slice(None, 50_257, None))
            return FakeLogits(50_257)

    class FakeMaskedLM:
        def __init__(self, *, model_type: str, bare_logits: bool = False):
            self.config = SimpleNamespace(model_type=model_type, vocab_size=50_258)
            self.bare_logits = bare_logits

        def forward(self, input_ids, timesteps):
            del input_ids, timesteps
            logits = FakeLogits(50_258)
            if self.bare_logits:
                return logits
            return SimpleNamespace(logits=logits)

    duo = capture_module._adapt_hf_masked_lm_backbone(
        FakeMaskedLM(model_type="DUO")
    )
    duo_bare = capture_module._adapt_hf_masked_lm_backbone(
        FakeMaskedLM(model_type="DUO", bare_logits=True)
    )
    other = capture_module._adapt_hf_masked_lm_backbone(
        FakeMaskedLM(model_type="MDLM")
    )

    duo_logits = duo(
        x=[[0, 0, 0], [0, 0, 0]],
        sigma=[1, 1],
    )
    duo_bare_logits = duo_bare(
        x=[[0, 0, 0], [0, 0, 0]],
        sigma=[1, 1],
    )
    other_logits = other(
        x=[[0, 0, 0], [0, 0, 0]],
        sigma=[1, 1],
    )

    assert duo_logits.shape == (2, 3, 50_257)
    assert duo_bare_logits.shape == (2, 3, 50_257)
    assert other_logits.shape == (2, 3, 50_258)


def prepare_conversion_root(tmp_path: Path) -> Path:
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "experiments.yaml").write_text(
        (ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "artifacts" / "data.yaml").write_text(
        (ROOT / "artifacts" / "data.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "artifacts" / "checkpoints.yaml").write_text(
        (ROOT / "artifacts" / "checkpoints.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return tmp_path


def canonical_conversion_request(root: Path, item: RunRequest) -> RunRequest:
    registry = load_registry(root / "configs" / "experiments.yaml")
    manifest_model = load_checkpoint_manifest(root / "artifacts" / "checkpoints.yaml")
    support = registry.models[item.model_id].datasets[item.dataset_id]
    manifest = root / "artifacts" / "checkpoints.yaml"
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if support.train_recipe:
        recipe = manifest_model.recipes[support.train_recipe]
        checkpoint = root / recipe.output / str(recipe.sampling_checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"ckp")
        provenance = _resolve_checkpoint_provenance(root, item, support.train_recipe)
        return replace(
            item,
            checkpoint_sha256=provenance.sha256,
            checkpoint_lock_id=provenance.lock_id,
            checkpoint_selection=provenance.selection,
            checkpoint_teacher_family=provenance.teacher_family,
        )

    coverage = manifest_model.coverage[(item.model_id, item.dataset_id)]
    resource = coverage.resource
    selected_path = coverage.path or "model.safetensors"
    files = [
        {
            "path": f"checkpoints/fixture/{resource}/{selected_path}",
            "size_bytes": 3,
            "sha256": "b" * 64,
        }
    ]
    (root / "artifacts" / "checkpoint_lock.json").write_text(
        json.dumps(
            {
                "manifest_sha256": manifest_sha256,
                "resources": {resource: {"status": "downloaded", "files": files}},
            }
        ),
        encoding="utf-8",
    )
    provenance = _resolve_checkpoint_provenance(root, item, support.train_recipe)
    return replace(
        item,
        checkpoint_sha256=provenance.sha256,
        checkpoint_lock_id=provenance.lock_id,
        checkpoint_selection=provenance.selection,
        checkpoint_teacher_family=provenance.teacher_family,
    )


def write_standard_output(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"generative_ppl": 1.0, "entropy": 2.0, "generated_seqs": texts}),
        encoding="utf-8",
    )


def write_capture(
    path: Path, count: int, token_length: int, *, duplicate_id: bool = False
) -> None:
    samples = [
        {
            "sample_id": 0 if duplicate_id and index == 1 else index,
            "text": f"sample {index}",
            "token_ids": [index % 100 + 1] * token_length,
        }
        for index in range(count)
    ]
    path.write_text(
        json.dumps({"schema": "dlb-upstream-token-capture-v1", "samples": samples}),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("adapter", "model", "upstream"),
    [
        (FLMAdapter(), "flm", "flm"),
        (DuoAdapter(), "duo", "duo"),
        (MDLMAdapter(), "mdlm", "mdlm"),
        (CANDIAdapter(), "candi", "candi"),
    ],
)
def test_commands_reference_real_pinned_entrypoints_and_actual_hydra_keys(
    adapter, model: str, upstream: str
) -> None:
    """Catch adapters drifting to a nonexistent entrypoint or invented override."""

    item = request(model, "owt")
    command = adapter.render_command(item, run_dir(ROOT, item), dry_run=True)
    entrypoint = Path(override(command, "--upstream-entrypoint"))

    assert entrypoint == ROOT / "upstreams" / upstream / "main.py"
    assert entrypoint.is_file()
    config = yaml.safe_load((entrypoint.parent / "configs" / "config.yaml").read_text())
    data_config = override(command, "data")
    assert (entrypoint.parent / "configs" / "data" / f"{data_config}.yaml").is_file()
    for key in ("sampling.steps", "sampling.num_sample_batches", "loader.eval_batch_size"):
        section, child = key.split(".")
        assert child in config[section]
        assert override(command, key)


def test_flm_and_fmlm_render_exact_paper_sampling_semantics() -> None:
    """Catch FLM losing Euler or FMLM losing its dataset-specific paper gamma."""

    flm = request("flm", "owt")
    fmlm_lm1b = request("fmlm", "lm1b", steps=8)
    fmlm_owt = request("fmlm", "owt", steps=8)

    flm_command = FLMAdapter().render_command(flm, run_dir(ROOT, flm), dry_run=True)
    lm1b_command = FLMAdapter().render_command(
        fmlm_lm1b, run_dir(ROOT, fmlm_lm1b), dry_run=True
    )
    owt_command = FLMAdapter().render_command(
        fmlm_owt, run_dir(ROOT, fmlm_owt), dry_run=True
    )

    assert override(flm_command, "sampling.solver") == "euler"
    assert override(flm_command, "sampling.steps") == "32"
    assert override(lm1b_command, "algo") == "fmlm"
    assert override(lm1b_command, "sampling.gamma") == "0.8"
    assert override(owt_command, "sampling.gamma") == "1.0"


def test_flm_family_commands_use_official_lightning_checkpoints(tmp_path: Path) -> None:
    """Catch FLM/FMLM official ckpts being routed through the HF AutoModel branch."""

    cells = (
        ("flm", "lm1b", "checkpoints/official/flm_ckpt/lm1b/lm1b_flm.ckpt"),
        ("flm", "owt", "checkpoints/official/flm_ckpt/owt/owt_flm.ckpt"),
        ("fmlm", "lm1b", "checkpoints/official/fmlm_ckpt/lm1b/lm1b_fmlm.ckpt"),
        ("fmlm", "owt", "checkpoints/official/fmlm_ckpt/owt/owt_fmlm.ckpt"),
    )
    adapter = FLMAdapter()
    root = prepare_conversion_root(tmp_path)
    (root / "upstreams" / "flm").mkdir(parents=True)
    (root / "upstreams" / "flm" / "main.py").write_text("# fixture\n", encoding="utf-8")

    for model, dataset, checkpoint in cells:
        item = request(model, dataset)
        command = adapter.render_command(item, run_dir(root, item), dry_run=True)

        assert override(command, "algo.backbone") == "dit"
        assert Path(override(command, "eval.checkpoint_path")) == root / checkpoint


@pytest.mark.parametrize(
    ("adapter", "model", "sampler_key", "sampler_value", "noise_value"),
    [
        (DuoAdapter(), "duo", "sampling.predictor", "ancestral", "ancestral"),
        (DuoAdapter(), "duo_dcd", "sampling.predictor", "ancestral", "ancestral"),
        (MDLMAdapter(), "mdlm", "sampling.predictor", "ddpm", "True"),
    ],
)
def test_discrete_commands_use_actual_ancestral_keys_without_invented_temperature(
    adapter, model: str, sampler_key: str, sampler_value: str, noise_value: str
) -> None:
    """Catch a non-ancestral sampler or a Hydra override absent from pinned configs."""

    item = request(model, "owt", steps=8)
    command = adapter.render_command(item, run_dir(ROOT, item), dry_run=True)

    assert override(command, sampler_key) == sampler_value
    assert override(command, "sampling.noise_removal") == noise_value
    assert not any(argument.startswith("sampling.temperature=") for argument in command)


def test_candi_command_uses_the_pinned_hybrid_sampler() -> None:
    """Catch CANDI being reduced to a discrete-only or continuous-only sampler."""

    item = request("candi", "owt", steps=16)
    command = CANDIAdapter().render_command(item, run_dir(ROOT, item), dry_run=True)

    assert override(command, "algo") == "candi"
    assert override(command, "algo.sampler") == "cached"
    assert override(command, "algo.mixed_coeff") == "0.5"
    assert override(command, "algo.step_size") == "1.0"
    assert override(command, "algo.temp") == "1.0"
    assert override(command, "sampling.steps") == "16"


def test_commands_accept_smoke_results_root_under_project_results(tmp_path: Path) -> None:
    """Catch adapters rejecting runner requests with --results-root results/smoke."""

    root = prepare_conversion_root(tmp_path)
    (root / "upstreams" / "candi").mkdir(parents=True)
    (root / "upstreams" / "candi" / "main.py").write_text("# fixture\n", encoding="utf-8")
    results_root = root / "results" / "smoke"
    item = replace(request("candi", "lm1b", steps=1, samples=1), results_root=str(results_root))
    output_dir = (
        results_root
        / "samples"
        / item.dataset_id
        / item.model_id
        / f"steps_{item.step_count}"
    )

    command = CANDIAdapter().render_command(item, output_dir, dry_run=True)

    assert Path(override(command, "--capture-path")).parent == output_dir


@pytest.mark.parametrize(
    ("adapter", "model", "dataset", "batch_size"),
    [
        (FLMAdapter(), "flm", "lm1b", 32),
        (FLMAdapter(), "fmlm", "owt", 16),
        (DuoAdapter(), "duo", "owt", 8),
        (MDLMAdapter(), "mdlm", "owt", 16),
        (CANDIAdapter(), "candi", "owt", 2),
    ],
)
def test_sample_batches_are_ceiled_from_the_configured_eval_batch(
    adapter, model: str, dataset: str, batch_size: int
) -> None:
    """Catch floor division silently generating fewer samples than requested."""

    item = request(model, dataset, samples=batch_size + 1)
    command = adapter.render_command(item, run_dir(ROOT, item), dry_run=True)

    assert override(command, "loader.eval_batch_size") == str(batch_size)
    assert override(command, "sampling.num_sample_batches") == "2"
    assert override(command, "trainer.devices") == "1"


def test_rendered_commands_have_only_concrete_nonempty_arguments() -> None:
    """Catch unresolved templates or sentinel paths reaching subprocess argv."""

    adapters = {
        "flm": FLMAdapter(),
        "fmlm": FLMAdapter(),
        "duo": DuoAdapter(),
        "duo_dcd": DuoAdapter(),
        "mdlm": MDLMAdapter(),
        "candi": CANDIAdapter(),
    }
    for model, adapter in adapters.items():
        for dataset in ("lm1b", "owt"):
            item = request(model, dataset)
            command = adapter.render_command(item, run_dir(ROOT, item), dry_run=True)
            entrypoint = Path(override(command, "--upstream-entrypoint"))
            data_config = override(command, "data")
            assert (entrypoint.parent / "configs" / "data" / f"{data_config}.yaml").is_file()
            assert command
            assert all(argument and "${" not in argument for argument in command)
            assert all("{" not in argument and "}" not in argument for argument in command)
            assert all("None" not in argument for argument in command)


def test_real_build_rejects_a_missing_checkpoint() -> None:
    """Catch non-dry execution silently relying on a future or wrong checkpoint."""

    item = request("flm", "owt")
    with pytest.raises(AdapterError, match="checkpoint"):
        FLMAdapter().build_command(item, run_dir(ROOT, item))


def test_conversion_prefers_captured_token_ids_and_trims_only_ceiling_excess(
    tmp_path: Path,
) -> None:
    """Catch re-tokenization despite IDs being available or nondeterministic trimming."""

    root = prepare_conversion_root(tmp_path)
    item = canonical_conversion_request(root, request("flm", "owt", samples=17))
    output_dir = run_dir(root, item)
    output_dir.mkdir(parents=True)
    texts = [f"sample {index}" for index in range(32)]
    write_standard_output(output_dir / "upstream_samples.json", texts)
    write_capture(output_dir / "upstream_token_ids.json", 32, 1024)

    records = list(FLMAdapter().convert_outputs(item, output_dir))

    assert len(records) == 17
    assert [record.sample_id for record in records] == list(range(17))
    assert records[1].token_ids == [2] * 1024
    metadata = json.loads((output_dir / "conversion_metadata.json").read_text())
    assert metadata["token_ids_source"] == "upstream"
    assert metadata["trimmed_samples"] == 15
    assert metadata["generation_seconds_source"] == "unavailable_excluded_sentinel"
    assert metadata["checkpoint_sha256"] == item.checkpoint_sha256
    assert metadata["checkpoint_lock_id"] == item.checkpoint_lock_id
    assert metadata["checkpoint_selection"] == item.checkpoint_selection
    assert metadata["teacher_family"] == "continuous_flm"
    assert all(record.generation_seconds == 0.0 for record in records)


def test_conversion_rejects_missing_runner_checkpoint_provenance_before_reading_outputs(
    tmp_path: Path,
) -> None:
    """Catch direct conversion bypassing the runner's canonical checkpoint resolution."""

    root = prepare_conversion_root(tmp_path)
    item = request("flm", "owt", samples=1)
    output_dir = run_dir(root, item)
    output_dir.mkdir(parents=True)

    with pytest.raises(AdapterError, match="runner-resolved checkpoint provenance is required"):
        list(FLMAdapter().convert_outputs(item, output_dir))


def test_conversion_rejects_wrong_teacher_family_before_reading_outputs(
    tmp_path: Path,
) -> None:
    """Catch a valid checkpoint identity being relabeled as another teacher family."""

    root = prepare_conversion_root(tmp_path)
    item = canonical_conversion_request(root, request("flm", "owt", samples=1))
    item = replace(item, checkpoint_teacher_family="continuous_rdlm")
    output_dir = run_dir(root, item)
    output_dir.mkdir(parents=True)

    with pytest.raises(AdapterError, match="teacher family"):
        list(FLMAdapter().convert_outputs(item, output_dir))


def test_conversion_rejects_extra_missing_empty_duplicate_and_invalid_samples(
    tmp_path: Path,
) -> None:
    """Catch malformed upstream artifacts being normalized into plausible samples."""

    root = prepare_conversion_root(tmp_path)
    item = canonical_conversion_request(root, request("flm", "owt", samples=17))
    output_dir = run_dir(root, item)
    output_dir.mkdir(parents=True)
    write_standard_output(output_dir / "upstream_samples.json", [f"sample {i}" for i in range(33)])
    write_capture(output_dir / "upstream_token_ids.json", 33, 1024)
    with pytest.raises(AdapterError, match="expected 32 generated samples, found 33"):
        list(FLMAdapter().convert_outputs(item, output_dir))

    write_standard_output(output_dir / "upstream_samples.json", [f"sample {i}" for i in range(32)])
    write_capture(output_dir / "upstream_token_ids.json", 32, 1024, duplicate_id=True)
    with pytest.raises(AdapterError, match="duplicate sample_id"):
        list(FLMAdapter().convert_outputs(item, output_dir))

    write_capture(output_dir / "upstream_token_ids.json", 32, 1024)
    capture = json.loads((output_dir / "upstream_token_ids.json").read_text())
    capture["samples"][0]["token_ids"] = [50257] + [1] * 1023
    (output_dir / "upstream_token_ids.json").write_text(json.dumps(capture))
    with pytest.raises(AdapterError, match="invalid token"):
        list(FLMAdapter().convert_outputs(item, output_dir))

    write_capture(output_dir / "upstream_token_ids.json", 32, 1024)
    actual = json.loads((output_dir / "upstream_samples.json").read_text())
    actual["generated_seqs"][0] = "   "
    (output_dir / "upstream_samples.json").write_text(json.dumps(actual))
    with pytest.raises(AdapterError, match="empty text"):
        list(FLMAdapter().convert_outputs(item, output_dir))


def test_mdlm_conversion_accepts_only_the_documented_capture_format(tmp_path: Path) -> None:
    """Catch parsing MDLM's lossy final-batch stdout representation."""

    root = prepare_conversion_root(tmp_path)
    item = canonical_conversion_request(root, request("mdlm", "lm1b", samples=17))
    output_dir = run_dir(root, item)
    output_dir.mkdir(parents=True)
    write_capture(output_dir / "upstream_token_ids.json", 32, 128)

    records = list(MDLMAdapter().convert_outputs(item, output_dir))
    assert len(records) == 17
    assert records[0].token_ids == [1] * 128

    (output_dir / "upstream_token_ids.json").write_text(json.dumps(["not", "supported"]))
    with pytest.raises(AdapterError, match="unexpected capture format"):
        list(MDLMAdapter().convert_outputs(item, output_dir))


@pytest.mark.parametrize(
    ("dataset", "length", "batch_size"), [("lm1b", 128, 32), ("owt", 1024, 16)]
)
def test_conversion_requires_exact_canonical_token_length_for_each_dataset(
    tmp_path: Path, dataset: str, length: int, batch_size: int
) -> None:
    """Catch short or long token sequences passing canonical conversion validation."""

    root = prepare_conversion_root(tmp_path)
    item = canonical_conversion_request(root, request("flm", dataset, samples=1))
    output_dir = run_dir(root, item)
    output_dir.mkdir(parents=True)
    texts = [f"sample {index}" for index in range(batch_size)]
    write_standard_output(output_dir / "upstream_samples.json", texts)

    write_capture(output_dir / "upstream_token_ids.json", batch_size, length)
    records = list(FLMAdapter().convert_outputs(item, output_dir))
    assert len(records[0].token_ids) == length

    write_capture(output_dir / "upstream_token_ids.json", batch_size, length - 1)
    with pytest.raises(AdapterError, match=f"expected {length} tokens"):
        list(FLMAdapter().convert_outputs(item, output_dir))

    write_capture(output_dir / "upstream_token_ids.json", batch_size, length + 1)
    with pytest.raises(AdapterError, match=f"expected {length} tokens"):
        list(FLMAdapter().convert_outputs(item, output_dir))


def test_text_only_conversion_pads_and_truncates_with_the_pinned_offline_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch text fallback emitting variable-length IDs or hiding its transformation."""

    class FakeTokenizer:
        eos_token = "<eos>"
        pad_token = None
        pad_token_id = 50256

        def __call__(self, texts, **settings):
            assert settings == {
                "add_special_tokens": False,
                "padding": "max_length",
                "truncation": True,
                "max_length": 1024,
                "return_attention_mask": False,
            }
            return {"input_ids": [[11] + [50256] * 1023 for _ in texts]}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name, **settings):
            assert name == "gpt2"
            assert settings == {
                "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
                "local_files_only": True,
            }
            return FakeTokenizer()

    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoTokenizer=FakeAutoTokenizer)
    )
    root = prepare_conversion_root(tmp_path)
    item = canonical_conversion_request(root, request("flm", "owt", samples=1))
    output_dir = run_dir(root, item)
    output_dir.mkdir(parents=True)
    write_standard_output(
        output_dir / "upstream_samples.json", [f"sample {index}" for index in range(16)]
    )

    records = list(FLMAdapter().convert_outputs(item, output_dir))
    metadata = json.loads((output_dir / "conversion_metadata.json").read_text())

    assert records[0].token_ids == [11] + [50256] * 1023
    assert metadata["token_ids_transformation"] == {
        "add_special_tokens": False,
        "max_length": 1024,
        "padding": "max_length",
        "pad_token_id": 50256,
        "truncation": True,
    }


def test_command_cli_dry_renders_six_models_on_both_datasets_without_writes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Catch dry-run omitting a cell or accidentally trying to acquire artifacts."""

    result = command_main(
        [
            "--root",
            str(ROOT),
            "--models",
            "flm,fmlm,duo,duo_dcd,mdlm,candi",
            "--datasets",
            "lm1b,owt",
            "--dry-run",
        ]
    )

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert result == 0
    assert len(records) == 12
    assert {(record["model"], record["dataset"]) for record in records} == {
        (model, dataset)
        for model in ("flm", "fmlm", "duo", "duo_dcd", "mdlm", "candi")
        for dataset in ("lm1b", "owt")
    }
    assert all(record["status"] == "supported" and record["command"] for record in records)


def test_command_cli_emits_a_structured_unsupported_record(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Catch unsupported combinations terminating the whole render matrix."""

    assert command_main(["--root", str(ROOT), "--models", "rdlm", "--datasets", "owt", "--dry-run"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "unsupported"
    assert record["model"] == "rdlm"
    assert record["dataset"] == "owt"
    assert record["reason"]

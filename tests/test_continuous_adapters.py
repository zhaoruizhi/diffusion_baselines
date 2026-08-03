"""Contracts for the pinned LangFlow and RDLM continuous teachers."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

import dlb.adapters.capture as capture_module
from dlb.adapters.base import AdapterError
from dlb.adapters.langflow import LangFlowAdapter
from dlb.adapters.rdlm import RDLMAdapter
from dlb.command import main as command_main
from dlb.registry import load_registry
from dlb.runner import RunRequest, _resolve_checkpoint_provenance


ROOT = Path(__file__).parents[1]


def request(model: str, dataset: str, steps: int = 32, samples: int = 17) -> RunRequest:
    return RunRequest(
        run_id=f"{model}-{dataset}-steps-{steps}",
        model_id=model,
        dataset_id=dataset,
        step_count=steps,
        seed=42,
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


def option(command: list[str], key: str) -> str:
    index = command.index(key)
    return command[index + 1]


def override(command: list[str], key: str) -> str:
    prefix = key + "="
    values = [argument[len(prefix) :] for argument in command if argument.startswith(prefix)]
    assert len(values) == 1, (key, command)
    return values[0]


def prepare_conversion_root(tmp_path: Path) -> Path:
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "configs").mkdir()
    for relative in (
        Path("artifacts/data.yaml"),
        Path("artifacts/checkpoints.yaml"),
        Path("configs/experiments.yaml"),
    ):
        target = tmp_path / relative
        target.write_text((ROOT / relative).read_text(encoding="utf-8"), encoding="utf-8")
    return tmp_path


def canonical_conversion_request(root: Path, item: RunRequest) -> RunRequest:
    resources = {
        "langflow": (
            "langflow_owt_hf",
            "continuous_langflow",
            [
                "config.json",
                "config.py",
                "model.py",
                "model.safetensors",
            ],
        ),
        "rdlm": (
            "rdlm_lm1b_drive",
            "continuous_rdlm",
            [
                "LM1B/checkpoint.pth",
                "LM1B/config.yaml",
                "LM1B/sde.pkl",
                "Text8/checkpoint.pth",
                "Text8/config.yaml",
                "Text8/sde.pkl",
            ],
        ),
    }
    resource, family, relative_files = resources[item.model_id]
    manifest = root / "artifacts" / "checkpoints.yaml"
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    files = [
        {
            "path": f"checkpoints/fixture/{resource}/{relative}",
            "size_bytes": index + 1,
            "sha256": f"{index + 1:064x}",
        }
        for index, relative in enumerate(relative_files)
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
    provenance = _resolve_checkpoint_provenance(root, item, None)
    return replace(
        item,
        checkpoint_sha256=provenance.sha256,
        checkpoint_lock_id=provenance.lock_id,
        checkpoint_selection=provenance.selection,
        checkpoint_teacher_family=provenance.teacher_family,
    )


def write_capture(path: Path, count: int, length: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "dlb-upstream-token-capture-v1",
                "samples": [
                    {
                        "sample_id": index,
                        "text": f"sample {index}",
                        "token_ids": [index + 1] * length,
                    }
                    for index in range(count)
                ],
            }
        ),
        encoding="utf-8",
    )


def prepare_tokenizer_binding(
    root: Path, dataset: str
) -> tuple[Path, Path, Path]:
    """Create the minimal immutable tokenizer lock consumed by capture tests."""

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


def test_registry_exposes_only_the_two_supported_continuous_cells() -> None:
    """Catch either unsupported cell being silently substituted by another model."""

    registry = load_registry(ROOT / "configs" / "experiments.yaml")

    assert registry.models["langflow"].datasets["owt"].status == "supported"
    assert registry.models["langflow"].datasets["lm1b"].status == "unsupported"
    assert registry.models["rdlm"].datasets["lm1b"].status == "supported"
    assert registry.models["rdlm"].datasets["owt"].status == "unsupported"


def test_langflow_renders_the_pinned_inference_argparse_contract(tmp_path: Path) -> None:
    """Catch flags that are not accepted by the pinned inference argparse parser."""

    root = prepare_conversion_root(tmp_path)
    entrypoint = root / "upstreams" / "langflow" / "inference.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# fixture\n", encoding="utf-8")
    item = request("langflow", "owt", steps=1024, samples=17)
    command = LangFlowAdapter().render_command(item, run_dir(root, item), dry_run=True)

    assert command[1:3] == ["-B", "-u"]
    assert Path(override(command, "--upstream-entrypoint")) == entrypoint
    assert override(command, "--capture-kind") == "langflow"
    assert Path(override(command, "--data-config-path")) == root / "artifacts/data.yaml"
    assert Path(override(command, "--downloads-manifest-path")) == (
        root / "data/manifests/downloads.json"
    )
    assert override(command, "--dataset-id") == "owt"
    assert Path(override(command, "--tokenizer-snapshot")) == (
        root
        / "data/raw/huggingface/hub/models--gpt2/snapshots"
        / "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    )
    checkpoint = Path(option(command, "--checkpoint"))
    assert checkpoint == root / "checkpoints/official/langflow/owt/model.safetensors"
    assert option(command, "--num_samples") == "17"
    assert not any(
        argument
        in {
            "--batch_size",
            "--num_steps",
            "--seq_length",
            "--seed",
            "--output",
            "--num-samples",
            "--num-steps",
            "--seq-length",
        }
        for argument in command
    )


def test_langflow_rejects_step_counts_not_exposed_by_pinned_inference(
    tmp_path: Path,
) -> None:
    """Catch labeling fixed-step LangFlow samples as a different step budget."""

    root = prepare_conversion_root(tmp_path)
    entrypoint = root / "upstreams" / "langflow" / "inference.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# fixture\n", encoding="utf-8")
    item = request("langflow", "owt", steps=32, samples=17)

    with pytest.raises(AdapterError, match="invalid step count 32 for fixed_1024"):
        LangFlowAdapter().render_command(item, run_dir(root, item), dry_run=True)


def test_rdlm_renders_the_official_sde_sampler_and_saved_asset_trio() -> None:
    """Catch RDLM bypassing its official Hydra sampler or omitting saved release assets."""

    item = request("rdlm", "lm1b", steps=32, samples=17)
    command = RDLMAdapter().render_command(item, run_dir(ROOT, item), dry_run=True)
    asset_root = ROOT / "checkpoints/official/rdlm/lm1b/LM1B"

    assert command[1:3] == ["-B", "-u"]
    assert Path(override(command, "--upstream-entrypoint")) == ROOT / "upstreams/rdlm/main.py"
    assert override(command, "--capture-kind") == "rdlm"
    assert Path(override(command, "--data-config-path")) == ROOT / "artifacts/data.yaml"
    assert Path(override(command, "--downloads-manifest-path")) == (
        ROOT / "data/manifests/downloads.json"
    )
    assert override(command, "--dataset-id") == "lm1b"
    assert Path(override(command, "--tokenizer-snapshot")) == (
        ROOT
        / "data/raw/huggingface/hub/models--bert-base-uncased/snapshots"
        / "86b5e0934494bd15c9632b12f734a8a67f723594"
    )
    assert Path(override(command, "--saved-config-path")) == asset_root / "config.yaml"
    assert Path(override(command, "--saved-sde-path")) == asset_root / "sde.pkl"
    assert override(command, "--expected-samples") == "17"
    assert override(command, "run_mode") == "sample"
    assert override(command, "server") == "sample"
    assert override(command, "exp") == "sample_lm1b"
    assert Path(override(command, "model_path")) == asset_root / "checkpoint.pth"
    assert override(command, "sampling.predictor") == "grw"
    assert override(command, "sampling.steps") == "32"
    assert override(command, "sampling.batch_per_gpu") == "8"
    assert override(command, "seed") == "42"
    assert override(command, "ngpus") == "1"
    assert override(command, "eval.entropy") == "False"
    assert override(command, "eval.nll") == "False"
    assert override(command, "eval.gen_ppl") == "False"


@pytest.mark.parametrize(
    ("adapter", "model", "dataset"),
    [
        (LangFlowAdapter(), "langflow", "owt"),
        (RDLMAdapter(), "rdlm", "lm1b"),
    ],
)
def test_real_render_requires_canonical_checkpoint_bytes(adapter, model: str, dataset: str) -> None:
    """Catch real mode accepting dry-run-only expected paths."""

    item = request(model, dataset)
    with pytest.raises(AdapterError, match="checkpoint"):
        adapter.build_command(item, run_dir(ROOT, item))


@pytest.mark.parametrize(
    ("adapter", "model", "dataset", "length", "family"),
    [
        (LangFlowAdapter(), "langflow", "owt", 1024, "continuous_langflow"),
        (RDLMAdapter(), "rdlm", "lm1b", 128, "continuous_rdlm"),
    ],
)
def test_capture_conversion_preserves_exact_upstream_ids_and_latency_sentinel(
    tmp_path: Path, adapter, model: str, dataset: str, length: int, family: str
) -> None:
    """Catch lossy parsing, wrong lengths/counts, or 0.0 being presented as measured latency."""

    root = prepare_conversion_root(tmp_path)
    item = canonical_conversion_request(root, request(model, dataset, samples=17))
    output_dir = run_dir(root, item)
    write_capture(output_dir / "upstream_token_ids.json", 17, length)

    records = list(adapter.convert_outputs(item, output_dir))
    metadata = json.loads((output_dir / "conversion_metadata.json").read_text())

    assert len(records) == 17
    assert records[0].token_ids == [1] * length
    assert metadata["format"] == "dlb-upstream-token-capture-v1"
    assert metadata["token_ids_source"] == "upstream"
    assert metadata["generated_samples"] == 17
    assert metadata["trimmed_samples"] == 0
    assert metadata["teacher_family"] == family
    assert metadata["generation_seconds_source"] == "unavailable_excluded_sentinel"
    assert metadata["exclude_from_latency"] is True
    assert all(record.generation_seconds == 0.0 for record in records)


@pytest.mark.parametrize("changed_name", ["LM1B/config.yaml", "LM1B/sde.pkl"])
def test_rdlm_conversion_binds_saved_config_and_sde_checksums_before_output_reads(
    tmp_path: Path, changed_name: str
) -> None:
    """Catch provenance that hashes only checkpoint.pth and ignores the saved trio."""

    root = prepare_conversion_root(tmp_path)
    item = canonical_conversion_request(root, request("rdlm", "lm1b", samples=1))
    lock_path = root / "artifacts/checkpoint_lock.json"
    lock = json.loads(lock_path.read_text())
    files = lock["resources"]["rdlm_lm1b_drive"]["files"]
    target = next(file for file in files if file["path"].endswith("/" + changed_name))
    target["sha256"] = "f" * 64
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    output_dir = run_dir(root, item)
    output_dir.mkdir(parents=True)

    with pytest.raises(AdapterError, match="checkpoint SHA"):
        list(RDLMAdapter().convert_outputs(item, output_dir))


def test_rdlm_retokenizes_variable_upstream_rows_offline_and_records_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch variable post-BOS RDLM rows being padded without explicit tokenizer provenance."""

    class FakeTokenizer:
        eos_token = "[SEP]"
        pad_token = "[PAD]"
        pad_token_id = 0

        def __call__(self, texts, **settings):
            assert settings["max_length"] == 128
            return {"input_ids": [[11] + [0] * 127 for _ in texts]}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name, **settings):
            assert name == "bert-base-uncased"
            assert settings == {
                "revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
                "local_files_only": True,
            }
            return FakeTokenizer()

    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoTokenizer=FakeAutoTokenizer)
    )
    root = prepare_conversion_root(tmp_path)
    item = canonical_conversion_request(root, request("rdlm", "lm1b", samples=1))
    output_dir = run_dir(root, item)
    write_capture(output_dir / "upstream_token_ids.json", 1, 127)

    records = list(RDLMAdapter().convert_outputs(item, output_dir))
    metadata = json.loads((output_dir / "conversion_metadata.json").read_text())

    assert records[0].token_ids == [11] + [0] * 127
    assert metadata["token_ids_source"] == "retokenized"
    assert metadata["token_ids_transformation"]["reason"] == "upstream_ids_not_canonical_length"


def test_shared_capture_runs_langflow_with_locked_offline_tokenizer_and_records_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch LangFlow resolving its movable GPT-2 alias or parsing its text file."""

    data_config, downloads, tokenizer_snapshot = prepare_tokenizer_binding(tmp_path, "owt")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    entrypoint = tmp_path / "inference.py"
    (entrypoint.parent / "configs").mkdir()
    source = """
import argparse
import os

class Rows:
    def __init__(self, rows): self.rows = rows
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return self.rows

class LangFlow:
    def generate_samples(self, num_samples, **kwargs):
        return Rows([[index + 1, index + 2] for index in range(num_samples)])

class Tokenizer:
    def batch_decode(self, rows, **kwargs):
        return [f\"text {index}\" for index, _ in enumerate(rows.rows)]

class AutoTokenizer:
    @staticmethod
    def from_pretrained(name, **settings):
        assert name == __SNAPSHOT__
        assert settings == {'local_files_only': True}
        assert os.environ['HF_HUB_OFFLINE'] == '1'
        assert os.environ['TRANSFORMERS_OFFLINE'] == '1'
        return Tokenizer()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = LangFlow().generate_samples(args.num_samples)
    texts = AutoTokenizer.from_pretrained('fixture').batch_decode(rows)
    with open(args.output, 'w', encoding='utf-8') as handle:
        handle.write('\\n'.join(texts))
"""
    entrypoint.write_text(
        source.replace("__SNAPSHOT__", repr(str(tokenizer_snapshot))), encoding="utf-8"
    )
    capture_path = tmp_path / "capture.json"
    output_path = tmp_path / "samples.txt"

    result = capture_module.main(
        [
            f"--upstream-entrypoint={entrypoint}",
            f"--capture-path={capture_path}",
            "--capture-kind=langflow",
            "--expected-samples=2",
            f"--data-config-path={data_config}",
            f"--downloads-manifest-path={downloads}",
            "--dataset-id=owt",
            f"--tokenizer-snapshot={tokenizer_snapshot}",
            "--",
            "--num_samples",
            "2",
            "--output",
            str(output_path),
        ]
    )
    capture = json.loads(capture_path.read_text())

    assert result == 0
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert output_path.read_text() == "text 0\ntext 1"
    assert capture == {
        "schema": "dlb-upstream-token-capture-v1",
        "samples": [
            {"sample_id": 0, "text": "text 0", "token_ids": [1, 2]},
            {"sample_id": 1, "text": "text 1", "token_ids": [2, 3]},
        ],
    }


def test_rdlm_capture_routes_saved_assets_and_uses_bounded_exact_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch movable tokenizers, fixed batches, or fallback assets replacing locked inputs."""

    data_config, downloads, tokenizer_snapshot = prepare_tokenizer_binding(tmp_path, "lm1b")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    saved_config = tmp_path / "config.yaml"
    saved_config.write_text("model:\n  length: 128\ndata:\n  train: lm1b\n", encoding="utf-8")
    saved_sde = tmp_path / "sde.pkl"
    saved_sde.write_bytes(b"saved-sde")
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    capture_path = tmp_path / "capture.json"

    class Rows:
        def __init__(self, rows):
            self.rows = rows

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.rows

    class FakeTorch:
        @staticmethod
        def load(path, **kwargs):
            assert Path(path) == checkpoint
            return {"config": "embedded-config"}

    class FakeSeqUtils:
        @staticmethod
        def find_bos_and_shift_fn(*args, **kwargs):
            def shift(rows):
                return ["one", "two", "excess"], [
                    [1] * 128,
                    [2] * 128,
                    [3] * 128,
                ]

            return shift

    class FakeBertTokenizer:
        @staticmethod
        def from_pretrained(name, **settings):
            assert name == str(tokenizer_snapshot)
            assert settings == {"local_files_only": True}
            assert os.environ["HF_HUB_OFFLINE"] == "1"
            assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
            return object()

    fake_data = SimpleNamespace(
        transformers=SimpleNamespace(BertTokenizer=FakeBertTokenizer),
        AutoTokenizer=FakeBertTokenizer,
    )

    def get_tokenizer(dataset):
        assert dataset == "lm1b"
        return fake_data.transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased"
        )

    fake_data.get_tokenizer = get_tokenizer

    fake_run_sample = SimpleNamespace(
        torch=FakeTorch(),
        sutils=FakeSeqUtils(),
        data=fake_data,
        tqdm=lambda iterable, **kwargs: iterable,
        instantiate=lambda config, **kwargs: (config, kwargs),
    )
    monkeypatch.setitem(sys.modules, "run_sample", fake_run_sample)

    class FakeOmegaConf:
        @staticmethod
        def load(path):
            assert Path(path) == saved_config
            return SimpleNamespace(
                model=SimpleNamespace(length=128),
                data=SimpleNamespace(train="lm1b"),
            )

        @staticmethod
        def select(config, key):
            section, field = key.split(".")
            return getattr(getattr(config, section), field)

    monkeypatch.setitem(
        sys.modules, "omegaconf", SimpleNamespace(OmegaConf=FakeOmegaConf)
    )

    class FakeMP:
        @staticmethod
        def spawn(*args, **kwargs):
            raise AssertionError("capture must replace process spawning")

    fake_entrypoint = SimpleNamespace(mp=FakeMP())

    def fake_main():
        import run_sample

        def worker(rank, marker):
            assert rank == 0 and marker == "worker"
            run_sample.data.get_tokenizer("lm1b")
            state = run_sample.torch.load(checkpoint)
            assert state["config"].model.length == 128
            with run_sample.open(tmp_path / "wrong" / "sde.pkl", "rb") as handle:
                assert handle.read() == b"saved-sde"
            with pytest.raises(ValueError, match="saved SDE"):
                run_sample.instantiate(
                    "sde",
                    manifold=object(),
                    scheduler=object(),
                    prior_dist=object(),
                    device="cpu",
                )
            run_sample.instantiate(
                "sde",
                manifold=object(),
                scheduler=object(),
                prior_dist=object(),
                preprocessed=(object(), object()),
            )
            assert list(run_sample.tqdm(range(8), leave=False)) == [0, 1]
            shift = run_sample.sutils.find_bos_and_shift_fn(3, 10, object())
            shift(Rows([[0], [0]]))

        fake_entrypoint.mp.spawn(worker, args=("worker",), nprocs=1, join=True)

    fake_entrypoint.main = fake_main
    invocation = capture_module.CaptureInvocation(
        entrypoint=tmp_path / "main.py",
        capture_path=capture_path,
        kind="rdlm",
        expected_samples=2,
        saved_config_path=saved_config,
        saved_sde_path=saved_sde,
        data_config_path=data_config,
        downloads_manifest_path=downloads,
        dataset_id="lm1b",
        tokenizer_snapshot=tokenizer_snapshot,
    )

    capture_module._capture_rdlm(
        fake_entrypoint,
        invocation,
        [f"model_path={checkpoint}", "sampling.batch_per_gpu=1"],
    )
    capture = json.loads(capture_path.read_text())

    assert [sample["text"] for sample in capture["samples"]] == ["one", "two"]
    assert capture["samples"][1]["token_ids"] == [2] * 128
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ


def test_continuous_cli_dry_run_emits_two_commands_and_two_registry_rejections(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Catch unsupported cells being substituted or dry-run trying to write artifacts."""

    results_root = ROOT / "results"
    assert not results_root.exists()

    result = command_main(
        [
            "--root",
            str(ROOT),
            "--models",
            "langflow,rdlm",
            "--datasets",
            "lm1b,owt",
            "--dry-run",
        ]
    )
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]

    assert result == 0
    assert len(records) == 4
    supported = [record for record in records if record["status"] == "supported"]
    unsupported = [record for record in records if record["status"] == "unsupported"]
    assert {(record["model"], record["dataset"]) for record in supported} == {
        ("langflow", "owt"),
        ("rdlm", "lm1b"),
    }
    assert {(record["model"], record["dataset"]) for record in unsupported} == {
        ("langflow", "lm1b"),
        ("rdlm", "owt"),
    }
    assert all(record["command"] for record in supported)
    assert all(record["reason"] for record in unsupported)
    assert all(
        override(record["command"], "seed") == "42"
        if record["model"] == "rdlm"
        else option(record["command"], "--seed") == "42"
        for record in supported
    )
    rdlm_command = next(record["command"] for record in supported if record["model"] == "rdlm")
    assert override(rdlm_command, "sampling.batch_per_gpu") == "8"
    assert not results_root.exists()

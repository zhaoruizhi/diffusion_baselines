"""Contracts for standardized, sampler-only generation latency."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import dlb.adapters.capture as capture_module
import dlb.timing as timing_module
from dlb.adapters.capture import CaptureInvocation, TokenizerBinding
from dlb.adapters.base import BaseTeacherAdapter
from dlb.benchmarking import render_benchmark_matrix
from dlb.timing import benchmark, publish_timing


ROOT = Path(__file__).parents[1]


class FakeClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class FakeTensor:
    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.rows


def _server_metadata(tmp_path: Path) -> Path:
    path = tmp_path / "metadata.json"
    path.write_text(
        json.dumps(
            {
                "seed": 42,
                "dataset": "owt",
                "model": "fixture",
                "steps": 32,
                "environment": "dlb-fixture",
                "source_commit": "a" * 40,
                "config_sha256": "c" * 64,
                "checkpoint_sha256": "b" * 64,
                "checkpoint_lock_id": "locked",
                "checkpoint_selection": {"resource": "fixture"},
                "checkpoint_teacher_family": "fixture",
                "adapter_identity": "fixture:v1",
                "requested_precision": "author",
                "precision": "bf16-mixed",
                "precision_policy": "fixture:pinned_internal_bf16_autocast",
            }
        ),
        encoding="utf-8",
    )
    return path


def _fake_cuda(monkeypatch) -> None:
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        timing_module,
        "cuda_runtime_metadata",
        lambda: {
            "gpu_name": "fake-gpu",
            "cuda_runtime": "12.1",
            "driver_version": "535.1",
            "synchronization_policy": "torch.cuda.synchronize_before_start_and_after_generate",
        },
    )


def test_benchmark_excludes_warmups_and_preserves_all_raw_durations() -> None:
    calls: list[str] = []
    clock = FakeClock([value / 10 for value in range(64)])

    result = benchmark(
        lambda: calls.append("generate"),
        lambda: calls.append("sync"),
        warmups=5,
        repeats=32,
        clock=clock,
        batch_size=1,
        mode="primary_latency",
    )

    assert calls.count("generate") == 37
    assert len(result.raw_durations_seconds) == 32
    assert result.raw_durations_seconds == pytest.approx([0.1] * 32)
    assert result.num_timed_samples == 32
    assert result.seconds_per_sample == pytest.approx(0.1)
    assert result.standard_deviation_convention == "population"


def test_benchmark_synchronizes_immediately_around_each_timed_call() -> None:
    events: list[str] = []
    clock_values = iter([1.0, 1.25, 2.0, 2.5])

    def clock() -> float:
        events.append("clock")
        return next(clock_values)

    benchmark(
        lambda: events.append("generate"),
        lambda: events.append("sync"),
        warmups=0,
        repeats=2,
        clock=clock,
    )

    assert events == [
        "sync", "clock", "generate", "sync", "clock",
        "sync", "clock", "generate", "sync", "clock",
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"warmups": True}, "warmups"),
        ({"repeats": 0}, "repeats"),
        ({"batch_size": 2}, "batch size 1"),
        ({"mode": "unknown"}, "mode"),
    ],
)
def test_benchmark_rejects_invalid_primary_inputs(kwargs: dict[str, object], message: str) -> None:
    arguments = {"warmups": 5, "repeats": 32, "batch_size": 1, "mode": "primary_latency"}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        benchmark(lambda: None, lambda: None, **arguments)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0])
def test_benchmark_rejects_invalid_clock_durations(bad: float) -> None:
    with pytest.raises(ValueError, match="clock"):
        benchmark(
            lambda: None,
            lambda: None,
            warmups=0,
            repeats=1,
            clock=FakeClock([0.0, bad]),
        )


def test_publish_timing_is_atomic_and_contains_full_metadata(tmp_path: Path) -> None:
    result = benchmark(
        lambda: None,
        lambda: None,
        warmups=0,
        repeats=2,
        clock=FakeClock([0.0, 0.1, 1.0, 1.3]),
    )
    output = tmp_path / "timing.json"
    metadata = {
        "seed": 42,
        "dataset": "owt",
        "model": "flm",
        "steps": 32,
        "gpu_name": "fake-gpu",
        "cuda_runtime": "12.1",
        "driver_version": "535.1",
        "precision": "bf16-mixed",
        "requested_precision": "author",
        "parameter_precision": "fp32",
        "precision_policy": "flm:pinned_internal_bf16_autocast_with_fp32_sensitive_ops",
        "synchronization_policy": "torch.cuda.synchronize_before_start_and_after_generate",
        "environment": "dlb-flm",
        "source_commit": "a" * 40,
        "config_sha256": "c" * 64,
        "checkpoint_sha256": "b" * 64,
        "checkpoint_lock_id": "locked",
        "checkpoint_selection": {"resource": "fixture"},
        "checkpoint_teacher_family": "continuous_flm",
        "adapter_identity": "fixture:v1",
    }

    publish_timing(output, result, metadata)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "dlb-generation-timing-v1"
    assert payload["timing"]["raw_durations_seconds"] == pytest.approx([0.1, 0.3])
    assert payload["metadata"] == metadata
    assert payload["timing"]["exclusions"] == [
        "model_and_checkpoint_loading",
        "first_compilation",
        "token_decoding",
        "metrics",
        "file_io",
    ]
    assert not list(tmp_path.glob("*.partial"))


def test_publish_timing_never_replaces_complete_result_on_invalid_metadata(tmp_path: Path) -> None:
    output = tmp_path / "timing.json"
    output.write_text('{"old":true}', encoding="utf-8")
    result = benchmark(
        lambda: None,
        lambda: None,
        warmups=0,
        repeats=1,
        clock=FakeClock([0.0, 0.1]),
    )

    with pytest.raises(ValueError, match="metadata"):
        publish_timing(output, result, {"model": "flm"})

    assert output.read_text(encoding="utf-8") == '{"old":true}'
    assert not list(tmp_path.glob("*.partial"))


def test_server_benchmark_records_parameter_dtype_without_mislabeling_compute_precision(
    monkeypatch, tmp_path: Path
) -> None:
    _fake_cuda(monkeypatch)

    class Parameter:
        dtype = "torch.float32"

    class Model:
        def parameters(self):
            return iter([Parameter()])

    calls: list[str] = []
    output = tmp_path / "mismatch.json"
    timing_module.benchmark_and_publish(
        lambda: calls.append("generate"),
        model=Model(),
        output=output,
        metadata_path=_server_metadata(tmp_path),
        precision="author",
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert calls == ["generate"] * 37
    assert payload["metadata"]["requested_precision"] == "author"
    assert payload["metadata"]["parameter_precision"] == "fp32"
    assert payload["metadata"]["precision"] == "bf16-mixed"
    assert payload["metadata"]["precision_policy"] == (
        "fixture:pinned_internal_bf16_autocast"
    )


def test_every_registry_cell_has_a_concrete_benchmark_command_or_structured_skip() -> None:
    records = render_benchmark_matrix(
        root=ROOT,
        models=None,
        datasets=("lm1b", "owt"),
        steps=32,
        seed=42,
        precision="author",
        output_root=ROOT / "results/timing",
        dry_run=True,
    )

    supported = [record for record in records if record["status"] == "supported"]
    skipped = [record for record in records if record["status"] == "unsupported"]
    assert len(supported) == 20
    assert {(record["model"], record["dataset"]) for record in skipped} == {
        ("langflow", "lm1b"),
        ("rdlm", "owt"),
    }
    for record in supported:
        command = record["command"]
        assert isinstance(command, list) and command
        assert "--benchmark-output" in " ".join(command)
        assert "--benchmark-metadata" in " ".join(command)
        assert record["hook"] in {
            "teacher.generate_samples",
            "mdlm._sample",
            "langflow.generate_samples",
            "rdlm.sampling_fn",
            "distilled.model.sample",
        }
        assert record["batch_size"] == 1
        assert record["warmups"] == 5
        assert record["repeats"] == 32
        assert record["precision"] == "bf16-mixed"
        policy_owner = {
            "fmlm": "flm",
            "duo_dcd": "duo",
            "duo_di4c": "di4c",
            "mdlm_di4c": "di4c",
            "mdlm_sdtt": "sdtt",
        }.get(record["model"], record["model"])
        assert str(record["precision_policy"]).startswith(policy_owner + ":")


def test_benchmark_matrix_calls_adapter_benchmark_hook_not_plain_sampling(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("plain render_command must not be the matrix integration boundary")

    monkeypatch.setattr(BaseTeacherAdapter, "render_command", forbidden)

    records = render_benchmark_matrix(
        root=ROOT,
        models=("langflow",),
        datasets=("owt",),
        steps=32,
        seed=42,
        precision="author",
        output_root=ROOT / "results/timing",
        dry_run=True,
    )

    assert records[0]["status"] == "supported"
    assert records[0]["hook"] == "langflow.generate_samples"


def test_teacher_capture_times_real_generate_only_after_loaded_model(monkeypatch, tmp_path: Path) -> None:
    _fake_cuda(monkeypatch)
    state = {"loaded": False, "calls": 0}

    class Owner:
        class Parameter:
            dtype = "torch.bfloat16"

        def parameters(self):
            return iter([self.Parameter()])

        def restore_model_and_sample(self, num_steps, eps=1e-5):
            raise AssertionError("benchmark must bypass the restore/decode wrapper")

        def _eval_mode(self):
            assert state["loaded"]

        def _train_mode(self):
            pass

        def generate_samples(self, *, num_samples, num_steps, eps):
            assert state["loaded"] and num_samples == 1 and num_steps == 32
            state["calls"] += 1
            return FakeTensor([[state["calls"]]])

    module = type("Module", (), {})()
    module.algo = type("Algo", (), {"trainer_base": type("TB", (), {"TrainerBase": Owner})})
    module.run = lambda: setattr(type("Marker", (), {}), "unused", True)
    output = tmp_path / "teacher.json"
    invocation = CaptureInvocation(
        entrypoint=ROOT / "upstreams/flm/main.py",
        capture_path=tmp_path / "capture.json",
        benchmark_output=output,
        benchmark_metadata=_server_metadata(tmp_path),
        benchmark_precision="author",
    )

    def run_main(module, entrypoint, forwarded):
        state["loaded"] = True
        module.algo.trainer_base.TrainerBase().restore_model_and_sample(num_steps=32)

    monkeypatch.setattr(capture_module, "_run_main", run_main)
    capture_module._capture_teacher(module, invocation, [])

    assert state["calls"] == 37
    assert json.loads(output.read_text())["schema"] == "dlb-generation-timing-v1"


def test_langflow_capture_times_loaded_generate_samples_before_decode(monkeypatch, tmp_path: Path) -> None:
    _fake_cuda(monkeypatch)
    state = {"loaded": False, "calls": 0, "decoded": 0}

    class Model:
        class Parameter:
            dtype = "torch.float16"

        def parameters(self):
            return iter([self.Parameter()])

        def generate_samples(self, **kwargs):
            assert state["loaded"] and kwargs["num_samples"] == 1
            state["calls"] += 1
            return FakeTensor([[state["calls"]]])

    class LoadedTokenizer:
        def batch_decode(self, rows, **kwargs):
            state["decoded"] += 1
            return ["decoded"]

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return LoadedTokenizer()

    module = type("Module", (), {"LangFlow": Model, "AutoTokenizer": AutoTokenizer})
    output = tmp_path / "langflow.json"
    invocation = CaptureInvocation(
        entrypoint=ROOT / "upstreams/langflow/inference.py",
        capture_path=tmp_path / "capture.json",
        kind="langflow",
        expected_samples=1,
        benchmark_output=output,
        benchmark_metadata=_server_metadata(tmp_path),
        benchmark_precision="author",
    )

    def run_main(module, entrypoint, forwarded):
        state["loaded"] = True
        model = module.LangFlow()
        tokenizer = module.AutoTokenizer.from_pretrained("ignored")
        result = model.generate_samples(num_samples=1, num_steps=32)
        tokenizer.batch_decode(result.tolist())

    monkeypatch.setattr(capture_module, "_run_main", run_main)
    capture_module._capture_langflow(
        module,
        invocation,
        [],
        TokenizerBinding("gpt2", "a" * 40, tmp_path),
    )

    assert state["calls"] == 37
    assert state["decoded"] == 1
    assert json.loads(output.read_text())["timing"]["repeats"] == 32


def test_rdlm_sampling_hook_times_the_real_sampling_function_once(monkeypatch, tmp_path: Path) -> None:
    _fake_cuda(monkeypatch)
    calls: list[object] = []
    output = tmp_path / "rdlm.json"
    invocation = CaptureInvocation(
        entrypoint=ROOT / "upstreams/rdlm/main.py",
        capture_path=tmp_path / "capture.json",
        kind="rdlm",
        expected_samples=1,
        benchmark_output=output,
        benchmark_metadata=_server_metadata(tmp_path),
        benchmark_precision="author",
    )

    def real_sampling(model):
        calls.append(model)
        return FakeTensor([[len(calls)]])

    hooked = capture_module._rdlm_benchmark_sampling_fn(real_sampling, invocation)
    class Model:
        class Parameter:
            dtype = "torch.float32"

        def parameters(self):
            return iter([self.Parameter()])

    model = Model()
    hooked(model)

    assert calls == [model] * 37
    assert json.loads(output.read_text())["metadata"]["gpu_name"] == "fake-gpu"


def test_distilled_runtime_times_materialized_model_sample(monkeypatch, tmp_path: Path) -> None:
    import importlib.util

    _fake_cuda(monkeypatch)
    specification = importlib.util.spec_from_file_location(
        "task11_distilled_runtime", ROOT / "adapters/_distilled_runtime.py"
    )
    assert specification is not None and specification.loader is not None
    runtime = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(runtime)

    class Model:
        def __init__(self):
            self.calls: list[dict[str, object]] = []

        class Parameter:
            dtype = "torch.bfloat16"

        def parameters(self):
            return iter([self.Parameter()])

        def sample(self, **kwargs):
            self.calls.append(kwargs)
            return FakeTensor([[len(self.calls)]])

    model = Model()
    output = tmp_path / "distilled.json"
    runtime.benchmark_model(
        model=model,
        output=output,
        metadata_path=_server_metadata(tmp_path),
        precision="author",
        num_steps=8,
        seq_len=128,
        sampler="ancestral",
    )

    assert len(model.calls) == 37
    assert all(call["n_samples"] == 1 and call["verbose"] is False for call in model.calls)
    assert json.loads(output.read_text())["timing"]["warmups"] == 5

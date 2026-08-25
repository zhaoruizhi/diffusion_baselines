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
import dlb.benchmarking as benchmarking_module
from dlb.benchmarking import render_benchmark_matrix, run_timing_attempt
from dlb.command import ADAPTERS
from dlb.runner import RunRequest
from dlb.timing import benchmark, publish_timing


ROOT = Path(__file__).parents[1]


def _prepare_benchmark_root(tmp_path: Path) -> Path:
    for relative in (
        Path("artifacts/data.yaml"),
        Path("artifacts/checkpoints.yaml"),
        Path("configs/experiments.yaml"),
        Path("configs/sampling/di4c_mdlm_owt.yaml"),
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((ROOT / relative).read_text(encoding="utf-8"), encoding="utf-8")
    for relative in (
        Path("upstreams/flm/main.py"),
        Path("upstreams/duo/main.py"),
        Path("upstreams/mdlm/main.py"),
        Path("upstreams/candi/main.py"),
        Path("upstreams/langflow/inference.py"),
        Path("upstreams/rdlm/main.py"),
        Path("upstreams/sdtt/src/sdtt/main.py"),
        Path("upstreams/di4c/sdtt/src/sdtt/main.py"),
        Path("adapters/sample_langflow.py"),
        Path("adapters/sample_sdtt.py"),
        Path("adapters/sample_di4c.py"),
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# fixture\n", encoding="utf-8")
    return tmp_path


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
                "precision_evidence": "static_policy_bound_to_checkpoint_and_runtime_code_not_runtime_autocast_observation",
                "precision_policy_binding": {
                    "model": "fixture",
                    "dataset": "owt",
                    "checkpoint_sha256": "b" * 64,
                },
                "attempt_id": "f" * 32,
            }
        ),
        encoding="utf-8",
    )
    return path


def _complete_metadata(tmp_path: Path, attempt_id: str = "f" * 32) -> dict[str, object]:
    value = json.loads(_server_metadata(tmp_path).read_text(encoding="utf-8"))
    value.update(
        {
            "attempt_id": attempt_id,
            "parameter_precision": "fp32",
            "gpu_name": "fake-gpu",
            "gpu_index": 0,
            "gpu_compute_capability": "8.0",
            "cuda_runtime_version": "12.4",
            "pytorch_compiled_cuda_toolkit": "12.1",
            "nvidia_driver_version": "535.104.05",
            "runtime_code_binding": [
                {"path": str(Path(__file__).resolve()), "sha256": "d" * 64}
            ],
            "synchronization_policy": "torch.cuda.synchronize_before_start_and_after_generate",
        }
    )
    return value


def _fake_cuda(monkeypatch) -> None:
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        timing_module,
        "cuda_runtime_metadata",
        lambda: {
            "gpu_name": "fake-gpu",
            "gpu_index": 0,
            "gpu_compute_capability": "8.0",
            "cuda_runtime_version": "12.4",
            "pytorch_compiled_cuda_toolkit": "12.1",
            "nvidia_driver_version": "535.104.05",
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
    metadata = _complete_metadata(tmp_path)
    metadata.update(
        {
            "model": "flm",
            "environment": "dlb-flm",
            "checkpoint_teacher_family": "continuous_flm",
            "precision_policy": "flm:pinned_static_policy_bound_at_execution",
        }
    )

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


def _write_staged_timing(path: Path, metadata: dict[str, object]) -> None:
    result = benchmark(
        lambda: None,
        lambda: None,
        warmups=0,
        repeats=32,
        clock=FakeClock([value / 10 for value in range(64)]),
    )
    # Controller acceptance requires the production 5-warmup declaration even
    # though this CPU fixture avoids doing redundant warmup work.
    result = timing_module.TimingResult(**{**result.__dict__, "warmups": 5})
    publish_timing(path, result, metadata)


def test_attempt_promotes_staged_timing_only_after_subprocess_success(monkeypatch, tmp_path: Path) -> None:
    attempt = "1" * 32
    final = tmp_path / "timing.json"
    staged = tmp_path / f".timing.json.{attempt}.staged.json"
    expected = _complete_metadata(tmp_path, attempt)

    def successful(command, cwd, check):
        _write_staged_timing(staged, expected)
        assert not final.exists()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(benchmarking_module.subprocess, "run", successful)
    observed = run_timing_attempt(
        ["fake-sampler"],
        cwd=tmp_path,
        final_output=final,
        staged_output=staged,
        expected_metadata=expected,
        attempt_id=attempt,
    )

    assert observed == final
    assert json.loads(final.read_text())["metadata"]["attempt_id"] == attempt
    assert not staged.exists()


def test_attempt_failure_retires_old_result_and_cleans_staged_output(monkeypatch, tmp_path: Path) -> None:
    attempt = "2" * 32
    final = tmp_path / "timing.json"
    final.write_text('{"schema":"old-complete"}', encoding="utf-8")
    staged = tmp_path / f".timing.json.{attempt}.staged.json"
    expected = _complete_metadata(tmp_path, attempt)

    def failing(command, cwd, check):
        _write_staged_timing(staged, expected)
        return SimpleNamespace(returncode=9)

    monkeypatch.setattr(benchmarking_module.subprocess, "run", failing)
    with pytest.raises(RuntimeError, match="status 9"):
        run_timing_attempt(
            ["fake-sampler"],
            cwd=tmp_path,
            final_output=final,
            staged_output=staged,
            expected_metadata=expected,
            attempt_id=attempt,
        )

    assert not final.exists()
    assert not staged.exists()
    superseded = list(tmp_path.glob(".timing.json.*.superseded.json"))
    assert len(superseded) == 1
    assert superseded[0].read_text(encoding="utf-8") == '{"schema":"old-complete"}'


def test_attempt_cannot_reuse_an_old_final_when_no_fresh_stage_is_written(monkeypatch, tmp_path: Path) -> None:
    attempt = "3" * 32
    final = tmp_path / "timing.json"
    final.write_text('{"schema":"old-complete"}', encoding="utf-8")
    staged = tmp_path / f".timing.json.{attempt}.staged.json"
    monkeypatch.setattr(
        benchmarking_module.subprocess,
        "run",
        lambda command, cwd, check: SimpleNamespace(returncode=0),
    )

    with pytest.raises(RuntimeError, match="fresh staged timing"):
        run_timing_attempt(
            ["fake-sampler"],
            cwd=tmp_path,
            final_output=final,
            staged_output=staged,
            expected_metadata=_complete_metadata(tmp_path, attempt),
            attempt_id=attempt,
        )

    assert not final.exists()
    assert not staged.exists()


def test_attempt_rejects_mismatched_attempt_provenance_and_cleans_partial(
    monkeypatch, tmp_path: Path
) -> None:
    attempt = "4" * 32
    final = tmp_path / "timing.json"
    staged = tmp_path / f".timing.json.{attempt}.staged.json"
    expected = _complete_metadata(tmp_path, attempt)

    def mismatched(command, cwd, check):
        _write_staged_timing(staged, {**expected, "attempt_id": "5" * 32})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(benchmarking_module.subprocess, "run", mismatched)
    with pytest.raises(RuntimeError, match="attempt provenance"):
        run_timing_attempt(
            ["fake-sampler"],
            cwd=tmp_path,
            final_output=final,
            staged_output=staged,
            expected_metadata=expected,
            attempt_id=attempt,
        )

    assert not final.exists()
    assert not staged.exists()


def test_driver_version_uses_nvidia_semantic_version_not_cuda_api_integer(monkeypatch) -> None:
    monkeypatch.setattr(
        timing_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="535.104.05\n", returncode=0),
    )
    assert timing_module._driver_version() == "535.104.05"

    monkeypatch.setattr(
        timing_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="12040\n", returncode=0),
    )
    with pytest.raises(RuntimeError, match="semantic NVIDIA driver"):
        timing_module._driver_version()


def test_driver_query_failure_fails_closed(monkeypatch) -> None:
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(timing_module.subprocess, "run", unavailable)
    with pytest.raises(RuntimeError, match="NVIDIA driver version"):
        timing_module._driver_version()


def test_cuda_metadata_distinguishes_loaded_runtime_from_compiled_toolkit(monkeypatch) -> None:
    class Cudart:
        @staticmethod
        def cudaRuntimeGetVersion():
            return 12040

    fake_torch = SimpleNamespace(
        version=SimpleNamespace(cuda="12.1"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 2,
            get_device_name=lambda device: f"fake-gpu-{device}",
            get_device_capability=lambda device: (8, 0),
            cudart=lambda: Cudart(),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(timing_module, "_driver_version", lambda: "535.104.05")

    assert timing_module.cuda_runtime_metadata() == {
        "gpu_name": "fake-gpu-2",
        "gpu_index": 2,
        "gpu_compute_capability": "8.0",
        "cuda_runtime_version": "12.4",
        "pytorch_compiled_cuda_toolkit": "12.1",
        "nvidia_driver_version": "535.104.05",
        "synchronization_policy": "torch.cuda.synchronize_before_start_and_after_generate",
    }


def test_cuda_metadata_falls_back_when_torch_cudart_lacks_runtime_version(
    monkeypatch,
) -> None:
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(cuda="12.1"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 1,
            get_device_name=lambda device: f"fallback-gpu-{device}",
            get_device_capability=lambda device: (8, 9),
            cudart=lambda: SimpleNamespace(),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(timing_module, "_driver_version", lambda: "535.104.05")
    monkeypatch.setattr(
        timing_module, "_cuda_runtime_version_from_ctypes", lambda: 12040
    )

    assert timing_module.cuda_runtime_metadata() == {
        "gpu_name": "fallback-gpu-1",
        "gpu_index": 1,
        "gpu_compute_capability": "8.9",
        "cuda_runtime_version": "12.4",
        "pytorch_compiled_cuda_toolkit": "12.1",
        "nvidia_driver_version": "535.104.05",
        "synchronization_policy": "torch.cuda.synchronize_before_start_and_after_generate",
    }


def test_cuda_metadata_fails_closed_without_loaded_runtime_evidence(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(cuda="12.1"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_name=lambda device: "fake-gpu",
            get_device_capability=lambda device: (8, 0),
            cudart=lambda: SimpleNamespace(),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(timing_module, "_driver_version", lambda: "535.104.05")
    monkeypatch.setattr(
        timing_module,
        "_cuda_runtime_version_from_ctypes",
        lambda: (_ for _ in ()).throw(RuntimeError("ctypes unavailable")),
    )

    with pytest.raises(RuntimeError, match="loaded CUDA runtime"):
        timing_module.cuda_runtime_metadata()


def test_flm_precision_policy_binds_checkpoint_revision_and_config_bytes(
    monkeypatch, tmp_path: Path
) -> None:
    destination = "official/flm/owt"
    config_path = tmp_path / "checkpoints" / destination / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "torch_dtype": "float32",
                "architectures": ["DiTForDiffusionLM"],
                "auto_map": {"AutoModelForMaskedLM": "modeling_dit.DiTForDiffusionLM"},
            }
        ),
        encoding="utf-8",
    )
    resource = SimpleNamespace(
        destination=destination,
        source=SimpleNamespace(
            repo_id="owner/flm",
            revision="e" * 40,
        ),
    )
    monkeypatch.setattr(
        benchmarking_module,
        "load_checkpoint_manifest",
        lambda path: SimpleNamespace(resources={"flm_owt_hf": resource}),
    )
    request = RunRequest(
        run_id="precision-fixture",
        model_id="flm",
        dataset_id="owt",
        step_count=32,
        seed=42,
        sample_count=1,
        config_sha256="c" * 64,
        source_sha256="a" * 40,
        checkpoint_sha256="b" * 64,
        checkpoint_lock_id="locked",
        checkpoint_selection={"resource": "flm_owt_hf"},
        checkpoint_teacher_family="continuous_flm",
        adapter_identity="fixture:v1",
        environment="dlb-flm",
    )

    binding = benchmarking_module._precision_policy_binding(
        tmp_path, request, ADAPTERS["flm"]
    )

    assert binding["checkpoint_revision"] == "e" * 40
    assert binding["checkpoint_config_torch_dtype"] == "float32"
    assert binding["checkpoint_config_sha256"] == timing_module._sha256_path(config_path)
    assert binding["checkpoint_config_auto_map"] == {
        "AutoModelForMaskedLM": "modeling_dit.DiTForDiffusionLM"
    }
    metadata = benchmarking_module._metadata(
        tmp_path, request, ADAPTERS["flm"], "f" * 32
    )
    assert metadata["precision"] == "bf16-mixed-static-author-policy"
    assert metadata["precision_evidence"].endswith(
        "not_runtime_autocast_observation"
    )


def test_flm_precision_policy_fails_closed_without_checkpoint_config(
    monkeypatch, tmp_path: Path
) -> None:
    resource = SimpleNamespace(
        destination="official/flm/owt",
        source=SimpleNamespace(repo_id="owner/flm", revision="e" * 40),
    )
    monkeypatch.setattr(
        benchmarking_module,
        "load_checkpoint_manifest",
        lambda path: SimpleNamespace(resources={"flm_owt_hf": resource}),
    )
    request = RunRequest(
        run_id="precision-fixture",
        model_id="flm",
        dataset_id="owt",
        step_count=32,
        seed=42,
        sample_count=1,
        config_sha256="c" * 64,
        source_sha256="a" * 40,
        checkpoint_sha256="b" * 64,
        checkpoint_lock_id="locked",
        checkpoint_selection={"resource": "flm_owt_hf"},
        checkpoint_teacher_family="continuous_flm",
        adapter_identity="fixture:v1",
        environment="dlb-flm",
    )

    with pytest.raises(ValueError, match="downloaded checkpoint config"):
        benchmarking_module._precision_policy_binding(tmp_path, request, ADAPTERS["flm"])


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
    assert payload["metadata"]["precision_evidence"].endswith(
        "not_runtime_autocast_observation"
    )
    assert payload["metadata"]["runtime_code_binding"]
    assert payload["metadata"]["runtime_code_binding"][0]["sha256"]


def test_runtime_code_binding_includes_the_loaded_checkpoint_model_class() -> None:
    class RemoteCheckpointModel:
        pass

    class Trainer:
        def __init__(self):
            self.model = RemoteCheckpointModel()

    records = timing_module.runtime_code_binding(Trainer(), lambda: None)

    assert any("RemoteCheckpointModel" in record["symbol"] for record in records)
    assert all(len(record["sha256"]) == 64 for record in records)


def test_every_registry_cell_has_a_concrete_benchmark_command_or_structured_skip(
    tmp_path: Path,
) -> None:
    root = _prepare_benchmark_root(tmp_path)
    records = render_benchmark_matrix(
        root=root,
        models=None,
        datasets=("lm1b", "owt"),
        steps=32,
        seed=42,
        precision="author",
        output_root=root / "results/timing",
        dry_run=True,
    )

    supported = [record for record in records if record["status"] == "supported"]
    skipped = [record for record in records if record["status"] == "unsupported"]
    errors = [record for record in records if record["status"] == "error"]
    assert len(supported) == 20
    assert {(record["model"], record["dataset"]) for record in skipped} == {
        ("rdlm", "owt"),
    }
    assert [(record["model"], record["dataset"]) for record in errors] == [
        ("rdlm", "lm1b")
    ]
    assert "allowed: 1000,1024" in errors[0]["reason"]
    for record in supported:
        command = record["command"]
        assert isinstance(command, list) and command
        assert "--benchmark-output" in " ".join(command)
        assert "--benchmark-metadata" in " ".join(command)
        assert ".timing.json." + ("0" * 32) + ".staged.json" in " ".join(command)
        assert ".benchmark_metadata." + ("0" * 32) + ".json" in " ".join(command)
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
        if record["model"] in {"flm", "fmlm"}:
            assert record["precision"] == "resolved-from-checkpoint-config-at-execution"
        else:
            assert record["precision"] == "bf16-mixed"
        policy_owner = {
            "fmlm": "flm",
            "duo_dcd": "duo",
            "duo_di4c": "di4c",
            "mdlm_di4c": "di4c",
            "mdlm_sdtt": "sdtt",
        }.get(record["model"], record["model"])
        assert str(record["precision_policy"]).startswith(policy_owner + ":")


def test_benchmark_matrix_calls_adapter_benchmark_hook_not_plain_sampling(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("plain render_command must not be the matrix integration boundary")

    monkeypatch.setattr(BaseTeacherAdapter, "render_command", forbidden)
    root = _prepare_benchmark_root(tmp_path)

    records = render_benchmark_matrix(
        root=root,
        models=("langflow",),
        datasets=("owt",),
        steps=32,
        seed=42,
        precision="author",
        output_root=root / "results/timing",
        dry_run=True,
    )

    assert records[0]["status"] == "supported"
    assert records[0]["hook"] == "langflow.generate_samples"


def test_benchmark_matrix_accepts_rdlm_official_default_step(tmp_path: Path) -> None:
    """Catch benchmark dry-run accepting only rejection paths for RDLM."""

    root = _prepare_benchmark_root(tmp_path)
    records = render_benchmark_matrix(
        root=root,
        models=("rdlm",),
        datasets=("lm1b",),
        steps=1000,
        seed=42,
        precision="author",
        output_root=root / "results/timing",
        dry_run=True,
    )

    assert records[0]["status"] == "supported"
    assert records[0]["hook"] == "rdlm.sampling_fn"
    assert "sampling.steps=1000" in records[0]["command"]


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

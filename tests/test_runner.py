import json
import hashlib
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

import dlb.runner as runner_module
from dlb.runner import RunRequest, run_experiment


class FakeAdapter:
    def __init__(
        self, command: list[str], records: list[dict[str, object]], identity: str = "tests.fake_adapter:v1"
    ) -> None:
        self.command = command
        self.records = records
        self.identity = identity
        self.calls = 0

    def build_command(self, request: RunRequest, run_dir: Path) -> list[str]:
        del request, run_dir
        return self.command

    def convert_outputs(self, request: RunRequest, run_dir: Path) -> list[dict[str, object]]:
        del request, run_dir
        self.calls += 1
        return self.records


def make_records(count: int) -> list[dict[str, object]]:
    return [
        {
            "sample_id": index,
            "text": f"sample {index}",
            "token_ids": [index + 1],
            "seed": 42,
            "generation_seconds": 0.25,
        }
        for index in range(count)
    ]


def make_request(command: list[str] | None = None) -> RunRequest:
    return RunRequest(
        run_id="flm-lm1b-steps-2",
        model_id="flm",
        dataset_id="lm1b",
        step_count=2,
        seed=42,
        sample_count=2,
        command=command,
    )


def prepare_canonical_root(
    root: Path, *, source_sha: str = "a" * 40, checkpoint_payload: bytes = b"checkpoint"
) -> None:
    """Create the minimal canonical provenance inputs for the FLM/LM1B cell."""

    project_root = Path(__file__).parents[1]
    (root / "configs").mkdir(exist_ok=True)
    (root / "artifacts").mkdir(exist_ok=True)
    (root / "configs" / "experiments.yaml").write_text(
        (project_root / "configs" / "experiments.yaml").read_text()
    )
    manifest = root / "artifacts" / "checkpoints.yaml"
    manifest.write_text((project_root / "artifacts" / "checkpoints.yaml").read_text())
    (root / "artifacts" / "source_lock.json").write_text(
        json.dumps({"sources": {"flm": {"commit": source_sha}}})
    )
    checkpoint = root / "checkpoints" / "official" / "flm_ckpt" / "lm1b" / "lm1b_flm.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(checkpoint_payload)


def test_runner_records_command_failure(tmp_path: Path) -> None:
    """Catch child failures that lose the exit code or diagnostic logs."""

    command = [sys.executable, "-c", "import sys; print('out'); print('bad', file=sys.stderr); sys.exit(7)"]
    prepare_canonical_root(tmp_path)
    result = run_experiment(make_request(command), tmp_path, adapter=FakeAdapter(command, make_records(2)))

    assert result.status == "failed"
    assert result.returncode == 7
    failure = json.loads((result.run_dir / "failure.json").read_text())
    assert failure["stage"] == "command"
    assert failure["exit_code"] == 7
    assert failure["stderr_tail"] == "bad\n"
    assert (result.run_dir / "stdout.log").read_text() == "out\n"


def test_runner_skips_only_matching_valid_publication(tmp_path: Path) -> None:
    """Catch resume logic that trusts metadata without checking output and provenance."""

    prepare_canonical_root(tmp_path)
    command = [sys.executable, "-c", "print('ok')"]
    adapter = FakeAdapter(command, make_records(2))
    request = make_request(command)

    first = run_experiment(request, tmp_path, adapter=adapter)
    second = run_experiment(request, tmp_path, adapter=adapter)

    assert first.status == "succeeded"
    assert second.status == "skipped"
    assert adapter.calls == 1
    metadata = json.loads((first.run_dir / "run_metadata.json").read_text())
    assert metadata["command"] == command
    assert metadata["command_sha256"]
    assert metadata["source_sha256"] == "a" * 40
    assert metadata["checkpoint_lock_id"].startswith("recipe:flm_lm1b_official_ckpt:")


def test_runner_reruns_when_checkpoint_identity_changes(tmp_path: Path) -> None:
    """Catch stale reuse after the checkpoint lock resolves to different content."""

    prepare_canonical_root(tmp_path)
    command = [sys.executable, "-c", "print('ok')"]
    adapter = FakeAdapter(command, make_records(2))
    first_request = make_request(command)
    assert run_experiment(first_request, tmp_path, adapter=adapter).status == "succeeded"
    prepare_canonical_root(tmp_path, checkpoint_payload=b"changed-checkpoint")
    assert run_experiment(first_request, tmp_path, adapter=adapter).status == "succeeded"
    assert adapter.calls == 2


def test_runner_does_not_publish_metadata_after_conversion_failure(tmp_path: Path) -> None:
    """Catch a success marker published before converted samples satisfy the contract."""

    prepare_canonical_root(tmp_path)
    command = [sys.executable, "-c", "print('ok')"]
    result = run_experiment(
        make_request(command),
        tmp_path,
        adapter=FakeAdapter(command, [{**make_records(1)[0], "text": ""}]),
    )

    assert result.status == "failed"
    assert result.returncode == 1
    assert not (result.run_dir / "run_metadata.json").exists()
    assert json.loads((result.run_dir / "failure.json").read_text())["stage"] == "conversion"


def test_failed_rerun_removes_stale_success_metadata(tmp_path: Path) -> None:
    """Catch a failed rerun retaining an old success marker for different provenance."""

    prepare_canonical_root(tmp_path)
    command = [sys.executable, "-c", "print('ok')"]
    first_request = make_request(command)
    assert run_experiment(
        first_request, tmp_path, adapter=FakeAdapter(command, make_records(2))
    ).status == "succeeded"
    prepare_canonical_root(tmp_path, source_sha="x" * 40)

    result = run_experiment(
        first_request,
        tmp_path,
        adapter=FakeAdapter(command, [{**make_records(1)[0], "text": ""}]),
    )

    assert result.status == "failed"
    assert not (result.run_dir / "run_metadata.json").exists()


def test_runner_refuses_symlinked_log_target(tmp_path: Path) -> None:
    """Catch a prepared log symlink that would truncate a file outside the run."""

    prepare_canonical_root(tmp_path)
    command = [sys.executable, "-c", "print('ok')"]
    request = make_request(command)
    run_dir = tmp_path / "results" / "samples" / "lm1b" / "flm" / "steps_2"
    run_dir.mkdir(parents=True)
    outside = tmp_path / "outside.log"
    outside.write_text("sentinel\n")
    (run_dir / "stdout.log").symlink_to(outside)

    try:
        run_experiment(request, tmp_path, adapter=FakeAdapter(command, make_records(2)))
    except ValueError as error:
        assert "symlink" in str(error)
    else:
        raise AssertionError("symlinked log target was accepted")
    assert outside.read_text() == "sentinel\n"


def test_runner_resolves_checkpoint_identity_from_coverage_lock(tmp_path: Path) -> None:
    """Catch CLI requests that cannot bind a coverage cell to its checkpoint lock."""

    project_root = Path(__file__).parents[1]
    (tmp_path / "configs").mkdir()
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "configs" / "experiments.yaml").write_text(
        (project_root / "configs" / "experiments.yaml").read_text()
    )
    checkpoints_manifest = tmp_path / "artifacts" / "checkpoints.yaml"
    checkpoints_manifest.write_text((project_root / "artifacts" / "checkpoints.yaml").read_text())
    (tmp_path / "artifacts" / "source_lock.json").write_text(
        json.dumps({"sources": {"duo": {"commit": "a" * 40}}})
    )
    files = [
        {
            "path": "checkpoints/reference_reproduction/flm_baselines/lm1b/lm1b_Duo.ckpt",
            "size_bytes": 3,
            "sha256": "b" * 64,
        }
    ]
    (tmp_path / "artifacts" / "checkpoint_lock.json").write_text(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(checkpoints_manifest.read_bytes()).hexdigest(),
                "resources": {
                    "flm_lm1b_reproductions": {
                        "status": "downloaded",
                        "files": files,
                    }
                },
            }
        )
    )
    command = [sys.executable, "-c", "print('ok')"]
    request = RunRequest(
        run_id="duo-lm1b-steps-2",
        model_id="duo",
        dataset_id="lm1b",
        step_count=2,
        seed=42,
        sample_count=2,
        command=command,
    )

    result = run_experiment(request, tmp_path, adapter=FakeAdapter(command, make_records(2)))

    metadata = json.loads((result.run_dir / "run_metadata.json").read_text())
    manifest_sha = hashlib.sha256(checkpoints_manifest.read_bytes()).hexdigest()
    expected = hashlib.sha256(
        json.dumps(
            {
                "manifest_sha256": manifest_sha,
                "selector": {
                    "resource": "flm_lm1b_reproductions",
                    "path": "lm1b_Duo.ckpt",
                    "teacher_family": "uniform_duo",
                },
                "files": files,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert metadata["checkpoint_sha256"] == expected
    assert (
        metadata["checkpoint_lock_id"]
        == "flm_lm1b_reproductions:" + manifest_sha + ":lm1b_Duo.ckpt"
    )


def test_runner_resolves_recipe_checkpoint_identity(tmp_path: Path) -> None:
    """Catch supported recipe-backed cells that cannot acquire a checkpoint identity."""

    project_root = Path(__file__).parents[1]
    (tmp_path / "configs").mkdir()
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "configs" / "experiments.yaml").write_text(
        (project_root / "configs" / "experiments.yaml").read_text()
    )
    (tmp_path / "artifacts" / "checkpoints.yaml").write_text(
        (project_root / "artifacts" / "checkpoints.yaml").read_text()
    )
    (tmp_path / "artifacts" / "source_lock.json").write_text(
        json.dumps({"sources": {"candi": {"commit": "a" * 40}}})
    )
    checkpoint = tmp_path / "checkpoints" / "reference_reproduction" / "candi" / "owt" / "model.bin"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"model")
    command = [sys.executable, "-c", "print('ok')"]
    request = RunRequest(
        run_id="candi-owt-steps-2",
        model_id="candi",
        dataset_id="owt",
        step_count=2,
        seed=42,
        sample_count=2,
        command=command,
    )

    result = run_experiment(request, tmp_path, adapter=FakeAdapter(command, make_records(2)))

    metadata = json.loads((result.run_dir / "run_metadata.json").read_text())
    assert metadata["checkpoint_lock_id"].startswith("recipe:candi_owt:")
    assert metadata["checkpoint_sha256"]


def test_recipe_checkpoint_identity_ignores_empty_runtime_logs(tmp_path: Path) -> None:
    """Catch successful recipe outputs being rejected because Hydra wrote empty logs."""

    project_root = Path(__file__).parents[1]
    (tmp_path / "configs").mkdir()
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "configs" / "experiments.yaml").write_text(
        (project_root / "configs" / "experiments.yaml").read_text()
    )
    (tmp_path / "artifacts" / "checkpoints.yaml").write_text(
        (project_root / "artifacts" / "checkpoints.yaml").read_text()
    )
    (tmp_path / "artifacts" / "source_lock.json").write_text(
        json.dumps({"sources": {"candi": {"commit": "a" * 40}}})
    )
    output = tmp_path / "checkpoints" / "reference_reproduction" / "candi" / "owt"
    output.mkdir(parents=True)
    (output / "model.bin").write_bytes(b"model")
    (output / "main.log").write_bytes(b"")
    (output / "stdout.log").write_bytes(b"")
    command = [sys.executable, "-c", "print('ok')"]
    request = RunRequest(
        run_id="candi-owt-steps-2",
        model_id="candi",
        dataset_id="owt",
        step_count=2,
        seed=42,
        sample_count=2,
        command=command,
    )

    result = run_experiment(request, tmp_path, adapter=FakeAdapter(command, make_records(2)))

    metadata = json.loads((result.run_dir / "run_metadata.json").read_text())
    assert metadata["checkpoint_lock_id"].startswith("recipe:candi_owt:")
    assert metadata["checkpoint_sha256"]


def test_runner_reruns_when_declared_adapter_identity_changes(tmp_path: Path) -> None:
    """Catch cache reuse after an adapter changes declared conversion semantics."""

    prepare_canonical_root(tmp_path)
    command = [sys.executable, "-c", "print('ok')"]
    request = make_request(command)
    first_adapter = FakeAdapter(command, make_records(2), identity="adapter-v1")
    second_adapter = FakeAdapter(command, make_records(2), identity="adapter-v2")

    assert run_experiment(request, tmp_path, adapter=first_adapter).status == "succeeded"
    assert run_experiment(request, tmp_path, adapter=second_adapter).status == "succeeded"
    assert second_adapter.calls == 1


def test_success_and_skip_clear_stale_failure_marker(tmp_path: Path) -> None:
    """Catch failure evidence surviving a later validated success or exact skip."""

    prepare_canonical_root(tmp_path)
    marker = tmp_path / "ran-once"
    command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; import sys; p=Path({str(marker)!r}); "
        "(p.write_text('done') if not p.exists() else None); sys.exit(7 if p.read_text() == 'done' else 0)",
    ]
    request = make_request(command)
    failed = run_experiment(request, tmp_path, adapter=FakeAdapter(command, make_records(2)))
    assert failed.status == "failed"
    assert (failed.run_dir / "failure.json").exists()
    marker.write_text("succeeded")

    succeeded = run_experiment(request, tmp_path, adapter=FakeAdapter(command, make_records(2)))
    skipped = run_experiment(request, tmp_path, adapter=FakeAdapter(command, make_records(2)))

    assert succeeded.status == "succeeded"
    assert skipped.status == "skipped"
    assert not (succeeded.run_dir / "failure.json").exists()


def test_explicit_provenance_assertion_cannot_override_source_lock(tmp_path: Path) -> None:
    """Catch a caller spoofing source provenance instead of matching the canonical lock."""

    prepare_canonical_root(tmp_path)
    command = [sys.executable, "-c", "print('ok')"]
    request = RunRequest(**{**make_request(command).__dict__, "source_sha256": "x" * 40})

    try:
        run_experiment(request, tmp_path, adapter=FakeAdapter(command, make_records(2)))
    except ValueError as error:
        assert "source SHA" in str(error)
    else:
        raise AssertionError("mismatched source assertion was accepted")


def test_cli_reemits_child_sigterm_after_writing_failure(tmp_path: Path) -> None:
    """Catch CLI conversion of a child SIGTERM into an unrelated numeric exit status."""

    prepare_canonical_root(tmp_path)
    source_root = Path(__file__).parents[1] / "src"
    child = """
import os
import signal
import sys
from dlb import runner

class Adapter:
    identity = 'signal-adapter'
    def build_command(self, request, run_dir):
        return [sys.executable, '-c', 'import os, signal; os.kill(os.getpid(), signal.SIGTERM)']
    def convert_outputs(self, request, run_dir):
        return []

runner.load_adapter = lambda adapter_id: Adapter()
raise SystemExit(runner.main([
    '--root', sys.argv[1], '--model', 'flm', '--dataset', 'lm1b', '--steps', '2',
    '--num-samples', '2', '--seed', '42',
]))
"""
    environment = {**os.environ, "PYTHONPATH": str(source_root)}
    completed = subprocess.run([sys.executable, "-c", child, str(tmp_path)], env=environment)

    assert completed.returncode == -signal.SIGTERM
    run_dir = tmp_path / "results" / "samples" / "lm1b" / "flm" / "steps_2"
    assert json.loads((run_dir / "failure.json").read_text())["signal"] == signal.SIGTERM


def test_run_one_rejects_registry_invalid_step_before_fake_conda() -> None:
    """Catch shell validation that launches Conda or loses the registry error status."""

    project_root = Path(__file__).parents[1]
    environment = {
        **os.environ,
        "DLB_PYTHON": sys.executable,
        "DLB_CONDA": "/usr/bin/false",
    }
    completed = subprocess.run(
        [
            "bash",
            str(project_root / "scripts" / "run_one.sh"),
            "--model",
            "flm",
            "--dataset",
            "lm1b",
            "--steps",
            "3",
            "--num-samples",
            "1",
            "--seed",
            "42",
        ],
        env=environment,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "invalid step count 3" in completed.stderr


@pytest.mark.parametrize("signum", [signal.SIGKILL, signal.SIGSTOP])
def test_cli_does_not_reset_uncatchable_signal_handlers(monkeypatch, tmp_path: Path, signum: int) -> None:
    """Catch attempts to reset SIGKILL/SIGSTOP before re-emitting a child signal."""

    reset_calls: list[int] = []
    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(runner_module, "run_experiment", lambda request, root: runner_module.RunResult("failed", tmp_path, -signum))
    monkeypatch.setattr(runner_module.os, "name", "posix")
    monkeypatch.setattr(signal, "signal", lambda observed, handler: reset_calls.append(observed))
    monkeypatch.setattr(runner_module.os, "kill", lambda process, observed: kill_calls.append((process, observed)))

    assert runner_module.main(["--root", str(tmp_path), "--model", "flm", "--dataset", "lm1b", "--steps", "2", "--seed", "42"]) == 128 + signum
    assert reset_calls == []
    assert kill_calls == [(os.getpid(), signum)]


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal semantics")
def test_cli_reemits_child_sigkill_without_handler_reset(tmp_path: Path) -> None:
    """Catch SIGKILL propagation failing before the parent can self-send SIGKILL."""

    prepare_canonical_root(tmp_path)
    source_root = Path(__file__).parents[1] / "src"
    child = """
import os
import signal
import sys
from dlb import runner

class Adapter:
    identity = 'sigkill-adapter'
    def build_command(self, request, run_dir):
        return [sys.executable, '-c', 'import os, signal; os.kill(os.getpid(), signal.SIGKILL)']
    def convert_outputs(self, request, run_dir):
        return []

runner.load_adapter = lambda adapter_id: Adapter()
raise SystemExit(runner.main([
    '--root', sys.argv[1], '--model', 'flm', '--dataset', 'lm1b', '--steps', '2',
    '--num-samples', '2', '--seed', '42',
]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", child, str(tmp_path)],
        env={**os.environ, "PYTHONPATH": str(source_root)},
    )

    assert completed.returncode == -signal.SIGKILL
    run_dir = tmp_path / "results" / "samples" / "lm1b" / "flm" / "steps_2"
    assert json.loads((run_dir / "failure.json").read_text())["signal"] == signal.SIGKILL

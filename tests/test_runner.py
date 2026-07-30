import json
import hashlib
import sys
from pathlib import Path

from dlb.runner import RunRequest, run_experiment


class FakeAdapter:
    identity = "tests.fake_adapter:v1"

    def __init__(self, command: list[str], records: list[dict[str, object]]) -> None:
        self.command = command
        self.records = records
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
        config_sha256="c" * 64,
        source_sha256="s" * 40,
        checkpoint_sha256="k" * 64,
        checkpoint_lock_id="flm_lm1b_hf",
        adapter_identity="tests.fake_adapter:v1",
        environment="dlb-flm",
    )


def test_runner_records_command_failure(tmp_path: Path) -> None:
    """Catch child failures that lose the exit code or diagnostic logs."""

    command = [sys.executable, "-c", "import sys; print('out'); print('bad', file=sys.stderr); sys.exit(7)"]
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
    assert metadata["source_sha256"] == "s" * 40
    assert metadata["checkpoint_lock_id"] == "flm_lm1b_hf"


def test_runner_reruns_when_checkpoint_identity_changes(tmp_path: Path) -> None:
    """Catch stale reuse after the checkpoint lock resolves to different content."""

    command = [sys.executable, "-c", "print('ok')"]
    adapter = FakeAdapter(command, make_records(2))
    first_request = make_request(command)
    changed_request = RunRequest(**{**first_request.__dict__, "checkpoint_sha256": "z" * 64})

    assert run_experiment(first_request, tmp_path, adapter=adapter).status == "succeeded"
    assert run_experiment(changed_request, tmp_path, adapter=adapter).status == "succeeded"
    assert adapter.calls == 2


def test_runner_does_not_publish_metadata_after_conversion_failure(tmp_path: Path) -> None:
    """Catch a success marker published before converted samples satisfy the contract."""

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

    command = [sys.executable, "-c", "print('ok')"]
    first_request = make_request(command)
    assert run_experiment(
        first_request, tmp_path, adapter=FakeAdapter(command, make_records(2))
    ).status == "succeeded"
    changed_request = RunRequest(**{**first_request.__dict__, "source_sha256": "x" * 40})

    result = run_experiment(
        changed_request,
        tmp_path,
        adapter=FakeAdapter(command, [{**make_records(1)[0], "text": ""}]),
    )

    assert result.status == "failed"
    assert not (result.run_dir / "run_metadata.json").exists()


def test_runner_refuses_symlinked_log_target(tmp_path: Path) -> None:
    """Catch a prepared log symlink that would truncate a file outside the run."""

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
        json.dumps({"sources": {"flm": {"commit": "a" * 40}}})
    )
    files = [{"path": "checkpoints/official/flm/lm1b/model.safetensors", "size_bytes": 3, "sha256": "b" * 64}]
    (tmp_path / "artifacts" / "checkpoint_lock.json").write_text(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(checkpoints_manifest.read_bytes()).hexdigest(),
                "resources": {"flm_lm1b_hf": {"status": "downloaded", "files": files}},
            }
        )
    )
    command = [sys.executable, "-c", "print('ok')"]
    request = RunRequest(
        run_id="flm-lm1b-steps-2",
        model_id="flm",
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
                "selector": {"resource": "flm_lm1b_hf", "path": None, "teacher_family": None},
                "files": files,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert metadata["checkpoint_sha256"] == expected
    assert metadata["checkpoint_lock_id"] == "flm_lm1b_hf:" + manifest_sha + ":all"


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

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dlb.gpu_matrix import active_gpus
from dlb.matrix import MatrixTask, write_matrix


ROOT = Path(__file__).parents[1]


def _task(index: int) -> MatrixTask:
    steps = 2**index
    return MatrixTask(
        task_id=f"flm-lm1b-steps-{steps}",
        category="many",
        model="flm",
        dataset="lm1b",
        steps=steps,
        sample_count=1024,
        seed=42,
        environment="dlb-flm",
        adapter="flm",
        source="flm",
        provenance="official",
        sample_dir=f"/tmp/dlb/samples/lm1b/flm/steps_{steps}",
        metrics_path=f"/tmp/dlb/metrics/lm1b/flm/steps_{steps}/metrics.json",
        timing_path=f"/tmp/dlb/timing/lm1b/flm/steps_{steps}/timing.json",
    )


def test_gpu_matrix_dry_run_binds_tasks_to_requested_gpus(tmp_path):
    matrix = write_matrix(tmp_path / "generation.tsv", [_task(0), _task(1), _task(2)])

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "dlb.gpu_matrix",
            "--root",
            str(ROOT),
            "--stage",
            "generate",
            "--matrix",
            str(matrix),
            "--gpus",
            "0,1",
            "--max-jobs",
            "2",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    records = [json.loads(line) for line in completed.stdout.splitlines()]
    assert [record["gpu"] for record in records] == ["0", "1", "0"]
    assert all(record["cuda_visible_devices"] == record["gpu"] for record in records)
    assert all(record["stage"] == "generate" for record in records)
    assert records[0]["command"][:2] == ["bash", str(ROOT / "scripts/run_one.sh")]
    assert "--num-samples" in records[0]["command"]


def test_gpu_matrix_rejects_empty_gpu_list(tmp_path):
    matrix = write_matrix(tmp_path / "generation.tsv", [_task(0)])

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "dlb.gpu_matrix",
            "--root",
            str(ROOT),
            "--stage",
            "benchmark",
            "--matrix",
            str(matrix),
            "--gpus",
            "",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 2
    assert "at least one GPU" in completed.stderr


def test_gpu_matrix_can_oversubscribe_gpus_when_max_jobs_exceeds_gpu_count():
    assert active_gpus(("0", "1"), 5) == ("0", "1", "0", "1", "0")


@pytest.mark.parametrize(
    "script",
    [
        "scripts/smoke_4gpu.sh",
        "scripts/run_4gpu.sh",
        "scripts/evaluate_4gpu.sh",
        "scripts/benchmark_4gpu.sh",
    ],
)
def test_4gpu_wrappers_do_not_regenerate_explicit_custom_matrix(
    tmp_path: Path, script: str
):
    custom = write_matrix(tmp_path / "custom.tsv", [_task(0)])
    before = custom.read_text(encoding="utf-8")

    completed = subprocess.run(
        [
            "bash",
            str(ROOT / script),
            "--root",
            str(ROOT),
            "--matrix",
            str(custom),
            "--gpus",
            "0,1",
            "--max-jobs",
            "1",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert custom.read_text(encoding="utf-8") == before
    records = [json.loads(line) for line in completed.stdout.splitlines()]
    assert [record["task_id"] for record in records] == ["flm-lm1b-steps-1"]

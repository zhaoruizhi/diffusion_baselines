import csv
import json
from io import StringIO
from pathlib import Path

import pytest

from dlb.matrix import MatrixTask
from dlb.timing_summary import main, summarize_timing, write_timing_csv


PROJECT_ROOT = Path(__file__).parents[1]


def prepare_registry(root: Path) -> None:
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs/experiments.yaml").write_text(
        (PROJECT_ROOT / "configs/experiments.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def fake_task(root: Path, steps: int) -> MatrixTask:
    timing = root / f"results/timing/lm1b/fmlm/steps_{steps}/timing.json"
    return MatrixTask(
        task_id=f"fmlm-lm1b-steps-{steps}",
        category="few",
        model="fmlm",
        dataset="lm1b",
        steps=steps,
        sample_count=1024,
        seed=42,
        environment="dlb-flm",
        adapter="flm",
        source="flm",
        provenance="official",
        sample_dir=str(root / f"results/samples/lm1b/fmlm/steps_{steps}"),
        metrics_path=str(root / f"results/metrics/lm1b/fmlm/steps_{steps}/metrics.json"),
        timing_path=str(timing),
    )


def write_timing(path: Path, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "dlb-generation-timing-v1",
                "timing": {
                    "mode": "primary_latency",
                    "warmups": 5,
                    "repeats": 32,
                    "batch_size": 1,
                    "num_timed_samples": 32,
                    "seconds_per_sample": seconds,
                },
            }
        ),
        encoding="utf-8",
    )


def test_summarize_timing_reports_present_rows_and_missing_steps(tmp_path: Path) -> None:
    tasks = [fake_task(tmp_path, step) for step in (1, 2, 4)]
    write_timing(Path(tasks[0].timing_path), 0.125)
    write_timing(Path(tasks[2].timing_path), 0.5)

    report = summarize_timing(tasks)

    assert [row.as_csv_row() for row in report.rows] == [
        {
            "dataset": "lm1b",
            "model": "fmlm",
            "steps": "1",
            "seconds_per_sample": "0.125000",
        },
        {
            "dataset": "lm1b",
            "model": "fmlm",
            "steps": "4",
            "seconds_per_sample": "0.500000",
        },
    ]
    assert [(item.dataset, item.model, item.steps) for item in report.missing] == [
        ("lm1b", "fmlm", 2),
    ]


def test_write_timing_csv_uses_requested_column_order(tmp_path: Path) -> None:
    tasks = [fake_task(tmp_path, 2)]
    write_timing(Path(tasks[0].timing_path), 0.25)
    report = summarize_timing(tasks)
    output = StringIO()

    write_timing_csv(report.rows, output)

    parsed = list(csv.DictReader(StringIO(output.getvalue())))
    assert output.getvalue().splitlines()[0] == "dataset,model,steps,seconds_per_sample"
    assert parsed == [
        {
            "dataset": "lm1b",
            "model": "fmlm",
            "steps": "2",
            "seconds_per_sample": "0.250000",
        }
    ]


@pytest.mark.parametrize("seconds", [None, -0.1, "0.1"])
def test_summarize_timing_rejects_invalid_seconds_per_sample(
    tmp_path: Path, seconds: object
) -> None:
    task = fake_task(tmp_path, 1)
    path = Path(task.timing_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "dlb-generation-timing-v1",
                "timing": {
                    "mode": "primary_latency",
                    "warmups": 5,
                    "repeats": 32,
                    "batch_size": 1,
                    "num_timed_samples": 32,
                    "seconds_per_sample": seconds,
                },
            }
        ),
        encoding="utf-8",
    )

    report = summarize_timing([task])

    assert report.rows == ()
    assert len(report.invalid) == 1
    assert report.invalid[0].steps == 1


def test_cli_filters_matrix_and_reports_missing_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prepare_registry(tmp_path)
    write_timing(
        tmp_path / "results/timing/lm1b/fmlm/steps_2/timing.json",
        0.25,
    )

    assert (
        main(
            [
                "--root",
                str(tmp_path),
                "--model",
                "fmlm",
                "--dataset",
                "lm1b",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        "dataset,model,steps,seconds_per_sample",
        "lm1b,fmlm,2,0.250000",
    ]
    assert "expected=6 present=1 missing=5 invalid=0" in captured.err
    assert "missing,lm1b,fmlm,1," in captured.err

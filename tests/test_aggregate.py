import json
from pathlib import Path

import pytest

import dlb.aggregate as aggregate_module
from dlb.aggregate import IncompleteMatrixError, aggregate
from dlb.io import write_samples_atomic
from dlb.matrix import MatrixTask

PROJECT_ROOT = Path(__file__).parents[1]


def prepare_registry(root: Path) -> None:
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs/experiments.yaml").write_text(
        (PROJECT_ROOT / "configs/experiments.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def fake_task(root: Path, sample_count: int = 2) -> MatrixTask:
    sample_dir = root / "results/samples/lm1b/flm/steps_2"
    metrics = root / "results/metrics/lm1b/flm/steps_2/metrics.json"
    timing = root / "results/timing/lm1b/flm/steps_2/timing.json"
    return MatrixTask(
        task_id="flm-lm1b-steps-2",
        category="many",
        model="flm",
        dataset="lm1b",
        steps=2,
        sample_count=sample_count,
        seed=42,
        environment="dlb-flm",
        adapter="flm",
        source="flm",
        provenance="official",
        sample_dir=str(sample_dir),
        metrics_path=str(metrics),
        timing_path=str(timing),
    )


def write_complete_fake_run(root: Path, task: MatrixTask) -> None:
    sample_dir = Path(task.sample_dir)
    write_samples_atomic(
        sample_dir / "samples.jsonl",
        [
            {
                "sample_id": index,
                "text": f"sample {index}",
                "token_ids": [101, index + 1, 102],
                "seed": task.seed,
                "generation_seconds": 0.01,
            }
            for index in range(task.sample_count)
        ],
        expected=task.sample_count,
    )
    provenance = {
        "source_sha256": "a" * 40,
        "config_sha256": "b" * 64,
        "checkpoint_sha256": "c" * 64,
        "checkpoint_lock_id": "lock:fake",
        "checkpoint_selection": {"resource": "fake"},
        "checkpoint_teacher_family": "continuous_flm",
        "adapter_identity": "tests.fake:v1",
        "environment": task.environment,
    }
    identity = {
        "model_id": task.model,
        "dataset_id": task.dataset,
        "step_count": task.steps,
        "seed": task.seed,
        "sample_count": task.sample_count,
    }
    metadata = {
        "status": "succeeded",
        "identity": {**identity, **provenance},
        **provenance,
    }
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "run_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    metrics_path = Path(task.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sample_count": task.sample_count,
                "partial": False,
                "metrics": {
                    "generative_perplexity": {"perplexity": 12.0},
                    "unigram_entropy": {"mean_entropy": 2.0},
                    "self_bleu": {"score": 0.4},
                },
            }
        ),
        encoding="utf-8",
    )
    timing_path = Path(task.timing_path)
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(
        json.dumps(
            {
                "schema": "dlb-generation-timing-v1",
                "timing": {
                    "mode": "primary_latency",
                    "warmups": 5,
                    "repeats": 32,
                    "batch_size": 1,
                    "num_timed_samples": 32,
                    "seconds_per_sample": 0.25,
                },
                "metadata": {
                    "attempt_id": "d" * 32,
                    "identity": {**identity, **provenance},
                    **provenance,
                },
            }
        ),
        encoding="utf-8",
    )


def patch_single_task(monkeypatch, task: MatrixTask):
    monkeypatch.setattr(
        aggregate_module,
        "build_matrix",
        lambda registry, **kwargs: [task],
    )


def test_aggregate_requires_all_metrics(monkeypatch, tmp_path):
    prepare_registry(tmp_path)
    task = fake_task(tmp_path)
    write_complete_fake_run(tmp_path, task)
    metrics = json.loads(Path(task.metrics_path).read_text(encoding="utf-8"))
    del metrics["metrics"]["self_bleu"]
    Path(task.metrics_path).write_text(json.dumps(metrics), encoding="utf-8")
    patch_single_task(monkeypatch, task)
    with pytest.raises(IncompleteMatrixError):
        aggregate(tmp_path, strict=True)


def test_partial_aggregate_publishes_failures_and_provenance(monkeypatch, tmp_path):
    prepare_registry(tmp_path)
    task = fake_task(tmp_path)
    write_complete_fake_run(tmp_path, task)
    patch_single_task(monkeypatch, task)
    report = aggregate(
        tmp_path,
        strict=False,
        partial=True,
        output_dir=tmp_path / "results/summary",
    )
    assert report.complete is True
    assert len(report.rows) == 1
    assert set(
        row["metric"]
        for row in __import__("csv").DictReader(
            (tmp_path / "results/summary/results_long.csv").open()
        )
    ) == {
        "generative_perplexity",
        "unigram_entropy",
        "self_bleu",
        "generation_seconds_per_sample",
    }
    assert "checkpoint_sha256" in (
        tmp_path / "results/summary/results_wide.csv"
    ).read_text(encoding="utf-8").splitlines()[0]

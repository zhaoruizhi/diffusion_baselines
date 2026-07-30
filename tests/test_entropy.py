import math
import json
from pathlib import Path
import subprocess
import sys

import pytest

from evaluation.unigram_entropy import mean_unigram_entropy, unigram_entropy


ROOT = Path(__file__).parents[1]


def record(tokens: list[int]) -> dict[str, object]:
    return {"token_ids": tokens}


def test_entropy_in_nats() -> None:
    """Catch bit-based entropy or a non-empirical token distribution."""

    assert unigram_entropy([1, 1, 2, 2]) == pytest.approx(math.log(2))
    assert unigram_entropy([7, 7, 7]) == 0.0


def test_mean_entropy_is_per_sample_and_removes_only_documented_padding() -> None:
    """Catch corpus-level entropy or accidental removal of generated BOS/EOS."""

    result = mean_unigram_entropy(
        [record([0, 101, 5, 5, 102, 0]), record([101, 7, 102, 0])],
        special_ids={0},
    )

    first = -(2 / 4) * math.log(2 / 4) - 2 * (1 / 4) * math.log(1 / 4)
    second = math.log(3)
    assert result.mean_entropy == pytest.approx((first + second) / 2)
    assert result.sample_count == 2
    assert result.token_count == 7
    assert result.excluded_token_ids == (0,)
    assert result.unit == "nats"


def test_mean_entropy_rejects_a_row_containing_only_excluded_ids() -> None:
    """Catch silently treating an empty generated sequence as zero entropy."""

    with pytest.raises(ValueError, match="record 0.*no tokens"):
        mean_unigram_entropy([record([0, 0])], special_ids={0})


def test_metric_cli_requires_explicit_partial_override(tmp_path: Path) -> None:
    """Catch a smoke result being published as a complete 1,024-sample baseline."""

    output = tmp_path / "metrics.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.evaluate",
            "--samples",
            str(ROOT / "tests" / "fixtures" / "sample_texts.jsonl"),
            "--metrics",
            "entropy,self_bleu",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "expected 1024 records" in result.stderr
    assert not output.exists()


def test_metric_cli_atomically_labels_partial_fixture_result(tmp_path: Path) -> None:
    """Catch partial output that lacks metric conventions or input provenance."""

    output = tmp_path / "metrics.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.evaluate",
            "--samples",
            str(ROOT / "tests" / "fixtures" / "sample_texts.jsonl"),
            "--metrics",
            "entropy,self_bleu",
            "--output",
            str(output),
            "--allow-partial",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["sample_count"] == 4
    assert document["partial"] is True
    assert document["production_sample_count"] == 1024
    assert len(document["samples_sha256"]) == 64
    assert document["metrics"]["unigram_entropy"]["unit"] == "nats"
    assert document["metrics"]["self_bleu"]["reference_rule"] == "all_other_samples"
    assert not list(tmp_path.glob(".metrics.json.*.partial"))

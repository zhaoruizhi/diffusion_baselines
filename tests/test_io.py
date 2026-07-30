import json
from decimal import Decimal
from pathlib import Path

import pytest

from dlb.io import SampleCountError, SampleValidationError, validate_samples, write_samples_atomic


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


def test_validate_requires_exact_count(tmp_path: Path) -> None:
    """Catch a completed-looking file with the wrong requested sample count."""

    path = tmp_path / "samples.jsonl"
    write_samples_atomic(path, make_records(3))

    with pytest.raises(SampleCountError, match="expected 1024 records, found 3"):
        validate_samples(path, expected=1024)


def test_writer_keeps_existing_final_when_new_records_are_invalid(tmp_path: Path) -> None:
    """Catch publication that replaces a valid final file before semantic validation."""

    path = tmp_path / "samples.jsonl"
    write_samples_atomic(path, make_records(1))

    with pytest.raises(SampleValidationError, match="record 0") as error:
        write_samples_atomic(
            path,
            [
                {
                    "sample_id": 0,
                    "text": "   ",
                    "token_ids": [1],
                    "seed": 42,
                    "generation_seconds": 0.1,
                }
            ],
        )

    assert "text" in str(error.value)

    assert [json.loads(line) for line in path.read_text().splitlines()] == make_records(1)
    assert not list(tmp_path.glob(".samples.jsonl.*.partial"))


def test_validator_reports_record_index_for_noncontiguous_ids(tmp_path: Path) -> None:
    """Catch duplicate or missing sample identifiers hidden in otherwise valid JSON."""

    path = tmp_path / "samples.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in make_records(2)) + "\n")
    rows = path.read_text().splitlines()
    second = json.loads(rows[1])
    second["sample_id"] = 3
    path.write_text(rows[0] + "\n" + json.dumps(second) + "\n")

    with pytest.raises(SampleValidationError, match="record 1.*expected sample_id 1, found 3"):
        validate_samples(path, expected=2)


def test_writer_rejects_symlinked_final_target(tmp_path: Path) -> None:
    """Catch a symlink attack that would redirect an atomic replace outside results."""

    target = tmp_path / "target.jsonl"
    target.write_text("outside\n")
    path = tmp_path / "samples.jsonl"
    path.symlink_to(target)

    with pytest.raises(SampleValidationError, match="symlink"):
        write_samples_atomic(path, make_records(1))

    assert target.read_text() == "outside\n"


def test_writer_counts_one_record_from_a_generator(tmp_path: Path) -> None:
    """Catch iterable publication that validates a one-row generator as empty."""

    path = tmp_path / "samples.jsonl"
    count = write_samples_atomic(path, (record for record in make_records(1)))

    assert count == 1
    assert validate_samples(path, expected=1) == 1


@pytest.mark.parametrize("timing", [True, 1, Decimal("0.1"), "0.1", float("nan"), float("inf")])
def test_writer_rejects_non_float_or_nonfinite_generation_time(tmp_path: Path, timing: object) -> None:
    """Catch coercive timing values before they enter a published sample artifact."""

    record = make_records(1)[0]
    record["generation_seconds"] = timing

    with pytest.raises(SampleValidationError, match="generation_seconds"):
        write_samples_atomic(tmp_path / "samples.jsonl", [record])


def test_validator_rejects_integer_generation_time_from_json(tmp_path: Path) -> None:
    """Catch a JSON numeric integer accepted as a timing float during streamed validation."""

    record = make_records(1)[0]
    record["generation_seconds"] = 1
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps(record) + "\n")

    with pytest.raises(SampleValidationError, match="generation_seconds"):
        validate_samples(path, expected=1)

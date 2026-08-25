import json
from pathlib import Path

import pytest

from dlb.io import (
    SampleValidationError,
    expected_conditional_schedule,
    read_conditional_samples,
    validate_conditional_samples,
    write_conditional_samples_atomic,
)


def conditional_records(
    schedule: list[tuple[int, int]], *, sequence_length: int = 128
) -> list[dict[str, object]]:
    """Return independently constructed records for a supplied publication schedule."""

    records: list[dict[str, object]] = []
    for sample_id, (prompt_id, completion_id) in enumerate(schedule):
        prefix = [sample_id % 10] * 64
        continuation = [(sample_id + 1) % 10] * (sequence_length - 64)
        records.append(
            {
                "sample_id": sample_id,
                "prompt_id": prompt_id,
                "completion_id": completion_id,
                "source_index": sample_id + 10,
                "prefix_token_ids": prefix,
                "continuation_token_ids": continuation,
                "reference_token_ids": [9] * 64,
                "full_token_ids": prefix + continuation,
                "prefix_text": "fixed prefix",
                "continuation_text": "generated continuation",
                "reference_text": "source continuation",
                "full_text": "fixed prefix generated continuation",
                "seed": 42,
                "generation_seconds": 0.25,
                "prefix_exact_match": True,
            }
        )
    return records


def test_schedule_lists_quality_before_diversity_completions() -> None:
    """Catch a schedule that changes the public prompt/completion ordering."""

    assert expected_conditional_schedule(3, 2, 3) == [
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 1),
        (1, 1),
        (0, 2),
        (1, 2),
    ]


def test_writer_and_reader_round_trip_a_parameterized_lm1b_artifact(tmp_path: Path) -> None:
    """Catch conditional I/O that loses validated records or small smoke schedules."""

    schedule = expected_conditional_schedule(2, 1, 2)
    path = tmp_path / "conditional.jsonl"

    assert write_conditional_samples_atomic(
        path,
        conditional_records(schedule),
        expected=3,
        schedule=schedule,
        sequence_length=128,
        vocab_size=10,
    ) == 3
    assert validate_conditional_samples(
        path,
        expected=3,
        schedule=schedule,
        sequence_length=128,
        vocab_size=10,
    ) == 3
    assert [(record.prompt_id, record.completion_id) for record in read_conditional_samples(path)] == schedule


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda records: records[0].update(
                continuation_token_ids=[1] * 63,
                full_token_ids=records[0]["prefix_token_ids"] + [1] * 63,
            ),
            "sequence_length",
        ),
        (
            lambda records: records[0].update(
                continuation_token_ids=[10] + [1] * 63,
                full_token_ids=records[0]["prefix_token_ids"] + [10] + [1] * 63,
            ),
            "vocabulary",
        ),
    ],
)
def test_validator_rejects_wrong_model_length_or_vocabulary_token(
    tmp_path: Path, mutation, message: str
) -> None:
    """Catch samples that cannot be evaluated on the declared dataset model contract."""

    schedule = [(0, 0)]
    records = conditional_records(schedule)
    mutation(records)
    path = tmp_path / "conditional.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    with pytest.raises(SampleValidationError, match=message):
        validate_conditional_samples(
            path,
            expected=1,
            schedule=schedule,
            sequence_length=128,
            vocab_size=10,
        )


def test_reader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    """Catch ambiguous JSON that could hide a changed conditional field."""

    path = tmp_path / "conditional.jsonl"
    record = conditional_records([(0, 0)])[0]
    payload = json.dumps(record, separators=(",", ":")).replace(
        '"prompt_id":0,', '"prompt_id":0,"prompt_id":1,', 1
    )
    path.write_text(payload + "\n", encoding="utf-8")

    with pytest.raises(SampleValidationError, match="duplicate JSON key"):
        list(read_conditional_samples(path))


def test_writer_rejects_duplicate_schedule_entry_without_replacing_final(tmp_path: Path) -> None:
    """Catch publication that replaces a finished shard before schedule validation."""

    final = tmp_path / "samples.jsonl"
    final.write_text("old\n", encoding="utf-8")
    records = conditional_records(expected_conditional_schedule())
    records[1024]["prompt_id"] = 1

    with pytest.raises(SampleValidationError, match="expected prompt/completion"):
        write_conditional_samples_atomic(
            final,
            records,
            expected=2048,
            sequence_length=128,
            vocab_size=30_522,
        )

    assert final.read_text(encoding="utf-8") == "old\n"


def test_validator_requires_all_schedule_records_when_expected_is_none(tmp_path: Path) -> None:
    """Catch a partial conditional shard accepted when its caller omits expected."""

    path = tmp_path / "conditional.jsonl"
    path.write_text(json.dumps(conditional_records([(0, 0)])[0]) + "\n", encoding="utf-8")

    with pytest.raises(SampleValidationError, match="expected 2 records, found 1"):
        validate_conditional_samples(
            path,
            expected=None,
            schedule=[(0, 0), (1, 0)],
            sequence_length=128,
            vocab_size=10,
        )


@pytest.mark.parametrize("operation", ["validate", "write"])
def test_conditional_publication_requires_explicit_dataset_contract(
    tmp_path: Path, operation: str
) -> None:
    """Catch accidental LM1B defaults used to validate an unspecified dataset artifact."""

    path = tmp_path / "conditional.jsonl"
    records = conditional_records([(0, 0)])
    path.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")

    with pytest.raises(TypeError):
        if operation == "validate":
            validate_conditional_samples(path, expected=1, schedule=[(0, 0)])
        else:
            write_conditional_samples_atomic(path, records, expected=1, schedule=[(0, 0)])


@pytest.mark.parametrize(
    "schedule",
    [
        [(0, 0), (0, 0)],
        [(1024, 0)],
        [(0, 5)],
        [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)],
    ],
)
def test_writer_rejects_noncanonical_custom_schedule(tmp_path: Path, schedule) -> None:
    """Catch duplicate, out-of-range, or incomplete prompt/completion schedule definitions."""

    with pytest.raises(ValueError, match="conditional schedule"):
        write_conditional_samples_atomic(
            tmp_path / "conditional.jsonl",
            [],
            expected=None,
            schedule=schedule,
            sequence_length=128,
            vocab_size=10,
        )

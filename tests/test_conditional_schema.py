import pytest
from pydantic import ValidationError

from dlb.schema import ConditionalSampleRecord


def valid_conditional_record() -> dict[str, object]:
    """Return a hand-authored LM1B-sized conditional publication record."""

    prefix = [7] * 64
    continuation = [8] * 64
    return {
        "sample_id": 0,
        "prompt_id": 0,
        "completion_id": 0,
        "source_index": 12,
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


def test_conditional_record_binds_prefix_full_and_suffix_slices() -> None:
    """Catch publication records whose retained prefix or suffix is inconsistent."""

    record = ConditionalSampleRecord.model_validate(valid_conditional_record())

    assert record.full_token_ids[:64] == record.prefix_token_ids
    assert record.full_token_ids[64:] == record.continuation_token_ids
    assert record.prefix_exact_match is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(prefix_exact_match=False), "prefix_exact_match"),
        (lambda value: value["full_token_ids"].__setitem__(0, 999), "prefix"),
        (lambda value: value.update(reference_token_ids=[1] * 63), "64"),
    ],
)
def test_conditional_record_rejects_invalid_publication(mutation, message: str) -> None:
    """Catch loss of the hard-prefix or aligned-reference publication invariants."""

    value = valid_conditional_record()
    mutation(value)

    with pytest.raises(ValidationError, match=message):
        ConditionalSampleRecord.model_validate(value)

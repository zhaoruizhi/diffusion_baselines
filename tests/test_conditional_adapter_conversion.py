import json
from pathlib import Path

import pytest

from dlb.adapters.base import AdapterError
from dlb.adapters.mdlm import MDLMAdapter


def _capture_record(index: int) -> dict[str, object]:
    prefix = [1] * 64
    continuation = [2] * 64
    return {
        "sample_id": index,
        "text": "fixed prefix generated continuation",
        "token_ids": prefix + continuation,
        "prompt_id": index,
        "completion_id": 0,
        "source_index": index + 100,
        "prefix_token_ids": prefix,
        "reference_token_ids": [3] * 64,
        "full_token_ids": prefix + continuation,
        "prefix_text": "fixed prefix",
        "continuation_text": "generated continuation",
        "reference_text": "source continuation",
        "full_text": "fixed prefix generated continuation",
        "prefix_exact_match": True,
    }


def _write_capture(path: Path, samples: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps({"schema": "dlb-upstream-token-capture-v1", "samples": samples}),
        encoding="utf-8",
    )


def test_conditional_capture_reader_accepts_complete_prompt_records(tmp_path: Path) -> None:
    """Catch converter regressions that drop prompt/source metadata from captures."""

    path = tmp_path / "upstream_token_ids.json"
    _write_capture(path, [_capture_record(0), _capture_record(1)])

    records = MDLMAdapter()._read_conditional_capture(path, 2)

    assert [record["prompt_id"] for record in records] == [0, 1]
    assert records[0]["prefix_exact_match"] is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda record: record.update(prefix_exact_match=False), "prefix mismatch"),
        (lambda record: record.update(extra="bad"), "record format"),
        (lambda record: record.update(sample_id=1), "expected sample_id 0"),
    ],
)
def test_conditional_capture_reader_rejects_invalid_prompt_records(
    tmp_path: Path, mutate, message: str
) -> None:
    """Catch malformed conditional captures before publication JSONL is written."""

    path = tmp_path / "upstream_token_ids.json"
    record = _capture_record(0)
    mutate(record)
    _write_capture(path, [record])

    with pytest.raises(AdapterError, match=message):
        MDLMAdapter()._read_conditional_capture(path, 1)

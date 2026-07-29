import pytest

from dlb.io import atomic_json_write, sha256_file
from dlb.schema import SampleRecord


def test_sample_record_rejects_empty_text():
    with pytest.raises(ValueError):
        SampleRecord(
            sample_id=0,
            text="",
            token_ids=[1],
            seed=42,
            generation_seconds=0.1,
        )


def test_sample_record_rejects_negative_generation_time():
    with pytest.raises(ValueError):
        SampleRecord(
            sample_id=0,
            text="generated text",
            token_ids=[1],
            seed=42,
            generation_seconds=-0.1,
        )


def test_atomic_json_write_replaces_file_with_deterministic_json(tmp_path):
    target = tmp_path / "nested" / "record.json"

    atomic_json_write(target, {"z": 1, "a": [2]})

    assert target.read_text() == '{\n  "a": [\n    2\n  ],\n  "z": 1\n}\n'


def test_sha256_file_streams_known_content_digest(tmp_path):
    target = tmp_path / "asset.bin"
    target.write_bytes(b"abc")

    assert sha256_file(target) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import sys
import threading
import types
from urllib.error import HTTPError

import pytest
import yaml

import dlb.checkpoints as checkpoints_module
from dlb.checkpoints import (
    CheckpointResource,
    DigestSpec,
    DirectSource,
    GDriveSource,
    HuggingFaceSource,
    ZenodoSource,
    build_hf_snapshot_kwargs,
    download_direct,
    fetch_resource,
    publish_partial,
    require_server_platform,
    safe_checkpoint_destination,
    safe_remote_path,
    select_zenodo_files,
    verify_checkpoint_lock,
    verify_published_file,
)


def write_single_resource_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "artifacts" / "checkpoints.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "resources": {
                    "test_direct": {
                        "id": "test_direct",
                        "backend": "direct",
                        "provenance": "official",
                        "teacher_family": "masked_mdlm",
                        "destination": "official/test",
                        "license": "test",
                        "terms_url": "https://example.test/terms",
                        "digest": {"policy": "capture_after_download"},
                        "required_files": ["model.bin"],
                        "source": {
                            "url": "https://example.test/model.bin",
                            "filename": "model.bin",
                        },
                    }
                },
                "recipes": {},
                "coverage": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return manifest


def manifest_bound_lock(manifest: Path, file_path: str, payload: bytes) -> dict[str, object]:
    return {
        "schema_version": 1,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "resources": {
            "test_direct": {
                "status": "downloaded",
                "backend": "direct",
                "provenance": "official",
                "teacher_family": "masked_mdlm",
                "destination": "checkpoints/official/test",
                "files": [
                    {
                        "path": file_path,
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
        },
    }


@contextmanager
def ranged_http_server(payload: bytes):
    requests: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            range_header = self.headers.get("Range")
            requests.append(range_header)
            start = int(range_header.removeprefix("bytes=").removesuffix("-")) if range_header else 0
            body = payload[start:]
            self.send_response(206 if range_header else 200)
            self.send_header("Content-Length", str(len(body)))
            if range_header:
                self.send_header("Content-Range", f"bytes {start}-{len(payload)-1}/{len(payload)}")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/checkpoint.bin", requests
    finally:
        server.shutdown()
        thread.join()


class FakeHTTPResponse(io.BytesIO):
    def __init__(self, payload: bytes, *, status: int, headers: dict[str, str]):
        super().__init__(payload)
        self.status = status
        self.headers = headers

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def install_fake_http(monkeypatch, responses):
    requests = []
    queue = list(responses)

    def fake_urlopen(request):
        requests.append({key.lower(): value for key, value in request.header_items()})
        result = queue.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(checkpoints_module, "urlopen", fake_urlopen)
    return requests


def write_partial_resume_state(
    target: Path, payload: bytes, validator: str, total_size: int
) -> None:
    partial = target.with_name(target.name + ".partial")
    partial.write_bytes(payload)
    partial.with_name(partial.name + ".meta.json").write_text(
        json.dumps(
            {
                "url": "https://example.test/model.bin",
                "validator_header": "ETag",
                "validator_value": validator,
                "total_size": total_size,
            }
        ),
        encoding="utf-8",
    )


def test_direct_download_discards_unvalidated_partial_and_atomically_publishes(
    tmp_path, monkeypatch
):
    """Catch partial bytes without entity metadata being trusted for resume."""

    payload = b"checkpoint-payload-for-resume"
    digest = hashlib.sha256(payload).hexdigest()
    target = tmp_path / "model.bin"
    partial = target.with_name(target.name + ".partial")
    partial.write_bytes(payload[:11])
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")

    with ranged_http_server(payload) as (url, requests):
        result = download_direct(
            DirectSource(url=url, filename="model.bin"),
            target,
            DigestSpec(policy="sha256", sha256=digest),
        )

    assert requests == [None]
    assert target.read_bytes() == payload
    assert not partial.exists()
    assert result["status"] == "downloaded"
    assert result["sha256"] == digest


def test_direct_resume_sends_if_range_and_accepts_exact_validated_206(
    tmp_path, monkeypatch
):
    """Catch a valid partial resume that is not bound to its original HTTP entity."""

    payload = b"abcdefghij"
    target = tmp_path / "model.bin"
    write_partial_resume_state(target, payload[:4], '"v1"', len(payload))
    requests = install_fake_http(
        monkeypatch,
        [
            FakeHTTPResponse(
                payload[4:],
                status=206,
                headers={
                    "Content-Range": "bytes 4-9/10",
                    "Content-Length": "6",
                    "ETag": '"v1"',
                },
            )
        ],
    )

    download_direct(
        DirectSource(url="https://example.test/model.bin", filename="model.bin"),
        target,
        DigestSpec(policy="sha256", sha256=hashlib.sha256(payload).hexdigest()),
    )

    assert requests == [{"range": "bytes=4-", "if-range": '"v1"'}]
    assert target.read_bytes() == payload
    assert not target.with_name("model.bin.partial.meta.json").exists()


def test_direct_resume_restarts_when_content_range_starts_at_wrong_offset(
    tmp_path, monkeypatch
):
    """Catch appending a 206 response that does not begin at the partial size."""

    payload = b"abcdefghij"
    target = tmp_path / "model.bin"
    write_partial_resume_state(target, payload[:4], '"v1"', len(payload))
    requests = install_fake_http(
        monkeypatch,
        [
            FakeHTTPResponse(
                payload[3:],
                status=206,
                headers={"Content-Range": "bytes 3-9/10", "ETag": '"v1"'},
            ),
            FakeHTTPResponse(
                payload,
                status=200,
                headers={"Content-Length": "10", "ETag": '"v1"'},
            ),
        ],
    )

    download_direct(
        DirectSource(url="https://example.test/model.bin", filename="model.bin"),
        target,
        DigestSpec(policy="sha256", sha256=hashlib.sha256(payload).hexdigest()),
    )

    assert requests[0] == {"range": "bytes=4-", "if-range": '"v1"'}
    assert requests[1] == {}
    assert target.read_bytes() == payload


def test_direct_resume_uses_full_200_fallback_without_appending(tmp_path, monkeypatch):
    """Catch a server's full-response fallback being appended to old partial bytes."""

    payload = b"abcdefghij"
    target = tmp_path / "model.bin"
    write_partial_resume_state(target, b"old!", '"v1"', len(payload))
    requests = install_fake_http(
        monkeypatch,
        [
            FakeHTTPResponse(
                payload,
                status=200,
                headers={"Content-Length": "10", "ETag": '"v2"'},
            )
        ],
    )

    download_direct(
        DirectSource(url="https://example.test/model.bin", filename="model.bin"),
        target,
        DigestSpec(policy="sha256", sha256=hashlib.sha256(payload).hexdigest()),
    )

    assert requests == [{"range": "bytes=4-", "if-range": '"v1"'}]
    assert target.read_bytes() == payload
    assert not target.with_name("model.bin.partial.meta.json").exists()


def test_direct_resume_restarts_when_entity_validator_changes(tmp_path, monkeypatch):
    """Catch combining old partial bytes with a different ETag's ranged response."""

    old_payload = b"abcdefghij"
    new_payload = b"1234567890"
    target = tmp_path / "model.bin"
    write_partial_resume_state(target, old_payload[:4], '"v1"', len(old_payload))
    requests = install_fake_http(
        monkeypatch,
        [
            FakeHTTPResponse(
                new_payload[4:],
                status=206,
                headers={"Content-Range": "bytes 4-9/10", "ETag": '"v2"'},
            ),
            FakeHTTPResponse(
                new_payload,
                status=200,
                headers={"Content-Length": "10", "ETag": '"v2"'},
            ),
        ],
    )

    download_direct(
        DirectSource(url="https://example.test/model.bin", filename="model.bin"),
        target,
        DigestSpec(policy="sha256", sha256=hashlib.sha256(new_payload).hexdigest()),
    )

    assert requests[0] == {"range": "bytes=4-", "if-range": '"v1"'}
    assert requests[1] == {}
    assert target.read_bytes() == new_payload


def test_complete_partial_revalidates_unchanged_entity_without_redownload(
    tmp_path, monkeypatch
):
    """Catch complete partials issuing an invalid bytes=<total>- request."""

    payload = b"abcdefghij"
    target = tmp_path / "model.bin"
    write_partial_resume_state(target, payload, '"v1"', len(payload))
    requests = install_fake_http(
        monkeypatch,
        [
            FakeHTTPResponse(
                payload[:1],
                status=206,
                headers={
                    "Content-Range": "bytes 0-0/10",
                    "Content-Length": "1",
                    "ETag": '"v1"',
                },
            )
        ],
    )

    download_direct(
        DirectSource(url="https://example.test/model.bin", filename="model.bin"),
        target,
        DigestSpec(policy="sha256", sha256=hashlib.sha256(payload).hexdigest()),
    )

    assert requests == [{"range": "bytes=0-0", "if-range": '"v1"'}]
    assert target.read_bytes() == payload


def test_complete_partial_overwrites_when_entity_changed(tmp_path, monkeypatch):
    """Catch a complete old entity being published after validator mismatch."""

    old_payload = b"abcdefghij"
    new_payload = b"1234567890"
    target = tmp_path / "model.bin"
    write_partial_resume_state(target, old_payload, '"v1"', len(old_payload))
    requests = install_fake_http(
        monkeypatch,
        [
            FakeHTTPResponse(
                new_payload,
                status=200,
                headers={"Content-Length": "10", "ETag": '"v2"'},
            )
        ],
    )

    download_direct(
        DirectSource(url="https://example.test/model.bin", filename="model.bin"),
        target,
        DigestSpec(policy="sha256", sha256=hashlib.sha256(new_payload).hexdigest()),
    )

    assert requests == [{"range": "bytes=0-0", "if-range": '"v1"'}]
    assert target.read_bytes() == new_payload


def test_complete_partial_handles_416_once_then_restarts_without_range(
    tmp_path, monkeypatch
):
    """Catch identical Range retry loops after an HTTP 416 response."""

    payload = b"abcdefghij"
    target = tmp_path / "model.bin"
    write_partial_resume_state(target, payload, '"v1"', len(payload))
    error = HTTPError(
        "https://example.test/model.bin", 416, "Range Not Satisfiable", {}, None
    )
    requests = install_fake_http(
        monkeypatch,
        [
            error,
            FakeHTTPResponse(
                payload,
                status=200,
                headers={"Content-Length": "10", "ETag": '"v1"'},
            ),
        ],
    )

    download_direct(
        DirectSource(url="https://example.test/model.bin", filename="model.bin"),
        target,
        DigestSpec(policy="sha256", sha256=hashlib.sha256(payload).hexdigest()),
    )

    assert requests == [
        {"range": "bytes=0-0", "if-range": '"v1"'},
        {},
    ]
    assert target.read_bytes() == payload


def test_direct_download_rejects_a_symlinked_partial_before_writing(tmp_path, monkeypatch):
    """Catch resume writes following a sibling .partial symlink outside staging."""

    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"sentinel")
    target = tmp_path / "checkpoints" / "model.bin"
    target.parent.mkdir()
    target.with_name("model.bin.partial").symlink_to(outside)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")

    with ranged_http_server(b"replacement") as (url, requests):
        with pytest.raises(ValueError, match="symlink"):
            download_direct(
                DirectSource(url=url, filename="model.bin"),
                target,
                DigestSpec(policy="capture_after_download"),
            )

    assert requests == []
    assert outside.read_bytes() == b"sentinel"


def test_huggingface_fetch_rejects_a_symlinked_staging_directory(tmp_path, monkeypatch):
    """Catch snapshot_download following an existing snapshot.partial symlink."""

    checkpoint_root = tmp_path / "checkpoints"
    outside = tmp_path / "outside"
    outside.mkdir()
    partial = checkpoint_root / "official" / "model.partial"
    partial.parent.mkdir(parents=True)
    partial.symlink_to(outside, target_is_directory=True)
    called = False

    def fake_snapshot_download(**kwargs):
        nonlocal called
        called = True

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_snapshot_download),
    )
    resource = CheckpointResource.model_validate(
        {
            "id": "test_hf",
            "backend": "huggingface",
            "provenance": "official",
            "teacher_family": "masked_mdlm",
            "destination": "official/model",
            "license": "test",
            "terms_url": "https://example.test/terms",
            "digest": {"policy": "capture_after_download"},
            "required_files": ["model.safetensors"],
            "source": {
                "repo_id": "owner/model",
                "revision": "a" * 40,
                "allow_patterns": ["*.safetensors"],
            },
        }
    )

    with pytest.raises(ValueError, match="symlink"):
        fetch_resource(tmp_path, resource)

    assert called is False
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"README.md": b"metadata", "config.json": b"{}"},
        {"config.json": b"{}", "model.safetensors": b""},
    ],
    ids=["empty", "metadata-only", "empty-primary"],
)
def test_huggingface_fetch_requires_primary_weight_before_publication(
    tmp_path, monkeypatch, payload
):
    """Catch empty or README/config-only snapshots being locked as models."""

    def fake_snapshot_download(**kwargs):
        staging = Path(kwargs["local_dir"])
        for name, content in payload.items():
            (staging / name).write_bytes(content)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_snapshot_download),
    )
    resource = CheckpointResource.model_validate(
        {
            "id": "test_hf",
            "backend": "huggingface",
            "provenance": "official",
            "teacher_family": "masked_mdlm",
            "destination": "official/model",
            "license": "test",
            "terms_url": "https://example.test/terms",
            "digest": {"policy": "capture_after_download"},
            "required_files": ["config.json", "model.safetensors"],
            "source": {
                "repo_id": "owner/model",
                "revision": "a" * 40,
                "allow_patterns": ["README.md", "config.json", "model.safetensors"],
            },
        }
    )

    with pytest.raises(FileNotFoundError, match="required checkpoint file"):
        fetch_resource(tmp_path, resource)

    assert not (tmp_path / "checkpoints" / "official" / "model").exists()


def test_direct_fetch_rejects_an_empty_required_file(tmp_path, monkeypatch):
    """Catch capture policy locking an empty direct-download payload as successful."""

    requests = install_fake_http(
        monkeypatch,
        [FakeHTTPResponse(b"", status=200, headers={"Content-Length": "0", "ETag": '"v1"'})],
    )
    resource = CheckpointResource.model_validate(
        {
            "id": "test_direct",
            "backend": "direct",
            "provenance": "official",
            "teacher_family": "masked_mdlm",
            "destination": "official/direct",
            "license": "test",
            "terms_url": "https://example.test/terms",
            "digest": {"policy": "capture_after_download"},
            "required_files": ["model.bin"],
            "source": {
                "url": "https://example.test/model.bin",
                "filename": "model.bin",
            },
        }
    )

    with pytest.raises(FileNotFoundError, match="empty"):
        fetch_resource(tmp_path, resource)

    assert requests == [{}]


def test_huggingface_fetch_enforces_per_file_manifest_digests(tmp_path, monkeypatch):
    """Catch HF publication bypassing the resource's declared digest policy."""

    def fake_snapshot_download(**kwargs):
        staging = Path(kwargs["local_dir"])
        (staging / "config.json").write_bytes(b"config")
        (staging / "model.safetensors").write_bytes(b"weights")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_snapshot_download),
    )
    resource = CheckpointResource.model_validate(
        {
            "id": "test_hf",
            "backend": "huggingface",
            "provenance": "official",
            "teacher_family": "masked_mdlm",
            "destination": "official/model",
            "license": "test",
            "terms_url": "https://example.test/terms",
            "digest": {
                "policy": "sha256",
                "per_file_sha256": {
                    "config.json": hashlib.sha256(b"config").hexdigest(),
                    "model.safetensors": "0" * 64,
                },
            },
            "required_files": ["config.json", "model.safetensors"],
            "source": {
                "repo_id": "owner/model",
                "revision": "a" * 40,
                "allow_patterns": ["config.json", "model.safetensors"],
            },
        }
    )

    with pytest.raises(ValueError, match="download digest mismatch"):
        fetch_resource(tmp_path, resource)

    assert not (tmp_path / "checkpoints" / "official" / "model").exists()


def test_mismatch_is_quarantined_before_replacement(tmp_path):
    """Catch corrupt existing checkpoints being overwritten without audit evidence."""

    target = tmp_path / "official" / "model.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"corrupt")
    partial = target.with_name(target.name + ".partial")
    partial.write_bytes(b"correct")
    expected = hashlib.sha256(b"correct").hexdigest()

    result = publish_partial(
        partial,
        target,
        DigestSpec(policy="sha256", sha256=expected),
        quarantine_root=tmp_path / "quarantine",
        timestamp="20260729T120000Z",
    )

    assert target.read_bytes() == b"correct"
    quarantined = tmp_path / "quarantine" / "20260729T120000Z" / "official" / "model.bin"
    assert quarantined.read_bytes() == b"corrupt"
    assert result["quarantined"] == quarantined.as_posix()


@pytest.mark.parametrize(
    "name",
    ["../escape.ckpt", "/absolute.ckpt", "nested/../../escape", "", ".", "a\\b"],
)
def test_remote_filenames_cannot_escape_destination(name):
    """Catch provider-controlled filenames traversing outside the checkpoint root."""

    with pytest.raises(ValueError, match="unsafe remote path"):
        safe_remote_path(name)


def test_checkpoint_destination_rejects_a_symlinked_ancestor(tmp_path):
    """Catch downloads escaping through a pre-existing symlink under checkpoints."""

    checkpoint_root = tmp_path / "root" / "checkpoints"
    outside = tmp_path / "outside"
    checkpoint_root.mkdir(parents=True)
    outside.mkdir()
    (checkpoint_root / "official").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        safe_checkpoint_destination(checkpoint_root, "official/model/checkpoint.bin")


def test_huggingface_backend_pins_revision_and_allow_patterns(tmp_path):
    """Catch broad or moving Hugging Face snapshot requests."""

    source = HuggingFaceSource(
        repo_id="owner/model",
        revision="a" * 40,
        allow_patterns=["*.json", "*.safetensors"],
    )

    assert build_hf_snapshot_kwargs(source, tmp_path / "snapshot.partial") == {
        "repo_id": "owner/model",
        "revision": "a" * 40,
        "allow_patterns": ["*.json", "*.safetensors"],
        "local_dir": str(tmp_path / "snapshot.partial"),
        "resume_download": True,
    }


def test_gdrive_backend_addresses_each_expected_file_by_immutable_id(tmp_path):
    """Catch mutable folder-name discovery replacing manifest-pinned file IDs."""

    source = GDriveSource(
        folder_id="folder123",
        expected_files={"LM1B/checkpoint.pth": "fileA", "LM1B/config.yaml": "fileB"},
    )
    staging = tmp_path / "rdlm.partial"

    assert checkpoints_module.build_gdrive_commands(source, staging) == [
        [
            "gdown",
            "--continue",
            "--id",
            "fileA",
            "--output",
            str(staging / ".objects" / "fileA.partial"),
        ],
        [
            "gdown",
            "--continue",
            "--id",
            "fileB",
            "--output",
            str(staging / ".objects" / "fileB.partial"),
        ],
    ]


def test_gdrive_fetch_replaces_same_name_staging_from_pinned_file_ids(tmp_path, monkeypatch):
    """Catch a stale same-name file passing without acquisition by its declared ID."""

    resource = CheckpointResource.model_validate(
        {
            "id": "test_drive",
            "backend": "gdrive",
            "provenance": "official",
            "teacher_family": "continuous_rdlm",
            "destination": "official/rdlm",
            "license": "test",
            "terms_url": "https://example.test/terms",
            "digest": {"policy": "capture_after_download"},
            "required_files": ["LM1B/checkpoint.pth", "LM1B/config.yaml"],
            "source": {
                "folder_id": "folder123",
                "expected_files": {
                    "LM1B/checkpoint.pth": "fileA",
                    "LM1B/config.yaml": "fileB",
                },
            },
        }
    )
    staging = tmp_path / "checkpoints" / "official" / "rdlm.partial"
    stale = staging / "LM1B" / "checkpoint.pth"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"wrong-same-name")
    calls = []

    def fake_run(command, check):
        assert check is True
        calls.append(command)
        file_id = command[command.index("--id") + 1]
        output = Path(command[command.index("--output") + 1])
        assert output.parent.is_dir()
        output.write_bytes({"fileA": b"weights", "fileB": b"config"}[file_id])
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(checkpoints_module.subprocess, "run", fake_run)

    record = fetch_resource(tmp_path, resource)

    destination = tmp_path / "checkpoints" / "official" / "rdlm" / "LM1B"
    assert (destination / "checkpoint.pth").read_bytes() == b"weights"
    assert (destination / "config.yaml").read_bytes() == b"config"
    assert [command[command.index("--id") + 1] for command in calls] == ["fileA", "fileB"]
    assert record["status"] == "downloaded"


def test_gdrive_fetch_enforces_per_file_manifest_digests(tmp_path, monkeypatch):
    """Catch Drive ID downloads bypassing the resource's declared digest policy."""

    resource = CheckpointResource.model_validate(
        {
            "id": "test_drive",
            "backend": "gdrive",
            "provenance": "official",
            "teacher_family": "continuous_rdlm",
            "destination": "official/rdlm",
            "license": "test",
            "terms_url": "https://example.test/terms",
            "digest": {
                "policy": "sha256",
                "per_file_sha256": {
                    "checkpoint.pth": "0" * 64,
                    "config.yaml": hashlib.sha256(b"config").hexdigest(),
                },
            },
            "required_files": ["checkpoint.pth", "config.yaml"],
            "source": {
                "folder_id": "folder123",
                "expected_files": {"checkpoint.pth": "fileA", "config.yaml": "fileB"},
            },
        }
    )

    def fake_run(command, check):
        file_id = command[command.index("--id") + 1]
        output = Path(command[command.index("--output") + 1])
        output.write_bytes({"fileA": b"weights", "fileB": b"config"}[file_id])
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(checkpoints_module.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="download digest mismatch"):
        fetch_resource(tmp_path, resource)

    assert not (tmp_path / "checkpoints" / "official" / "rdlm").exists()


def test_gdrive_stale_partial_is_quarantined_and_rerun_succeeds(
    tmp_path, monkeypatch
):
    """Catch unexpected staging files causing every Drive rerun to fail forever."""

    resource = CheckpointResource.model_validate(
        {
            "id": "test_drive",
            "backend": "gdrive",
            "provenance": "official",
            "teacher_family": "continuous_rdlm",
            "destination": "official/rdlm",
            "license": "test",
            "terms_url": "https://example.test/terms",
            "digest": {"policy": "capture_after_download"},
            "required_files": ["checkpoint.pth", "config.yaml"],
            "source": {
                "folder_id": "folder123",
                "expected_files": {"checkpoint.pth": "fileA", "config.yaml": "fileB"},
            },
        }
    )
    staging = tmp_path / "checkpoints" / "official" / "rdlm.partial"
    object_path = staging / ".objects" / "fileA.partial"
    object_path.parent.mkdir(parents=True)
    object_path.write_bytes(b"resume-prefix")
    (staging / "unexpected.bin").write_bytes(b"stale")

    def interrupted_run(command, check):
        raise checkpoints_module.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(checkpoints_module.subprocess, "run", interrupted_run)
    with pytest.raises(checkpoints_module.subprocess.CalledProcessError):
        fetch_resource(tmp_path, resource)

    assert object_path.read_bytes() == b"resume-prefix"
    assert not (staging / "unexpected.bin").exists()
    quarantined = list(
        (tmp_path / "checkpoints" / "quarantine").glob(
            "*/official/rdlm.partial/unexpected.bin"
        )
    )
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"stale"

    def successful_run(command, check):
        file_id = command[command.index("--id") + 1]
        output = Path(command[command.index("--output") + 1])
        output.write_bytes({"fileA": b"weights", "fileB": b"config"}[file_id])
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(checkpoints_module.subprocess, "run", successful_run)
    record = fetch_resource(tmp_path, resource)

    assert record["status"] == "downloaded"
    destination = tmp_path / "checkpoints" / "official" / "rdlm"
    assert (destination / "checkpoint.pth").read_bytes() == b"weights"
    assert (destination / "config.yaml").read_bytes() == b"config"


def test_real_downloads_positively_require_linux():
    """Catch accidental acquisition on any unapproved operating system."""

    for system_name in ("Darwin", "Windows", "FreeBSD", ""):
        with pytest.raises(RuntimeError, match="Linux server-only"):
            require_server_platform(system_name)
    require_server_platform("Linux")


def test_zenodo_record_selects_named_files_and_rejects_traversal():
    """Catch downloading unrelated Zenodo files or trusting unsafe record filenames."""

    metadata = {
        "files": [
            {
                "key": "sdtt7-di4c2.ckpt",
                "size": 17,
                "checksum": "md5:b55b961b1f0d7f2ed55539452875ec15",
                "links": {"content": "https://zenodo.example/sdtt7"},
            },
            {
                "key": "other.bin",
                "size": 3,
                "checksum": "md5:abc",
                "links": {"content": "https://zenodo.example/other"},
            },
        ]
    }
    source = ZenodoSource(record_id=15124163, files=["sdtt7-di4c2.ckpt"])

    assert select_zenodo_files(metadata, source) == [
        {
            "filename": "sdtt7-di4c2.ckpt",
            "url": "https://zenodo.example/sdtt7",
            "size_bytes": 17,
            "published_checksum": "md5:b55b961b1f0d7f2ed55539452875ec15",
        }
    ]

    metadata["files"][0]["key"] = "../escape.ckpt"
    with pytest.raises(ValueError, match="unsafe remote path"):
        select_zenodo_files(metadata, ZenodoSource(record_id=15124163, files=["../escape.ckpt"]))


def test_zenodo_published_size_and_checksum_are_enforced(tmp_path):
    """Catch capturing a new SHA-256 for content that differs from Zenodo metadata."""

    target = tmp_path / "checkpoint.bin"
    target.write_bytes(b"zenodo")
    verify_published_file(
        target,
        size_bytes=6,
        checksum="md5:aa4386efabed44fe87a07a77480593a2",
    )
    with pytest.raises(ValueError, match="published size mismatch"):
        verify_published_file(target, size_bytes=7, checksum=None)
    with pytest.raises(ValueError, match="published checksum mismatch"):
        verify_published_file(target, size_bytes=6, checksum="md5:" + "0" * 32)


def test_zenodo_fetch_preserves_the_manifest_digest_policy(tmp_path, monkeypatch):
    """Catch Zenodo replacing a declared SHA-256 policy with capture-after-download."""

    payload = b"zenodo"
    expected_sha = hashlib.sha256(payload).hexdigest()
    metadata = {
        "files": [
            {
                "key": "model.ckpt",
                "size": len(payload),
                "checksum": "md5:" + hashlib.md5(payload, usedforsecurity=False).hexdigest(),
                "links": {"content": "https://zenodo.example/model"},
            }
        ]
    }

    class MetadataResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(
        checkpoints_module,
        "urlopen",
        lambda request: MetadataResponse(json.dumps(metadata).encode("utf-8")),
    )
    seen = []

    def fake_download(source, target, digest, **kwargs):
        seen.append(digest)
        target.write_bytes(payload)
        return {"status": "downloaded", "quarantined": None}

    monkeypatch.setattr(checkpoints_module, "download_direct", fake_download)
    resource = CheckpointResource.model_validate(
        {
            "id": "test_zenodo",
            "backend": "zenodo",
            "provenance": "official",
            "teacher_family": "masked_mdlm",
            "destination": "official/zenodo",
            "license": "test",
            "terms_url": "https://example.test/terms",
            "digest": {"policy": "sha256", "sha256": expected_sha},
            "required_files": ["model.ckpt"],
            "source": {"record_id": 1, "files": ["model.ckpt"]},
        }
    )

    fetch_resource(tmp_path, resource)

    assert len(seen) == 1
    assert seen[0].policy == "sha256"
    assert seen[0].sha256 == expected_sha


def test_lock_verification_reports_missing_mismatch_and_verified(tmp_path):
    """Catch verification treating absent or changed files as successful."""

    present = tmp_path / "checkpoints" / "official" / "ok.bin"
    changed = tmp_path / "checkpoints" / "official" / "changed.bin"
    present.parent.mkdir(parents=True)
    present.write_bytes(b"ok")
    changed.write_bytes(b"changed")
    lock = {
        "schema_version": 1,
        "resources": {
            "ok": {
                "status": "downloaded",
                "files": [
                    {
                        "path": "checkpoints/official/ok.bin",
                        "size_bytes": 2,
                        "sha256": hashlib.sha256(b"ok").hexdigest(),
                    }
                ],
            },
            "changed": {
                "status": "downloaded",
                "files": [
                    {
                        "path": "checkpoints/official/changed.bin",
                        "size_bytes": 7,
                        "sha256": "0" * 64,
                    }
                ],
            },
            "missing": {
                "status": "unavailable",
                "files": [],
                "error": {"type": "FileNotFoundError", "message": "not published"},
            },
        },
    }

    report = verify_checkpoint_lock(tmp_path, lock)

    assert report["resources"]["ok"]["status"] == "verified"
    assert report["resources"]["changed"]["status"] == "mismatch"
    assert report["resources"]["missing"]["status"] == "unavailable"
    assert report["ok"] is False


def test_lock_verification_rejects_path_traversal(tmp_path):
    """Catch a tampered lock reading files outside the project checkpoint tree."""

    outside = tmp_path.parent / "outside.bin"
    outside.write_bytes(b"outside")
    lock = {
        "schema_version": 1,
        "resources": {
            "tampered": {
                "status": "downloaded",
                "files": [
                    {
                        "path": "../outside.bin",
                        "size_bytes": 7,
                        "sha256": hashlib.sha256(b"outside").hexdigest(),
                    }
                ],
            }
        },
    }

    report = verify_checkpoint_lock(tmp_path, lock)

    assert report["resources"]["tampered"]["status"] == "invalid"
    assert report["ok"] is False


def test_lock_verification_rejects_manifest_drift(tmp_path):
    """Catch verification against a lock produced from a different checkpoint manifest."""

    manifest = tmp_path / "artifacts" / "checkpoints.yaml"
    manifest.parent.mkdir()
    manifest.write_text("schema_version: 1\n", encoding="utf-8")
    lock = {"schema_version": 1, "manifest_sha256": "0" * 64, "resources": {}}

    report = verify_checkpoint_lock(tmp_path, lock, manifest_path=manifest)

    assert report["manifest_status"] == "mismatch"
    assert report["ok"] is False


def test_lock_verification_rejects_a_downloaded_resource_without_files(tmp_path):
    """Catch an empty provider response being recorded as a successful download."""

    lock = {
        "schema_version": 1,
        "resources": {"empty": {"status": "downloaded", "files": []}},
    }

    report = verify_checkpoint_lock(tmp_path, lock)

    assert report["resources"]["empty"]["status"] == "invalid"
    assert report["ok"] is False


@pytest.mark.parametrize("malformed", [[], "not-a-record", None])
def test_lock_verification_structures_nonmapping_resource_records(
    tmp_path, malformed
):
    """Catch malformed lock values reaching dict.get and raising AttributeError."""

    report = verify_checkpoint_lock(
        tmp_path,
        {"schema_version": 1, "resources": {"broken_resource": malformed}},
    )

    record = report["resources"]["broken_resource"]
    assert record["status"] == "invalid"
    assert record["error"]["type"] == "InvalidResourceRecord"
    assert "broken_resource" in record["error"]["message"]
    assert report["ok"] is False


def test_lock_verification_rejects_an_empty_resource_set(tmp_path):
    """Catch an empty lock being accepted as complete verification."""

    report = verify_checkpoint_lock(tmp_path, {"schema_version": 1, "resources": {}})

    assert report["resource_set_status"] == "invalid"
    assert report["ok"] is False


def test_lock_verification_cross_checks_manifest_resource_ids(tmp_path):
    """Catch a nonempty lock that silently omits resources from the pinned manifest."""

    manifest = Path(__file__).parents[1] / "artifacts" / "checkpoints.yaml"
    lock = {
        "schema_version": 1,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "resources": {
            "flm_lm1b_hf": {
                "status": "unavailable",
                "files": [],
                "error": {"type": "HTTPError", "message": "test"},
            }
        },
    }

    report = verify_checkpoint_lock(tmp_path, lock, manifest_path=manifest)

    assert report["resource_set_status"] == "mismatch"
    assert report["ok"] is False


@pytest.mark.parametrize("schema_version", [2, True, 1.0, "1", None])
def test_lock_verification_requires_exact_schema_version(tmp_path, schema_version):
    """Catch a future or malformed lock schema being interpreted as schema 1."""

    report = verify_checkpoint_lock(
        tmp_path,
        {
            "schema_version": schema_version,
            "resources": {"x": {"status": "unavailable", "files": []}},
        },
    )

    assert report["schema_status"] == "invalid"
    assert report["ok"] is False


def test_lock_verification_rejects_a_missing_schema_version(tmp_path):
    report = verify_checkpoint_lock(
        tmp_path,
        {"resources": {"x": {"status": "unavailable", "files": []}}},
    )

    assert report["schema_status"] == "invalid"
    assert report["ok"] is False


@pytest.mark.parametrize(
    ("field", "rewritten"),
    [
        ("backend", "huggingface"),
        ("provenance", "reference_reproduction"),
        ("destination", "checkpoints/official/other"),
    ],
)
def test_lock_verification_binds_resource_metadata_to_manifest(tmp_path, field, rewritten):
    """Catch a lock record being relabeled independently of its manifest resource."""

    manifest = write_single_resource_manifest(tmp_path)
    payload = b"weights"
    model = tmp_path / "checkpoints" / "official" / "test" / "model.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(payload)
    lock = manifest_bound_lock(
        manifest, "checkpoints/official/test/model.bin", payload
    )
    lock["resources"]["test_direct"][field] = rewritten

    report = verify_checkpoint_lock(tmp_path, lock, manifest_path=manifest)

    assert report["resources"]["test_direct"]["status"] == "invalid"
    assert report["ok"] is False


def test_lock_verification_rejects_a_file_outside_its_resource_destination(tmp_path):
    """Catch one resource claiming a regular checkpoint owned by another resource."""

    manifest = write_single_resource_manifest(tmp_path)
    payload = b"swapped"
    swapped = tmp_path / "checkpoints" / "official" / "other" / "model.bin"
    swapped.parent.mkdir(parents=True)
    swapped.write_bytes(payload)
    lock = manifest_bound_lock(
        manifest, "checkpoints/official/other/model.bin", payload
    )

    report = verify_checkpoint_lock(tmp_path, lock, manifest_path=manifest)

    assert report["resources"]["test_direct"]["status"] == "invalid"
    assert report["ok"] is False


@pytest.mark.parametrize("symlink_level", ["root", "destination", "file"])
def test_lock_verification_lstats_every_checkpoint_path_component(tmp_path, symlink_level):
    """Catch symlinked roots, destination ancestors, and final files in lock verification."""

    manifest = write_single_resource_manifest(tmp_path)
    payload = b"linked"
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "model.bin"
    outside_file.write_bytes(payload)
    checkpoint_root = tmp_path / "checkpoints"
    if symlink_level == "root":
        nested = outside / "official" / "test" / "model.bin"
        nested.parent.mkdir(parents=True)
        nested.write_bytes(payload)
        checkpoint_root.symlink_to(outside, target_is_directory=True)
        file_path = "checkpoints/official/test/model.bin"
    elif symlink_level == "destination":
        checkpoint_root.mkdir()
        (checkpoint_root / "official").mkdir()
        (checkpoint_root / "official" / "test").symlink_to(
            outside, target_is_directory=True
        )
        file_path = "checkpoints/official/test/model.bin"
    else:
        target = checkpoint_root / "official" / "test" / "model.bin"
        target.parent.mkdir(parents=True)
        target.symlink_to(outside_file)
        file_path = "checkpoints/official/test/model.bin"
    lock = manifest_bound_lock(manifest, file_path, payload)

    report = verify_checkpoint_lock(tmp_path, lock, manifest_path=manifest)

    assert report["resources"]["test_direct"]["status"] == "invalid"
    assert report["ok"] is False

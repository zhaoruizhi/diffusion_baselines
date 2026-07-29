from __future__ import annotations

from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import types

import pytest

from dlb.checkpoints import (
    CheckpointResource,
    DigestSpec,
    DirectSource,
    GDriveSource,
    HuggingFaceSource,
    ZenodoSource,
    build_gdrive_command,
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


def test_direct_download_resumes_partial_and_atomically_publishes(tmp_path, monkeypatch):
    """Catch a downloader that discards partial bytes or publishes incomplete content."""

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

    assert requests == ["bytes=11-"]
    assert target.read_bytes() == payload
    assert not partial.exists()
    assert result["status"] == "downloaded"
    assert result["sha256"] == digest


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


def test_gdrive_folder_backend_uses_resumable_partial_staging(tmp_path):
    """Catch single-file Drive mode or direct publication of incomplete folders."""

    source = GDriveSource(folder_id="folder123")
    destination = tmp_path / "rdlm"

    assert build_gdrive_command(source, destination) == [
        "gdown",
        "--folder",
        "--continue",
        "--output",
        str(destination.with_name("rdlm.partial")),
        "https://drive.google.com/drive/folders/folder123",
    ]


def test_real_downloads_are_refused_on_darwin():
    """Catch accidental checkpoint acquisition on the local Mac."""

    with pytest.raises(RuntimeError, match="server-only"):
        require_server_platform("Darwin")
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

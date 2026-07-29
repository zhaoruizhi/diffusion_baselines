"""Typed checkpoint manifests, safe download helpers, and lock verification."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from typing import Literal
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dlb.io import atomic_json_write, sha256_file


Provenance = Literal["official", "reference_reproduction", "self_trained"]
TeacherFamily = Literal[
    "continuous_flm",
    "continuous_langflow",
    "continuous_rdlm",
    "uniform_duo",
    "masked_mdlm",
    "hybrid_candi",
    "mixed_baselines",
]
TeacherAdapter = Literal["uniform_to_absorbing", "masked_to_absorbing"]


class ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DigestSpec(ManifestModel):
    policy: Literal["sha256", "capture_after_download"]
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_policy(self) -> "DigestSpec":
        if (self.policy == "sha256") != (self.sha256 is not None):
            raise ValueError("sha256 policy requires one digest; capture policy forbids it")
        return self


class HuggingFaceSource(ManifestModel):
    repo_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    allow_patterns: list[str] = Field(min_length=1)


class GDriveSource(ManifestModel):
    folder_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    expected_files: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_expected_files(self) -> "GDriveSource":
        for path, file_id in self.expected_files.items():
            safe_remote_path(path)
            if not file_id or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for character in file_id):
                raise ValueError(f"invalid Google Drive file id for {path}")
        return self


class ZenodoSource(ManifestModel):
    record_id: int = Field(gt=0)
    files: list[str] = Field(min_length=1)
    published_checksums: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_files(self) -> "ZenodoSource":
        for name in self.files:
            safe_remote_path(name)
        if not set(self.published_checksums) <= set(self.files):
            raise ValueError("published checksums must refer to selected Zenodo files")
        return self


class DirectSource(ManifestModel):
    url: str
    filename: str

    @model_validator(mode="after")
    def validate_direct(self) -> "DirectSource":
        if not self.url.startswith(("http://", "https://")):
            raise ValueError("direct URL must use HTTP(S)")
        safe_remote_path(self.filename)
        return self


Source = HuggingFaceSource | GDriveSource | ZenodoSource | DirectSource


class CheckpointResource(ManifestModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    backend: Literal["huggingface", "gdrive", "zenodo", "direct"]
    provenance: Provenance
    teacher_family: TeacherFamily
    destination: str
    license: str = Field(min_length=1)
    terms_url: str
    digest: DigestSpec
    source: Source
    note: str | None = None

    @model_validator(mode="after")
    def validate_resource(self) -> "CheckpointResource":
        expected = {
            "huggingface": HuggingFaceSource,
            "gdrive": GDriveSource,
            "zenodo": ZenodoSource,
            "direct": DirectSource,
        }[self.backend]
        if not isinstance(self.source, expected):
            raise ValueError(f"{self.backend} resource has the wrong source descriptor")
        path = safe_remote_path(self.destination)
        if path.parts[0] != self.provenance:
            raise ValueError("resource destination must begin with its provenance")
        if not self.terms_url.startswith("https://"):
            raise ValueError("resource terms URL must use HTTPS")
        return self


class CoverageEntry(ManifestModel):
    model: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    dataset: Literal["lm1b", "owt"]
    resource: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    path: str | None = None
    teacher_family: TeacherFamily | None = None

    @model_validator(mode="after")
    def validate_path(self) -> "CoverageEntry":
        if self.path is not None:
            safe_remote_path(self.path)
        return self


class TrainingRecipe(ManifestModel):
    model: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    dataset: Literal["lm1b", "owt"]
    provenance: Provenance
    teacher_family: TeacherFamily
    source: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    teacher_adapter: TeacherAdapter | None = None
    command: str = Field(min_length=1)
    output: str
    prerequisites: list[str] = Field(default_factory=list)
    note: str | None = None

    @model_validator(mode="after")
    def validate_output(self) -> "TrainingRecipe":
        path = safe_remote_path(self.output)
        if path.parts[0] != "checkpoints" or path.parts[1] != self.provenance:
            raise ValueError("recipe output must be inside checkpoints/<provenance>")
        if self.source in {"sdtt", "di4c"}:
            expected = {
                "uniform_duo": "uniform_to_absorbing",
                "masked_mdlm": "masked_to_absorbing",
            }.get(self.teacher_family)
            if self.teacher_adapter != expected:
                raise ValueError(
                    f"{self.source} recipe for {self.teacher_family} requires {expected}"
                )
        elif self.teacher_adapter is not None:
            raise ValueError("teacher adapters are only valid for SDTT/Di4C recipes")
        return self


class CheckpointManifest(ManifestModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal[1]
    resources: dict[str, CheckpointResource]
    recipes: dict[str, TrainingRecipe]
    coverage_entries: list[CoverageEntry] = Field(alias="coverage")

    @property
    def coverage(self) -> dict[tuple[str, str], CoverageEntry]:
        return {(item.model, item.dataset): item for item in self.coverage_entries}

    @model_validator(mode="after")
    def validate_links(self) -> "CheckpointManifest":
        for key, resource in self.resources.items():
            if key != resource.id:
                raise ValueError(f"resource key {key} does not match id {resource.id}")
        cells = [(entry.model, entry.dataset) for entry in self.coverage_entries]
        if len(cells) != len(set(cells)):
            raise ValueError("checkpoint coverage cells must be unique")
        for entry in self.coverage_entries:
            if entry.resource not in self.resources:
                raise ValueError(f"unknown checkpoint resource {entry.resource}")
            resource = self.resources[entry.resource]
            if resource.teacher_family == "mixed_baselines":
                if entry.path is None or entry.teacher_family is None:
                    raise ValueError("mixed checkpoint resources require a path and teacher family")
            elif entry.teacher_family not in {None, resource.teacher_family}:
                raise ValueError("coverage teacher family differs from its checkpoint resource")
        return self


def load_checkpoint_manifest(path: Path) -> CheckpointManifest:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("checkpoint manifest must be a mapping")
    return CheckpointManifest.model_validate(document)


def validate_checkpoint_coverage(registry, manifest: CheckpointManifest) -> None:
    """Cross-check registry support cells against resources and recipe fallbacks."""

    referenced_recipes: set[str] = set()
    for model_name, model in registry.models.items():
        for dataset, support in model.datasets.items():
            cell = (model_name, dataset)
            coverage = manifest.coverage.get(cell)
            recipe_id = support.train_recipe
            if support.status == "unsupported":
                if coverage is not None:
                    raise ValueError(f"unsupported cell {model_name}/{dataset} has a checkpoint")
                continue
            if (coverage is None) == (recipe_id is None):
                raise ValueError(
                    f"supported cell {model_name}/{dataset} requires exactly one resource or recipe"
                )
            if coverage is not None:
                resource = manifest.resources[coverage.resource]
                if resource.provenance != support.provenance:
                    raise ValueError(f"checkpoint provenance differs for {model_name}/{dataset}")
            else:
                if recipe_id not in manifest.recipes:
                    raise ValueError(f"unknown recipe {recipe_id} for {model_name}/{dataset}")
                recipe = manifest.recipes[recipe_id]
                if (recipe.model, recipe.dataset) != cell:
                    raise ValueError(f"recipe {recipe_id} does not describe {model_name}/{dataset}")
                if recipe.provenance != support.provenance:
                    raise ValueError(f"recipe provenance differs for {model_name}/{dataset}")
                referenced_recipes.add(recipe_id)
    known_cells = {
        (model_name, dataset)
        for model_name, model in registry.models.items()
        for dataset in model.datasets
    }
    extras = set(manifest.coverage) - known_cells
    if extras:
        raise ValueError(f"checkpoint coverage contains unknown cells: {sorted(extras)}")
    if referenced_recipes != set(manifest.recipes):
        raise ValueError("checkpoint manifest contains unreferenced training recipes")


def safe_remote_path(name: str) -> PurePosixPath:
    """Return a safe relative POSIX path controlled by a remote provider."""

    if not name or name in {".", ".."} or "\\" in name:
        raise ValueError(f"unsafe remote path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe remote path: {name!r}")
    return path


def require_server_platform(system_name: str) -> None:
    """Refuse real checkpoint acquisition on the explicitly out-of-scope Mac."""

    if system_name == "Darwin":
        raise RuntimeError("checkpoint downloads are server-only and refused on Darwin")


def safe_checkpoint_destination(checkpoint_root: Path, relative_name: str) -> Path:
    """Resolve a manifest path while rejecting symlink ancestors and escapes."""

    relative = safe_remote_path(relative_name)
    root = checkpoint_root.absolute()
    if root.is_symlink():
        raise ValueError(f"checkpoint root is a symlink: {root}")
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"checkpoint destination contains a symlink: {current}")
    try:
        candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError as error:
        raise ValueError(f"checkpoint destination escapes root: {candidate}") from error
    return candidate


def verify_published_file(
    path: Path,
    *,
    size_bytes: int | None,
    checksum: str | None,
) -> None:
    """Validate provider-published size/checksum metadata before publication."""

    if size_bytes is not None and path.stat().st_size != size_bytes:
        raise ValueError(f"published size mismatch for {path}")
    if checksum is None:
        return
    try:
        algorithm, expected = checksum.split(":", 1)
    except ValueError as error:
        raise ValueError(f"invalid published checksum for {path}") from error
    if algorithm == "md5":
        digest = hashlib.md5(usedforsecurity=False)
    elif algorithm == "sha256":
        digest = hashlib.sha256()
    else:
        raise ValueError(f"unsupported published checksum algorithm: {algorithm}")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"published checksum mismatch for {path}")


def build_hf_snapshot_kwargs(source: HuggingFaceSource, partial: Path) -> dict[str, object]:
    return {
        "repo_id": source.repo_id,
        "revision": source.revision,
        "allow_patterns": list(source.allow_patterns),
        "local_dir": str(partial),
        "resume_download": True,
    }


def build_gdrive_command(source: GDriveSource, destination: Path) -> list[str]:
    partial = destination.with_name(destination.name + ".partial")
    return [
        "gdown",
        "--folder",
        "--continue",
        "--output",
        str(partial),
        f"https://drive.google.com/drive/folders/{source.folder_id}",
    ]


def _quarantine_path(target: Path, quarantine_root: Path, timestamp: str) -> Path:
    try:
        relative = target.relative_to(quarantine_root.parent)
    except ValueError:
        relative = Path(target.name)
    destination = quarantine_root / timestamp / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def publish_partial(
    partial: Path,
    target: Path,
    digest: DigestSpec,
    *,
    quarantine_root: Path | None = None,
    timestamp: str | None = None,
) -> dict[str, object]:
    """Validate a partial file, quarantine a mismatch, and atomically publish it."""

    if partial.is_symlink():
        raise ValueError(f"checkpoint partial path is a symlink: {partial}")
    if target.is_symlink():
        raise ValueError(f"checkpoint target path is a symlink: {target}")
    observed = sha256_file(partial)
    if digest.sha256 is not None and observed != digest.sha256:
        raise ValueError(f"download digest mismatch for {target}")
    quarantined = None
    if target.exists():
        existing = sha256_file(target)
        if digest.sha256 is not None and existing == digest.sha256:
            partial.unlink()
            return {
                "status": "reused",
                "sha256": existing,
                "size_bytes": target.stat().st_size,
                "quarantined": None,
            }
        quarantine_root = quarantine_root or target.parents[1] / "quarantine"
        timestamp = timestamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        quarantine = _quarantine_path(target, quarantine_root, timestamp)
        os.replace(target, quarantine)
        quarantined = quarantine.as_posix()
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, target)
    return {
        "status": "downloaded",
        "sha256": observed,
        "size_bytes": target.stat().st_size,
        "quarantined": quarantined,
    }


def download_direct(
    source: DirectSource,
    target: Path,
    digest: DigestSpec,
    *,
    quarantine_root: Path | None = None,
    published_size_bytes: int | None = None,
    published_checksum: str | None = None,
) -> dict[str, object]:
    """Resume an HTTP download into ``.partial`` and atomically publish it."""

    if target.is_symlink():
        raise ValueError(f"checkpoint target path is a symlink: {target}")
    if target.exists() and digest.sha256 and sha256_file(target) == digest.sha256:
        return {
            "status": "reused",
            "sha256": digest.sha256,
            "size_bytes": target.stat().st_size,
            "quarantined": None,
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    if partial.is_symlink():
        raise ValueError(f"checkpoint partial path is a symlink: {partial}")
    if partial.exists() and not partial.is_file():
        raise ValueError(f"checkpoint partial path is not a file: {partial}")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = Request(source.url, headers=headers)
    with urlopen(request) as response:
        status = getattr(response, "status", response.getcode())
        mode = "ab" if offset and status == 206 else "wb"
        with partial.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())
    verify_published_file(
        partial,
        size_bytes=published_size_bytes,
        checksum=published_checksum,
    )
    return publish_partial(
        partial,
        target,
        digest,
        quarantine_root=quarantine_root,
    )


def select_zenodo_files(
    metadata: dict[str, object], source: ZenodoSource
) -> list[dict[str, object]]:
    by_name: dict[str, dict[str, object]] = {}
    for item in metadata.get("files", []):
        name = str(item["key"])
        safe_remote_path(name)
        by_name[name] = item
    selected = []
    for name in source.files:
        safe_remote_path(name)
        if name not in by_name:
            raise FileNotFoundError(f"Zenodo record {source.record_id} lacks {name}")
        item = by_name[name]
        links = item["links"]
        selected.append(
            {
                "filename": name,
                "url": links.get("content") or links["self"],
                "size_bytes": int(item["size"]),
                "published_checksum": item.get("checksum"),
            }
        )
    return selected


def _file_records(path: Path, root: Path) -> list[dict[str, object]]:
    paths = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    records = []
    for item in paths:
        if item.is_symlink():
            raise ValueError(f"checkpoint tree contains a symlink: {item}")
        safe_remote_path(item.relative_to(path.parent if path.is_file() else path).as_posix())
        records.append(
            {
                "path": item.relative_to(root).as_posix(),
                "size_bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    return records


def _publish_directory(partial: Path, destination: Path, quarantine_root: Path) -> str | None:
    if partial.is_symlink():
        raise ValueError(f"checkpoint partial path is a symlink: {partial}")
    for item in partial.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"checkpoint tree contains a symlink: {item}")
        safe_remote_path(item.relative_to(partial).as_posix())
    quarantined = None
    if destination.exists():
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        quarantine = _quarantine_path(destination, quarantine_root, stamp)
        os.replace(destination, quarantine)
        quarantined = quarantine.as_posix()
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, destination)
    return quarantined


def fetch_resource(root: Path, resource: CheckpointResource) -> dict[str, object]:
    """Fetch one resource and return a lock-ready status record."""

    checkpoint_root = root / "checkpoints"
    destination = safe_checkpoint_destination(checkpoint_root, resource.destination)
    quarantine_root = safe_checkpoint_destination(checkpoint_root, "quarantine")
    if resource.backend == "direct":
        source = resource.source
        target = safe_checkpoint_destination(
            checkpoint_root, f"{resource.destination}/{source.filename}"
        )
        result = download_direct(source, target, resource.digest, quarantine_root=quarantine_root)
        quarantined = result.get("quarantined")
    elif resource.backend == "huggingface":
        source = resource.source
        partial = safe_checkpoint_destination(
            checkpoint_root, resource.destination + ".partial"
        )
        if partial.exists() and not partial.is_dir():
            raise ValueError(f"HF partial path is not a directory: {partial}")
        partial.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import snapshot_download

        snapshot_download(**build_hf_snapshot_kwargs(source, partial))
        quarantined = _publish_directory(partial, destination, quarantine_root)
    elif resource.backend == "gdrive":
        source = resource.source
        partial = safe_checkpoint_destination(
            checkpoint_root, resource.destination + ".partial"
        )
        subprocess.run(build_gdrive_command(source, destination), check=True)
        if not partial.is_dir():
            raise FileNotFoundError(f"gdown did not create folder staging path {partial}")
        for relative_name in source.expected_files:
            expected = partial.joinpath(*safe_remote_path(relative_name).parts)
            if not expected.is_file() or expected.is_symlink():
                raise FileNotFoundError(f"Google Drive folder lacks expected file {relative_name}")
        quarantined = _publish_directory(partial, destination, quarantine_root)
    elif resource.backend == "zenodo":
        source = resource.source
        api_url = f"https://zenodo.org/api/records/{source.record_id}"
        with urlopen(api_url) as response:
            metadata = json.load(response)
        destination.mkdir(parents=True, exist_ok=True)
        quarantined = None
        for item in select_zenodo_files(metadata, source):
            filename = str(item["filename"])
            published_checksum = item.get("published_checksum")
            pinned_checksum = source.published_checksums.get(filename)
            if pinned_checksum is not None and published_checksum != pinned_checksum:
                raise ValueError(f"Zenodo metadata checksum drift for {filename}")
            result = download_direct(
                DirectSource(url=str(item["url"]), filename=filename),
                safe_checkpoint_destination(
                    checkpoint_root, f"{resource.destination}/{filename}"
                ),
                DigestSpec(policy="capture_after_download"),
                quarantine_root=quarantine_root,
                published_size_bytes=int(item["size_bytes"]),
                published_checksum=pinned_checksum or published_checksum,
            )
            quarantined = quarantined or result.get("quarantined")
    else:  # pragma: no cover - Pydantic makes this unreachable
        raise ValueError(f"unsupported backend {resource.backend}")
    files = _file_records(destination, root)
    if not files:
        raise FileNotFoundError(f"checkpoint resource {resource.id} produced no files")
    return {
        "status": "downloaded",
        "backend": resource.backend,
        "provenance": resource.provenance,
        "teacher_family": resource.teacher_family,
        "destination": destination.relative_to(root).as_posix(),
        "quarantined": quarantined,
        "files": files,
    }


def fetch_all_resources(root: Path, manifest_path: Path, manifest: CheckpointManifest) -> dict[str, object]:
    records: dict[str, object] = {}
    for resource_id, resource in manifest.resources.items():
        try:
            records[resource_id] = fetch_resource(root, resource)
        except Exception as error:  # preserve provider failures as structured lock evidence
            status = "unavailable" if isinstance(error, (FileNotFoundError, HTTPError)) else "error"
            records[resource_id] = {
                "status": status,
                "backend": resource.backend,
                "provenance": resource.provenance,
                "teacher_family": resource.teacher_family,
                "destination": f"checkpoints/{resource.destination}",
                "files": [],
                "error": {"type": type(error).__name__, "message": str(error)},
            }
    lock = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "manifest_sha256": sha256_file(manifest_path),
        "resources": records,
    }
    atomic_json_write(root / "artifacts" / "checkpoint_lock.json", lock)
    return lock


def verify_checkpoint_lock(
    root: Path,
    lock: dict[str, object],
    *,
    manifest_path: Path | None = None,
) -> dict[str, object]:
    report: dict[str, object] = {"ok": True, "resources": {}}
    resources = lock.get("resources")
    if not isinstance(resources, dict) or not resources:
        report["resource_set_status"] = "invalid"
        report["ok"] = False
        resources = resources if isinstance(resources, dict) else {}
    else:
        report["resource_set_status"] = "verified"
    if manifest_path is not None:
        expected_manifest = lock.get("manifest_sha256")
        observed_manifest = sha256_file(manifest_path) if manifest_path.is_file() else None
        report["manifest_status"] = (
            "verified" if observed_manifest == expected_manifest else "mismatch"
        )
        if report["manifest_status"] != "verified":
            report["ok"] = False
        try:
            expected_ids = set(load_checkpoint_manifest(manifest_path).resources)
        except (OSError, ValueError):
            report["resource_set_status"] = "invalid"
            report["ok"] = False
        else:
            if set(resources) != expected_ids:
                report["resource_set_status"] = "mismatch"
                report["ok"] = False
    for resource_id, record in resources.items():
        if record.get("status") not in {"downloaded", "verified"}:
            report["resources"][resource_id] = {
                "status": record.get("status", "unavailable"),
                "error": record.get("error"),
            }
            report["ok"] = False
            continue
        if not record.get("files"):
            report["resources"][resource_id] = {"status": "invalid", "files": []}
            report["ok"] = False
            continue
        status = "verified"
        details = []
        for expected in record.get("files", []):
            try:
                relative = safe_remote_path(str(expected["path"]))
                if relative.parts[0] != "checkpoints":
                    raise ValueError("lock paths must remain inside checkpoints")
                path = root.joinpath(*relative.parts)
                path.resolve().relative_to((root / "checkpoints").resolve())
            except (ValueError, OSError):
                status = "invalid"
                details.append({"path": expected.get("path"), "status": "invalid"})
                break
            if not path.is_file():
                status = "missing"
                details.append({"path": expected["path"], "status": "missing"})
            elif path.stat().st_size != expected["size_bytes"] or sha256_file(path) != expected["sha256"]:
                status = "mismatch"
                details.append({"path": expected["path"], "status": "mismatch"})
            else:
                details.append({"path": expected["path"], "status": "verified"})
        report["resources"][resource_id] = {"status": status, "files": details}
        if status != "verified":
            report["ok"] = False
    return report

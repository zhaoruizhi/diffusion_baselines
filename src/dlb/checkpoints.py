"""Typed checkpoint manifests, safe download helpers, and lock verification."""

from __future__ import annotations

from datetime import datetime, timezone
from fnmatch import fnmatchcase
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
from typing import Literal, Union
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    per_file_sha256: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_policy(self) -> "DigestSpec":
        for name, digest in self.per_file_sha256.items():
            safe_remote_path(name)
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"invalid per-file SHA-256 for {name}")
        declared = (self.sha256 is not None) + bool(self.per_file_sha256)
        if self.policy == "sha256" and declared != 1:
            raise ValueError("sha256 policy requires one aggregate or per-file digest set")
        if self.policy == "capture_after_download" and declared:
            raise ValueError("capture policy forbids declared SHA-256 values")
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
        if len(self.expected_files.values()) != len(set(self.expected_files.values())):
            raise ValueError("Google Drive file IDs must be unique")
        for path, file_id in self.expected_files.items():
            safe_remote_path(path)
            allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
            if not file_id or any(character not in allowed for character in file_id):
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


Source = Union[HuggingFaceSource, GDriveSource, ZenodoSource, DirectSource]


class CheckpointResource(ManifestModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    backend: Literal["huggingface", "gdrive", "zenodo", "direct"]
    provenance: Provenance
    teacher_family: TeacherFamily
    destination: str
    license: str = Field(min_length=1)
    terms_url: str
    digest: DigestSpec
    required_files: list[str] = Field(min_length=1)
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
        if len(self.required_files) != len(set(self.required_files)):
            raise ValueError("required checkpoint files must be unique")
        for name in self.required_files:
            safe_remote_path(name)
        if isinstance(self.source, HuggingFaceSource) and any(
            not any(fnmatchcase(name, pattern) for pattern in self.source.allow_patterns)
            for name in self.required_files
        ):
            raise ValueError("required Hugging Face files must be included by allow patterns")
        if isinstance(self.source, GDriveSource) and set(self.required_files) != set(
            self.source.expected_files
        ):
            raise ValueError("Google Drive required files must exactly match pinned file IDs")
        if isinstance(self.source, ZenodoSource) and set(self.required_files) != set(
            self.source.files
        ):
            raise ValueError("Zenodo required files must exactly match selected record files")
        if isinstance(self.source, DirectSource) and self.required_files != [
            self.source.filename
        ]:
            raise ValueError("direct required file must match the declared filename")
        if self.digest.sha256 is not None and len(self.required_files) != 1:
            raise ValueError("aggregate SHA-256 is incompatible with a multi-file resource")
        if self.digest.per_file_sha256 and set(self.digest.per_file_sha256) != set(
            self.required_files
        ):
            raise ValueError("per-file SHA-256 inventory must match required files exactly")
        return self


class CoverageEntry(ManifestModel):
    model: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    dataset: Literal["lm1b", "owt"]
    resource: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    path: str | None = None
    teacher_family: TeacherFamily | None = None
    sampling_config: str | None = None
    sampling_config_source: Literal["resource", "project"] | None = None
    sampling_config_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    sampling_config_source_commit: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{40}$"
    )

    @model_validator(mode="after")
    def validate_path(self) -> "CoverageEntry":
        if self.path is not None:
            safe_remote_path(self.path)
        config_fields = (self.sampling_config, self.sampling_config_source)
        if any(value is None for value in config_fields) != all(
            value is None for value in config_fields
        ):
            raise ValueError("coverage sampling config requires a path and source")
        if self.sampling_config is not None:
            config = safe_remote_path(self.sampling_config)
            if config.suffix.lower() not in {".json", ".yaml", ".yml"}:
                raise ValueError("coverage sampling config has an unsupported suffix")
        if self.sampling_config_source == "project":
            if (
                self.sampling_config_sha256 is None
                or self.sampling_config_source_commit is None
            ):
                raise ValueError(
                    "project sampling config requires a pinned SHA-256 and source commit"
                )
        elif (
            self.sampling_config_sha256 is not None
            or self.sampling_config_source_commit is not None
        ):
            raise ValueError(
                "only project sampling configs declare SHA-256 and source commit"
            )
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
    sampling_checkpoint: str | None = None
    sampling_config: str | None = None
    prerequisites: list[str] = Field(default_factory=list)
    note: str | None = None

    @model_validator(mode="after")
    def validate_output(self) -> "TrainingRecipe":
        path = safe_remote_path(self.output)
        if path.parts[0] != "checkpoints" or path.parts[1] != self.provenance:
            raise ValueError("recipe output must be inside checkpoints/<provenance>")
        if self.sampling_checkpoint is not None:
            checkpoint = safe_remote_path(self.sampling_checkpoint)
            if checkpoint.suffix.lower() not in {
                ".bin",
                ".ckpt",
                ".pt",
                ".pth",
                ".safetensors",
            }:
                raise ValueError("recipe sampling checkpoint has an unsupported suffix")
        if self.sampling_config is not None:
            config = safe_remote_path(self.sampling_config)
            if config.suffix.lower() not in {".json", ".yaml", ".yml"}:
                raise ValueError("recipe sampling config has an unsupported suffix")
        if self.source in {"sdtt", "di4c"}:
            expected = {
                "uniform_duo": "uniform_to_absorbing",
                "masked_mdlm": "masked_to_absorbing",
            }.get(self.teacher_family)
            if self.teacher_adapter != expected:
                raise ValueError(
                    f"{self.source} recipe for {self.teacher_family} requires {expected}"
                )
            if self.sampling_checkpoint is None or self.sampling_config is None:
                raise ValueError(
                    f"{self.source} recipe requires exact sampling checkpoint and config"
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

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("checkpoint manifest schema_version must be integer 1")
        return value

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
    """Permit real checkpoint acquisition only on the approved Linux server."""

    if system_name != "Linux":
        raise RuntimeError(
            f"checkpoint downloads are Linux server-only; detected {system_name or 'unknown'}"
        )


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


def build_gdrive_commands(source: GDriveSource, staging: Path) -> list[list[str]]:
    """Address each Drive object by immutable file ID, never folder-discovered name."""

    object_root = staging / ".objects"
    return [
        [
            "gdown",
            "--continue",
            file_id,
            "--output",
            str(object_root / f"{file_id}.partial"),
        ]
        for file_id in source.expected_files.values()
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
        timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
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

    quarantined_existing = None
    try:
        target_info = target.lstat()
    except FileNotFoundError:
        target_info = None
    if target_info is not None:
        valid_target = stat.S_ISREG(target_info.st_mode) and target_info.st_size > 0
        if not valid_target:
            effective_quarantine_root = (
                quarantine_root or target.parents[1] / "quarantine"
            )
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            quarantine = _quarantine_path(
                target, effective_quarantine_root, stamp
            )
            os.replace(target, quarantine)
            quarantined_existing = quarantine.as_posix()
        elif digest.sha256 and sha256_file(target) == digest.sha256:
            return {
                "status": "reused",
                "sha256": digest.sha256,
                "size_bytes": target_info.st_size,
                "quarantined": None,
            }
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    if partial.is_symlink():
        raise ValueError(f"checkpoint partial path is a symlink: {partial}")
    if partial.exists() and not partial.is_file():
        raise ValueError(f"checkpoint partial path is not a file: {partial}")
    metadata_path = partial.with_name(partial.name + ".meta.json")

    def discard_partial() -> None:
        partial.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

    def load_resume_metadata() -> dict[str, object] | None:
        if not partial.exists() or partial.stat().st_size == 0:
            metadata_path.unlink(missing_ok=True)
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("url") != source.url:
                raise ValueError("partial URL differs")
            if metadata.get("validator_header") not in {"ETag", "Last-Modified"}:
                raise ValueError("partial has no supported validator")
            if not isinstance(metadata.get("validator_value"), str) or not metadata[
                "validator_value"
            ]:
                raise ValueError("partial validator is empty")
            total_size = metadata.get("total_size")
            if not isinstance(total_size, int) or total_size < partial.stat().st_size:
                raise ValueError("partial total size is invalid")
            return metadata
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            discard_partial()
            return None

    def response_header(response, name: str) -> str | None:
        value = response.headers.get(name)
        if value is not None:
            return value
        lowered = name.lower()
        for key, candidate in response.headers.items():
            if key.lower() == lowered:
                return candidate
        return None

    def content_range(response) -> tuple[int, int, int] | None:
        value = response_header(response, "Content-Range")
        if value is None or not value.startswith("bytes "):
            return None
        try:
            span, total = value.removeprefix("bytes ").split("/", 1)
            start, end = span.split("-", 1)
            return int(start), int(end), int(total)
        except (TypeError, ValueError):
            return None

    resume_metadata = load_resume_metadata()
    for attempt in range(2):
        offset = partial.stat().st_size if partial.exists() and resume_metadata else 0
        complete_partial = bool(
            offset and offset == resume_metadata.get("total_size")
        )
        headers = {}
        if offset:
            headers = {
                "Range": "bytes=0-0" if complete_partial else f"bytes={offset}-",
                "If-Range": str(resume_metadata["validator_value"]),
            }
        request = Request(source.url, headers=headers)
        try:
            response_context = urlopen(request)
        except HTTPError as error:
            if error.code == 416 and offset and attempt == 0:
                discard_partial()
                resume_metadata = None
                continue
            raise
        with response_context as response:
            status = getattr(response, "status", response.getcode())
            response_range = content_range(response)
            if offset and status == 416:
                discard_partial()
                resume_metadata = None
                if attempt == 0:
                    continue
                raise ValueError("HTTP range remained unsatisfiable after restart")
            if complete_partial and status == 206:
                validator_header = str(resume_metadata["validator_header"])
                validator = response_header(response, validator_header)
                probe = response.read()
                with partial.open("rb") as handle:
                    first_byte = handle.read(1)
                trusted = (
                    response_range == (0, 0, resume_metadata["total_size"])
                    and validator == resume_metadata["validator_value"]
                    and probe == first_byte
                )
                if trusted:
                    break
                discard_partial()
                resume_metadata = None
                if attempt == 0:
                    continue
                raise ValueError("complete HTTP partial could not be revalidated")
            if offset and status == 206:
                validator_header = str(resume_metadata["validator_header"])
                validator = response_header(response, validator_header)
                trusted = (
                    response_range is not None
                    and response_range[0] == offset
                    and response_range[1] + 1 == response_range[2]
                    and response_range[2] == resume_metadata["total_size"]
                    and validator == resume_metadata["validator_value"]
                )
                if not trusted:
                    discard_partial()
                    resume_metadata = None
                    if attempt == 0:
                        continue
                    raise ValueError("HTTP range response could not be validated")
                mode = "ab"
                total_size = response_range[2]
            elif offset and status == 200:
                mode = "wb"
                length = response_header(response, "Content-Length")
                total_size = int(length) if length is not None else None
            elif not offset and status in {200, 206}:
                if status == 206 and (
                    response_range is None
                    or response_range[0] != 0
                    or response_range[1] + 1 != response_range[2]
                ):
                    raise ValueError("fresh HTTP range response is incomplete")
                mode = "wb"
                length = response_header(response, "Content-Length")
                total_size = (
                    response_range[2]
                    if response_range is not None
                    else int(length) if length is not None else None
                )
            else:
                if offset and attempt == 0:
                    discard_partial()
                    resume_metadata = None
                    continue
                raise ValueError(f"unexpected HTTP download status {status}")

            etag = response_header(response, "ETag")
            modified = response_header(response, "Last-Modified")
            validator_header = "ETag" if etag else "Last-Modified" if modified else None
            validator_value = etag or modified
            if validator_header and validator_value and total_size is not None:
                atomic_json_write(
                    metadata_path,
                    {
                        "url": source.url,
                        "validator_header": validator_header,
                        "validator_value": validator_value,
                        "total_size": total_size,
                    },
                )
            else:
                metadata_path.unlink(missing_ok=True)
            with partial.open(mode) as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
                handle.flush()
                os.fsync(handle.fileno())
            if total_size is not None and partial.stat().st_size != total_size:
                raise ValueError("HTTP response size differs from validated total")
            break
    else:  # pragma: no cover - both attempts either break or raise
        raise RuntimeError("HTTP download retry loop exhausted")
    if partial.stat().st_size == 0:
        raise FileNotFoundError(f"required checkpoint file is empty: {source.filename}")
    verify_published_file(
        partial,
        size_bytes=published_size_bytes,
        checksum=published_checksum,
    )
    metadata_path.unlink(missing_ok=True)
    result = publish_partial(
        partial,
        target,
        digest,
        quarantine_root=quarantine_root,
    )
    if quarantined_existing is not None and result["quarantined"] is None:
        result["quarantined"] = quarantined_existing
    return result


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


def _verify_required_files(staging: Path, required_files: list[str]) -> None:
    for relative_name in required_files:
        expected = staging.joinpath(*safe_remote_path(relative_name).parts)
        if not expected.is_file() or expected.is_symlink():
            raise FileNotFoundError(f"required checkpoint file is missing: {relative_name}")
        if expected.stat().st_size == 0:
            raise FileNotFoundError(f"required checkpoint file is empty: {relative_name}")


def _digest_for_file(digest: DigestSpec, relative_name: str) -> DigestSpec:
    if digest.policy == "capture_after_download":
        return digest
    if digest.sha256 is not None:
        return digest
    return DigestSpec(policy="sha256", sha256=digest.per_file_sha256[relative_name])


def _verify_resource_digest(
    staging: Path, digest: DigestSpec, required_files: list[str]
) -> None:
    if digest.policy == "capture_after_download":
        return
    for relative_name in required_files:
        expected = _digest_for_file(digest, relative_name).sha256
        path = staging.joinpath(*safe_remote_path(relative_name).parts)
        if expected is None or sha256_file(path) != expected:
            raise ValueError(f"download digest mismatch for {path}")


def _quarantine_unexpected_gdrive_staging(
    partial: Path,
    source: GDriveSource,
    quarantine_root: Path,
) -> list[str]:
    allowed_files = set(source.expected_files)
    allowed_object_files = {
        f".objects/{file_id}.partial" for file_id in source.expected_files.values()
    }
    allowed_files.update(allowed_object_files)
    allowed_directories = {".objects"}
    for name in allowed_files:
        relative = PurePosixPath(name)
        allowed_directories.update(
            parent.as_posix() for parent in relative.parents if parent.as_posix() != "."
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    quarantined = []
    for name in allowed_object_files:
        relative = PurePosixPath(name)
        current = partial
        valid_parents = True
        for part in relative.parts[:-1]:
            current = current / part
            try:
                parent_info = current.lstat()
            except FileNotFoundError:
                valid_parents = False
                break
            if not stat.S_ISDIR(parent_info.st_mode):
                valid_parents = False
                break
        if not valid_parents:
            continue
        item = partial.joinpath(*relative.parts)
        try:
            item_info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISREG(item_info.st_mode) and item_info.st_size > 0:
            continue
        quarantine = _quarantine_path(item, quarantine_root, stamp)
        os.replace(item, quarantine)
        quarantined.append(quarantine.as_posix())
    for item in sorted(
        partial.rglob("*"), key=lambda path: len(path.parts), reverse=True
    ):
        relative_name = item.relative_to(partial).as_posix()
        if item.is_symlink() or not item.is_dir():
            if relative_name not in allowed_files:
                quarantine = _quarantine_path(item, quarantine_root, stamp)
                os.replace(item, quarantine)
                quarantined.append(quarantine.as_posix())
        elif relative_name not in allowed_directories:
            try:
                item.rmdir()
            except OSError:
                pass
    return quarantined


def _publish_directory(partial: Path, destination: Path, quarantine_root: Path) -> str | None:
    if partial.is_symlink():
        raise ValueError(f"checkpoint partial path is a symlink: {partial}")
    for item in partial.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"checkpoint tree contains a symlink: {item}")
        safe_remote_path(item.relative_to(partial).as_posix())
    quarantined = None
    if destination.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
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
        result = download_direct(
            source,
            target,
            _digest_for_file(resource.digest, source.filename),
            quarantine_root=quarantine_root,
        )
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
        _verify_required_files(partial, resource.required_files)
        _verify_resource_digest(partial, resource.digest, resource.required_files)
        quarantined = _publish_directory(partial, destination, quarantine_root)
    elif resource.backend == "gdrive":
        source = resource.source
        partial = safe_checkpoint_destination(
            checkpoint_root, resource.destination + ".partial"
        )
        partial.mkdir(parents=True, exist_ok=True)
        _quarantine_unexpected_gdrive_staging(partial, source, quarantine_root)
        object_root = partial / ".objects"
        object_root.mkdir(parents=True, exist_ok=True)
        commands = build_gdrive_commands(source, partial)
        for (relative_name, file_id), command in zip(
            source.expected_files.items(), commands
        ):
            object_path = safe_checkpoint_destination(
                checkpoint_root,
                f"{resource.destination}.partial/.objects/{file_id}.partial",
            )
            subprocess.run(command, check=True)
            if not object_path.is_file() or object_path.is_symlink():
                raise FileNotFoundError(f"Google Drive file ID {file_id} produced no file")
            declared_path = safe_checkpoint_destination(
                checkpoint_root, f"{resource.destination}.partial/{relative_name}"
            )
            declared_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(object_path, declared_path)
        object_root.rmdir()
        _verify_required_files(partial, resource.required_files)
        observed_inventory = {
            item.relative_to(partial).as_posix()
            for item in partial.rglob("*")
            if item.is_file()
        }
        if observed_inventory != set(source.expected_files):
            raise ValueError("Google Drive staging inventory differs from pinned ID/path map")
        _verify_resource_digest(partial, resource.digest, resource.required_files)
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
                _digest_for_file(resource.digest, filename),
                quarantine_root=quarantine_root,
                published_size_bytes=int(item["size_bytes"]),
                published_checksum=pinned_checksum or published_checksum,
            )
            quarantined = quarantined or result.get("quarantined")
        _verify_required_files(destination, resource.required_files)
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
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "manifest_sha256": sha256_file(manifest_path),
        "resources": records,
    }
    atomic_json_write(root / "artifacts" / "checkpoint_lock.json", lock)
    return lock


def _lstat_regular_checkpoint_path(root: Path, relative: PurePosixPath) -> Path:
    """Reject symlinks/non-directories at every component and require a regular file."""

    current = root.absolute()
    for index, part in enumerate(relative.parts):
        current = current / part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"checkpoint lock path contains a symlink: {current}")
        final = index == len(relative.parts) - 1
        if final and not stat.S_ISREG(info.st_mode):
            raise ValueError(f"checkpoint lock path is not a regular file: {current}")
        if not final and not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"checkpoint lock ancestor is not a directory: {current}")
    return current


def verify_checkpoint_lock(
    root: Path,
    lock: dict[str, object],
    *,
    manifest_path: Path | None = None,
) -> dict[str, object]:
    report: dict[str, object] = {
        "ok": True,
        "schema_status": (
            "verified"
            if type(lock.get("schema_version")) is int and lock.get("schema_version") == 1
            else "invalid"
        ),
        "resources": {},
    }
    if report["schema_status"] != "verified":
        report["ok"] = False
    resources = lock.get("resources")
    if not isinstance(resources, dict) or not resources:
        report["resource_set_status"] = "invalid"
        report["ok"] = False
        resources = resources if isinstance(resources, dict) else {}
    else:
        report["resource_set_status"] = "verified"
    manifest = None
    if manifest_path is not None:
        expected_manifest = lock.get("manifest_sha256")
        observed_manifest = sha256_file(manifest_path) if manifest_path.is_file() else None
        report["manifest_status"] = (
            "verified" if observed_manifest == expected_manifest else "mismatch"
        )
        if report["manifest_status"] != "verified":
            report["ok"] = False
        try:
            manifest = load_checkpoint_manifest(manifest_path)
            expected_ids = set(manifest.resources)
        except (OSError, ValueError):
            report["resource_set_status"] = "invalid"
            report["ok"] = False
        else:
            if set(resources) != expected_ids:
                report["resource_set_status"] = "mismatch"
                report["ok"] = False
    for resource_id, record in resources.items():
        if not isinstance(record, dict):
            report["resources"][resource_id] = {
                "status": "invalid",
                "files": [],
                "error": {
                    "type": "InvalidResourceRecord",
                    "message": f"lock resource {resource_id} must be a mapping",
                },
            }
            report["ok"] = False
            continue
        manifest_resource = manifest.resources.get(resource_id) if manifest is not None else None
        if manifest_resource is not None:
            expected_metadata = {
                "backend": manifest_resource.backend,
                "provenance": manifest_resource.provenance,
                "teacher_family": manifest_resource.teacher_family,
                "destination": f"checkpoints/{manifest_resource.destination}",
            }
            if any(record.get(field) != value for field, value in expected_metadata.items()):
                report["resources"][resource_id] = {
                    "status": "invalid",
                    "error": {"message": "lock resource metadata differs from manifest"},
                }
                report["ok"] = False
                continue
        if record.get("status") not in {"downloaded", "verified"}:
            report["resources"][resource_id] = {
                "status": record.get("status", "unavailable"),
                "error": record.get("error"),
            }
            report["ok"] = False
            continue
        files = record.get("files")
        if not isinstance(files, list) or not files:
            report["resources"][resource_id] = {"status": "invalid", "files": []}
            report["ok"] = False
            continue
        status = "verified"
        details = []
        seen_paths: set[str] = set()
        relative_to_destination: set[str] = set()
        destination_prefix = PurePosixPath(
            f"checkpoints/{manifest_resource.destination}"
            if manifest_resource is not None
            else "checkpoints"
        )
        for expected in files:
            try:
                if not isinstance(expected, dict):
                    raise ValueError("lock file record must be a mapping")
                relative = safe_remote_path(str(expected["path"]))
                if relative == destination_prefix or not relative.is_relative_to(
                    destination_prefix
                ):
                    raise ValueError("lock file lies outside its resource destination")
                path_text = relative.as_posix()
                if path_text in seen_paths:
                    raise ValueError("lock contains a duplicate file path")
                seen_paths.add(path_text)
                size_bytes = expected["size_bytes"]
                digest = expected["sha256"]
                if not isinstance(size_bytes, int) or size_bytes < 0:
                    raise ValueError("invalid locked file size")
                if not isinstance(digest, str) or len(digest) != 64 or any(
                    character not in "0123456789abcdef" for character in digest
                ):
                    raise ValueError("invalid locked SHA-256")
                relative_to_destination.add(
                    relative.relative_to(destination_prefix).as_posix()
                )
                path = _lstat_regular_checkpoint_path(root, relative)
            except FileNotFoundError:
                status = "missing"
                details.append({"path": expected.get("path"), "status": "missing"})
                continue
            except (KeyError, TypeError, ValueError, OSError):
                status = "invalid"
                details.append({"path": expected.get("path"), "status": "invalid"})
                break
            if path.stat().st_size != size_bytes or sha256_file(path) != digest:
                status = "mismatch"
                details.append({"path": expected["path"], "status": "mismatch"})
            else:
                details.append({"path": expected["path"], "status": "verified"})
        if (
            status == "verified"
            and manifest_resource is not None
            and not set(manifest_resource.required_files) <= relative_to_destination
        ):
            status = "invalid"
            details.append({"status": "invalid", "error": "required file missing from lock"})
        report["resources"][resource_id] = {"status": status, "files": details}
        if status != "verified":
            report["ok"] = False
    return report

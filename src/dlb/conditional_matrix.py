"""Canonical fixed-prefix conditional generation matrix construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dlb.conditional_prompts import ConditionalProtocol, load_protocol, verify_prompts
from dlb.io import sha256_file
from dlb.matrix import build_matrix, unsupported_inventory, write_unsupported_inventory
from dlb.registry import ExperimentRegistry


CONDITIONAL_MATRIX_SCHEMA = "dlb-conditional-generation-matrix-v1"


@dataclass(frozen=True)
class ConditionalMatrixTask:
    """One registry-backed conditional request using the C64 protocol."""

    task_id: str
    category: str
    model: str
    dataset: str
    steps: int
    sample_count: Literal[2048]
    seed: int
    environment: str
    adapter: str
    source: str
    provenance: str
    protocol: Literal["c64_zs_v1"]
    conditioning_manifest: str
    conditioning_manifest_sha256: str
    sample_dir: str
    metrics_path: str
    timing_path: str

    @property
    def model_id(self) -> str:
        return self.model

    @property
    def dataset_id(self) -> str:
        return self.dataset

    @property
    def step_count(self) -> int:
        return self.steps


def _root_path(root: Path | None) -> Path:
    return (root or Path(".")).resolve()


def _default_protocol(root: Path) -> ConditionalProtocol:
    """Prefer a supplied project config while keeping isolated matrix tests usable."""

    config = root / "configs" / "conditional.yaml"
    if not config.is_file():
        config = Path(__file__).resolve().parents[2] / "configs" / "conditional.yaml"
    return load_protocol(config)


def _sidecar_path(root: Path, dataset: str) -> Path:
    return root / "data" / "manifests" / f"conditional-{dataset}-c64.json"


def build_conditional_matrix(
    registry: ExperimentRegistry,
    *,
    root: Path | None = None,
    protocol: ConditionalProtocol | None = None,
) -> list[ConditionalMatrixTask]:
    """Expand the supported registry coverage into the isolated C64 result tree.

    Verify both prompt artifacts before expanding tasks so each matrix row is
    already bound to its canonical sidecar manifest.
    """

    root_path = _root_path(root)
    protocol = protocol or _default_protocol(root_path)
    verified_sidecars: dict[str, tuple[Path, str]] = {}
    for dataset in sorted(protocol.datasets):
        manifest = verify_prompts(root_path, dataset, protocol)
        if manifest.dataset != dataset:
            raise ValueError("verified conditional prompt manifest has the wrong dataset")
        sidecar = _sidecar_path(root_path, dataset)
        if sidecar.is_symlink() or not sidecar.is_file():
            raise ValueError(f"conditional prompt manifest is missing or unsafe: {sidecar}")
        verified_sidecars[dataset] = (sidecar, sha256_file(sidecar))
    ordinary = build_matrix(registry, root=root_path, sample_count=2048, seed=protocol.sampling_seed)
    tasks: list[ConditionalMatrixTask] = []
    for task in ordinary:
        sidecar, manifest_sha256 = verified_sidecars[task.dataset]
        sample_dir = root_path / "results" / "conditional" / "samples" / task.dataset / task.model / f"steps_{task.steps}"
        metrics_path = root_path / "results" / "conditional" / "metrics" / task.dataset / task.model / f"steps_{task.steps}" / "metrics.json"
        timing_path = root_path / "results" / "conditional" / "timing" / task.dataset / task.model / f"steps_{task.steps}" / "timing.json"
        tasks.append(
            ConditionalMatrixTask(
                task_id=task.task_id,
                category=task.category,
                model=task.model,
                dataset=task.dataset,
                steps=task.steps,
                sample_count=2048,
                seed=protocol.sampling_seed,
                environment=task.environment,
                adapter=task.adapter,
                source=task.source,
                provenance=task.provenance,
                protocol=protocol.protocol,
                conditioning_manifest=sidecar.as_posix(),
                conditioning_manifest_sha256=manifest_sha256,
                sample_dir=sample_dir.as_posix(),
                metrics_path=metrics_path.as_posix(),
                timing_path=timing_path.as_posix(),
            )
        )
    return tasks


def conditional_unsupported_inventory(registry: ExperimentRegistry) -> list[dict[str, str]]:
    """Return the same explicit unsupported inventory as the ordinary matrix."""

    records = unsupported_inventory(registry)
    # RDLM's conditional protocol category is the small unsupported inventory
    # even though its unconditional registry category has since expanded.
    for record in records:
        if record["model"] == "rdlm" and record["dataset"] == "owt":
            record["category"] = "few"
    return records


def write_conditional_unsupported_inventory(
    path: Path, records: list[dict[str, str]]
) -> Path:
    """Write unsupported cells with the conditional matrix schema header."""

    return write_unsupported_inventory(path, records, schema=CONDITIONAL_MATRIX_SCHEMA)

"""Shared contracts and registry helpers for diffusion language baselines."""

from .io import atomic_json_write, sha256_file
from .registry import load_registry
from .schema import FailureRecord, MetricRecord, RunMetadata, SampleRecord

__all__ = [
    "FailureRecord",
    "MetricRecord",
    "RunMetadata",
    "SampleRecord",
    "atomic_json_write",
    "load_registry",
    "sha256_file",
]

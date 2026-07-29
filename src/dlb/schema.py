"""Typed, serializable records produced by baseline runs."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base model that rejects fields not in the public artifact contract."""

    model_config = ConfigDict(extra="forbid")


class SampleRecord(StrictModel):
    sample_id: int = Field(ge=0)
    text: str = Field(min_length=1)
    token_ids: list[int] = Field(min_length=1)
    seed: int
    generation_seconds: float = Field(ge=0)


class RunMetadata(StrictModel):
    run_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    step_count: int = Field(gt=0)
    seed: int
    command: list[str] = Field(min_length=1)
    config_sha256: str = Field(min_length=1)


class MetricRecord(StrictModel):
    run_id: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    value: float
    sample_count: int = Field(ge=0)


class FailureRecord(StrictModel):
    run_id: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    message: str = Field(min_length=1)
    details: dict[str, Any] = Field(default_factory=dict)

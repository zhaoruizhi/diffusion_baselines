"""Typed, serializable records produced by baseline runs."""

import math
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt, field_validator


class StrictModel(BaseModel):
    """Base model that rejects fields not in the public artifact contract."""

    model_config = ConfigDict(extra="forbid")


class SampleRecord(StrictModel):
    sample_id: StrictInt = Field(ge=0)
    text: str = Field(min_length=1)
    token_ids: list[Annotated[StrictInt, Field(ge=0)]] = Field(min_length=1)
    seed: StrictInt
    generation_seconds: Annotated[FiniteFloat, Field(ge=0)]

    @field_validator("text")
    @classmethod
    def require_non_whitespace_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty or whitespace")
        return value

    @field_validator("generation_seconds", mode="before")
    @classmethod
    def require_builtin_finite_float(cls, value: object) -> object:
        if type(value) is not float or not math.isfinite(value) or value < 0:
            raise ValueError("generation_seconds must be a finite non-negative built-in float")
        return value


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

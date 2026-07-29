"""Validation and loading for the experiment coverage registry."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


MANY_STEPS = [1, 2, 4, 8, 16, 32, 1024]
FEW_STEPS = [1, 2, 4, 8, 16, 32]

MODEL_IDENTIFIERS = {
    "flm": ("many", "dlb-flm", "flm", "flm"),
    "fmlm": ("few", "dlb-flm", "flm", "flm"),
    "langflow": ("many", "dlb-langflow", "langflow", "langflow"),
    "duo": ("many", "dlb-duo", "duo", "duo"),
    "duo_dcd": ("few", "dlb-duo", "duo", "duo"),
    "mdlm": ("many", "dlb-mdlm", "mdlm", "mdlm"),
    "candi": ("many", "dlb-candi", "candi", "candi"),
    "rdlm": ("many", "dlb-rdlm", "rdlm", "rdlm"),
    "mdlm_sdtt": ("few", "dlb-sdtt", "sdtt", "sdtt"),
    "duo_di4c": ("few", "dlb-di4c", "di4c", "di4c"),
    "mdlm_di4c": ("few", "dlb-di4c", "di4c", "di4c"),
}

Category = Literal["many", "few"]
SupportStatus = Literal["supported", "unsupported"]
Provenance = Literal["official", "reference_reproduction", "self_trained"]


class RegistryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetSupport(RegistryModel):
    status: SupportStatus
    provenance: Provenance | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_support_details(self) -> "DatasetSupport":
        if self.status == "supported":
            if self.provenance is None or self.reason is not None:
                raise ValueError("supported datasets require provenance and no reason")
        elif self.reason is None or self.reason.strip() == "" or self.provenance is not None:
            raise ValueError("unsupported datasets require a reason and no provenance")
        return self


class ModelRegistryEntry(RegistryModel):
    category: Category
    environment: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    adapter: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    source: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    datasets: dict[Literal["lm1b", "owt"], DatasetSupport]

    @model_validator(mode="after")
    def require_all_datasets(self) -> "ModelRegistryEntry":
        if set(self.datasets) != {"lm1b", "owt"}:
            raise ValueError("each model must declare lm1b and owt support")
        return self


class ExperimentRegistry(RegistryModel):
    step_grids: dict[Category, list[int]]
    models: dict[str, ModelRegistryEntry]

    @model_validator(mode="after")
    def validate_coverage(self) -> "ExperimentRegistry":
        if self.step_grids != {"many": MANY_STEPS, "few": FEW_STEPS}:
            raise ValueError("step grids must match the prescribed many and few schedules")
        if set(self.models) != set(MODEL_IDENTIFIERS):
            raise ValueError("registry must contain the complete baseline model scope")
        for model_id, model in self.models.items():
            if (model.category, model.environment, model.adapter, model.source) != (
                MODEL_IDENTIFIERS[model_id]
            ):
                raise ValueError(f"registry identifiers do not match {model_id}")
        return self


def load_registry(path: Path) -> ExperimentRegistry:
    """Load and validate a registry YAML document from *path*."""

    with path.open(encoding="utf-8") as registry_file:
        document = yaml.safe_load(registry_file)
    if not isinstance(document, dict):
        raise ValueError("registry document must be a mapping")
    return ExperimentRegistry.model_validate(document)

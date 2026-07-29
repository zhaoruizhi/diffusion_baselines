from pathlib import Path

import pytest

from dlb.registry import load_registry


@pytest.fixture
def registry():
    return load_registry(Path("configs/experiments.yaml"))


def test_registry_contains_full_scope(registry):
    assert set(registry.models) == {
        "flm",
        "langflow",
        "duo",
        "mdlm",
        "candi",
        "rdlm",
        "fmlm",
        "duo_dcd",
        "duo_di4c",
        "mdlm_sdtt",
        "mdlm_di4c",
    }
    assert registry.step_grids["many"] == [1, 2, 4, 8, 16, 32, 1024]
    assert registry.step_grids["few"] == [1, 2, 4, 8, 16, 32]


def test_unsupported_cells_have_reason(registry):
    for model in registry.models.values():
        for support in model.datasets.values():
            if support.status == "unsupported":
                assert len(support.reason) >= 20


def test_supported_cells_keep_provenance_separate_from_status(registry):
    supported = registry.models["duo"].datasets["lm1b"]

    assert supported.status == "supported"
    assert supported.provenance == "reference_reproduction"
    assert supported.reason is None


def test_loader_rejects_unsupported_cell_with_provenance(tmp_path):
    registry_config = tmp_path / "invalid.yaml"
    registry_config.write_text(
        """
step_grids:
  many: [1, 2, 4, 8, 16, 32, 1024]
  few: [1, 2, 4, 8, 16, 32]
models:
  flm:
    category: many
    environment: dlb-flm
    adapter: flm
    source: flm
    datasets:
      lm1b:
        status: unsupported
        reason: Unsupported for a clearly documented project reason.
        provenance: official
""".strip()
    )

    with pytest.raises(ValueError):
        load_registry(registry_config)


def test_loader_rejects_wrong_adapter_for_known_model(tmp_path):
    registry_config = tmp_path / "invalid.yaml"
    registry_config.write_text(
        Path("configs/experiments.yaml")
        .read_text()
        .replace("    adapter: flm", "    adapter: wrong-adapter", 1)
    )

    with pytest.raises(ValueError):
        load_registry(registry_config)

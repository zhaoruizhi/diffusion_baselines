from pathlib import Path

import pytest

from dlb.matrix import build_matrix
from dlb.registry import load_registry


ROOT = Path(__file__).parents[1]


@pytest.fixture
def registry():
    return load_registry(ROOT / "configs/experiments.yaml")


def test_conditional_matrix_matches_supported_unconditional_tasks(registry, tmp_path):
    """Catch a conditional matrix that diverges from the supported registry cells."""

    from dlb.conditional_matrix import build_conditional_matrix

    ordinary = build_matrix(registry, root=tmp_path)
    conditional = build_conditional_matrix(registry, root=tmp_path)

    assert len(conditional) == len(ordinary) == 137
    assert [(task.model, task.dataset, task.steps) for task in conditional] == [
        (task.model, task.dataset, task.steps) for task in ordinary
    ]
    assert all("/results/conditional/" in task.sample_dir for task in conditional)
    assert all(task.sample_count == 2048 for task in conditional)


def test_conditional_unsupported_inventory_is_rdlm_owt(registry):
    """Catch conditional reporting that omits the explicit unsupported RDLM/OWT cell."""

    from dlb.conditional_matrix import conditional_unsupported_inventory

    assert conditional_unsupported_inventory(registry) == [
        {
            "status": "unsupported",
            "model": "rdlm",
            "dataset": "owt",
            "category": "few",
            "reason": registry.models["rdlm"].datasets["owt"].reason,
        }
    ]

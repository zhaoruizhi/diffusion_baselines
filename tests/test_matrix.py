from pathlib import Path

import pytest

from dlb.matrix import (
    MATRIX_COLUMNS,
    MATRIX_SCHEMA,
    build_matrix,
    read_matrix,
    unsupported_inventory,
    validate_matrix,
    write_matrix,
    write_unsupported_inventory,
)
from dlb.registry import load_registry


ROOT = Path(__file__).parents[1]


@pytest.fixture
def registry():
    return load_registry(ROOT / "configs/experiments.yaml")


def test_matrix_cardinality_and_declared_steps(registry):
    tasks = build_matrix(registry)
    assert len(tasks) == 132
    assert sum(task.category == "many" for task in tasks) == 72
    assert sum(task.category == "few" for task in tasks) == 60
    assert sum(task.category == "fixed_1024" for task in tasks) == 0
    assert [
        task.steps for task in tasks if (task.model, task.dataset) == ("rdlm", "lm1b")
    ] == [1000, 1024]
    for task in tasks:
        model = registry.models[task.model]
        assert task.steps in (model.step_override or registry.step_grids[task.category])
        assert task.sample_count == 1024
        assert task.seed == 42


def test_matrix_is_stably_sorted_and_round_trips_without_eval(registry, tmp_path):
    tasks = build_matrix(registry, root=tmp_path)
    output = write_matrix(tmp_path / "generation.tsv", tasks)
    content = output.read_text(encoding="utf-8")
    assert content.splitlines()[0] == f"# schema={MATRIX_SCHEMA}"
    assert content.splitlines()[1].split("\t") == list(MATRIX_COLUMNS)
    restored = read_matrix(output)
    assert restored == tasks
    assert validate_matrix(output, registry) == tasks
    assert len({task.task_id for task in restored}) == 132
    assert "eval(" not in content


def test_matrix_rejects_duplicate_task_ids(tmp_path, registry):
    tasks = build_matrix(registry, root=tmp_path)
    output = write_matrix(tmp_path / "generation.tsv", tasks)
    lines = output.read_text(encoding="utf-8").splitlines()
    lines.append(lines[2])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_matrix(output)


def test_unsupported_inventory_is_not_expanded(registry, tmp_path):
    records = unsupported_inventory(registry)
    assert {(item["model"], item["dataset"]) for item in records} == {
        ("rdlm", "owt"),
    }
    assert all(item["reason"] for item in records)
    output = write_unsupported_inventory(tmp_path / "unsupported.tsv", records)
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[0] == f"# schema={MATRIX_SCHEMA}-unsupported"
    assert len(lines) == 3

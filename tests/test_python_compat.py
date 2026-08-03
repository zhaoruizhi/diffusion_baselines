import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]


def python_runtime_files() -> list[Path]:
    return sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))


def test_runtime_code_avoids_python311_datetime_utc_symbol() -> None:
    """Catch imports that break Python 3.9/3.10 baseline environments."""

    violations: list[str] = []
    for path in python_runtime_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "datetime":
                for alias in node.names:
                    if alias.name == "UTC":
                        violations.append(f"{path.relative_to(ROOT)} imports datetime.UTC")
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "UTC"
                and isinstance(node.value, ast.Name)
                and node.value.id == "datetime"
            ):
                violations.append(f"{path.relative_to(ROOT)} uses datetime.UTC")

    assert violations == []


def test_runtime_pep604_annotations_are_future_gated_for_python39() -> None:
    """Catch imports that evaluate A | None annotations under Python 3.9."""

    violations: list[str] = []
    for path in python_runtime_files():
        text = path.read_text(encoding="utf-8")
        if "| None" not in text and " | " not in text:
            continue
        tree = ast.parse(text, filename=str(path))
        future_annotations = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
            and any(alias.name == "annotations" for alias in node.names)
            for node in tree.body
        )
        if not future_annotations:
            violations.append(f"{path.relative_to(ROOT)} uses PEP 604 annotations")

    assert violations == []


def test_runtime_type_aliases_avoid_pep604_expressions_for_python39() -> None:
    """Catch top-level A | B aliases that execute before Python 3.10."""

    violations: list[str] = []
    for path in python_runtime_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp):
                if isinstance(node.value.op, ast.BitOr):
                    violations.append(
                        f"{path.relative_to(ROOT)} has runtime PEP 604 type alias"
                    )

    assert violations == []

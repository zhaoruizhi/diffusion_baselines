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

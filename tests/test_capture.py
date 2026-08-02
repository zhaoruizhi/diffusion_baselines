from pathlib import Path
import sys
from types import ModuleType

import dlb.adapters.capture as capture_module


def test_load_entrypoint_registers_module_for_hydra_relative_configs(
    tmp_path: Path,
) -> None:
    """Catch Hydra entrypoints losing module metadata for relative config paths."""

    entrypoint = tmp_path / "main.py"
    entrypoint.write_text(
        """
import sys

MODULE_FILE_AT_IMPORT = getattr(sys.modules.get(__name__), "__file__", None)

def main():
    pass
""",
        encoding="utf-8",
    )

    try:
        module = capture_module._load_entrypoint(entrypoint)
    finally:
        sys.modules.pop("dlb_pinned_upstream_main", None)

    assert Path(module.MODULE_FILE_AT_IMPORT) == entrypoint


def test_run_main_points_hydra_at_sibling_config_directory(tmp_path: Path) -> None:
    """Catch dynamic imports making Hydra treat sibling configs as a missing package."""

    entrypoint = tmp_path / "main.py"
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    entrypoint.write_text("# fixture\n", encoding="utf-8")
    observed: dict[str, list[str]] = {}
    module = ModuleType("fixture_upstream")

    def main() -> None:
        observed["argv"] = list(sys.argv)

    module.main = main

    capture_module._run_main(module, entrypoint, ["mode=sample_eval"])

    assert observed["argv"] == [
        str(entrypoint),
        f"--config-path={config_dir.resolve()}",
        "mode=sample_eval",
    ]

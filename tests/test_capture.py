from pathlib import Path
import sys

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

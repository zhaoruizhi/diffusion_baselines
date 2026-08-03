import json
from pathlib import Path
import sys
from types import SimpleNamespace
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


def test_teacher_capture_accepts_upstream_python_list_samples(tmp_path: Path) -> None:
    """Catch FLM restore_model_and_sample returning list rows instead of a tensor."""

    class Tokenizer:
        def batch_decode(self, rows):
            assert rows == [[1, 2], [3, 4]]
            return ["one", "two"]

    class TrainerBase:
        tokenizer = Tokenizer()

        def restore_model_and_sample(self):
            return [[1, 2], [3, 4]]

    module = ModuleType("fixture_teacher")
    module.algo = SimpleNamespace(
        trainer_base=SimpleNamespace(TrainerBase=TrainerBase)
    )
    module.main = lambda: TrainerBase().restore_model_and_sample()
    capture_path = tmp_path / "capture.json"
    invocation = capture_module.CaptureInvocation(
        entrypoint=tmp_path / "main.py",
        capture_path=capture_path,
    )

    capture_module._capture_teacher(module, invocation, [])

    assert json.loads(capture_path.read_text(encoding="utf-8")) == {
        "samples": [
            {"sample_id": 0, "text": "one", "token_ids": [1, 2]},
            {"sample_id": 1, "text": "two", "token_ids": [3, 4]},
        ],
        "schema": "dlb-upstream-token-capture-v1",
    }

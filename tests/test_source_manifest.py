import os
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml


EXPECTED = {
    "flm": "https://github.com/david3684/flm.git",
    "langflow": "https://github.com/nealchen2003/LangFlow.git",
    "duo": "https://github.com/s-sahoo/duo.git",
    "mdlm": "https://github.com/kuleshov-group/mdlm.git",
    "candi": "https://github.com/patrickpynadath1/candi-diffusion.git",
    "rdlm": "https://github.com/harryjo97/RDLM.git",
    "sdtt": "https://github.com/jdeschena/sdtt.git",
    "di4c": "https://github.com/sony/di4c.git",
}


@pytest.fixture
def source_manifest():
    with Path("artifacts/sources.yaml").open(encoding="utf-8") as manifest_file:
        return yaml.safe_load(manifest_file)


def test_source_manifest_has_exact_repositories(source_manifest):
    assert {name: source["url"] for name, source in source_manifest.items()} == EXPECTED
    assert all(len(source["commit"]) == 40 for source in source_manifest.values())


def test_fetch_sources_uses_active_conda_python(tmp_path):
    conda_prefix = tmp_path / "conda"
    python_bin = conda_prefix / "bin" / "python"
    python_log = tmp_path / "python.log"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            printf '%s\n' "$0" > "$FAKE_PYTHON_LOG"
            printf '%s\n' "$@" >> "$FAKE_PYTHON_LOG"
            exit 42
            """
        )
    )
    python_bin.chmod(0o755)

    environment = {
        **os.environ,
        "CONDA_PREFIX": str(conda_prefix),
        "FAKE_PYTHON_LOG": str(python_log),
    }
    environment.pop("DLB_PYTHON", None)
    environment.pop("PYTHON_BIN", None)

    completed = subprocess.run(
        ["bash", "scripts/fetch_sources.sh"],
        cwd=Path.cwd(),
        env=environment,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 42, completed.stderr
    assert python_log.read_text().splitlines() == [str(python_bin), "-", str(Path.cwd())]

import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[1]


def test_run_one_exports_project_src_to_method_environment(tmp_path: Path) -> None:
    """Catch manual single-task runs losing dlb package imports after conda env switch."""

    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        """#!/usr/bin/env python3
import sys

if sys.argv[1:3] == ["-m", "dlb.runner"] and "--validate-only" in sys.argv:
    print("dlb-candi")
    raise SystemExit(0)
raise SystemExit(99)
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    conda_log = tmp_path / "conda.jsonl"
    fake_conda = tmp_path / "fake-conda"
    fake_conda.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

Path(os.environ["FAKE_CONDA_LOG"]).write_text(
    json.dumps({
        "argv": sys.argv[1:],
        "pythonpath": os.environ.get("PYTHONPATH"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }) + "\\n",
    encoding="utf-8",
)
raise SystemExit(0)
""",
        encoding="utf-8",
    )
    fake_conda.chmod(0o755)

    environment = {
        **os.environ,
        "DLB_PYTHON": str(fake_python),
        "DLB_CONDA": str(fake_conda),
        "FAKE_CONDA_LOG": str(conda_log),
        "CUDA_VISIBLE_DEVICES": "6",
    }
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_one.sh"),
            "--model",
            "candi",
            "--dataset",
            "lm1b",
            "--steps",
            "2",
            "--num-samples",
            "1",
            "--seed",
            "42",
            "--results-root",
            str(ROOT / "results" / "smoke"),
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(conda_log.read_text(encoding="utf-8"))
    assert record["argv"][:4] == ["run", "-n", "dlb-candi", "env"]
    assert record["argv"][4] == f"PYTHONPATH={ROOT / 'src'}"
    assert record["argv"][5:8] == ["python", "-m", "dlb.runner"]
    assert record["pythonpath"] is None
    assert record["cuda_visible_devices"] == "6"

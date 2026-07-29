import re
import json
import os
from pathlib import Path
import subprocess

import pytest
import yaml

from dlb.registry import load_registry


ENVIRONMENT_DIR = Path("envs")
EXPECTED_ENVIRONMENTS = {
    "dlb-flm",
    "dlb-langflow",
    "dlb-duo",
    "dlb-mdlm",
    "dlb-candi",
    "dlb-rdlm",
    "dlb-sdtt",
    "dlb-di4c",
    "dlb-eval",
}


@pytest.fixture
def registry():
    return load_registry(Path("configs/experiments.yaml"))


def load_environments():
    return {
        path.stem: yaml.safe_load(path.read_text())
        for path in sorted(ENVIRONMENT_DIR.glob("*.yml"))
        if not path.name.startswith(".")
    }


def pip_dependencies(environment):
    return next(
        dependency["pip"]
        for dependency in environment["dependencies"]
        if isinstance(dependency, dict) and "pip" in dependency
    )


def pinned_pip_dependencies(environment):
    return {
        package.split("==", 1)[0]: package.split("==", 1)[1]
        for package in pip_dependencies(environment)
        if "==" in package
    }


def test_all_declared_environments_exist(registry):
    environments = load_environments()
    names = {environment["name"] for environment in environments.values()}

    assert {model.environment for model in registry.models.values()} <= names
    assert names == EXPECTED_ENVIRONMENTS


def test_gpu_environments_pin_python_torch_and_cuda_strategy():
    for environment in load_environments().values():
        dependencies = environment["dependencies"]
        rendered_dependencies = "\n".join(
            dependency if isinstance(dependency, str) else ""
            for dependency in dependencies
        )
        pip_packages = "\n".join(pip_dependencies(environment))

        assert re.search(r"^python=", rendered_dependencies, re.MULTILINE)
        assert "torch==" in pip_packages or re.search(
            r"^pytorch=", rendered_dependencies, re.MULTILINE
        )
        assert re.search(r"^pytorch-cuda=", rendered_dependencies, re.MULTILINE)
        assert environment["channels"]


def test_special_upstream_constraints_are_explicit():
    environments = load_environments()

    assert "flash-attn==2.8.3" not in pip_dependencies(environments["flm"])
    assert "flash-attn==2.7.4.post1" not in pip_dependencies(environments["duo"])
    assert "python=3.9" in environments["rdlm"]["dependencies"]
    assert "pytorch=2.3.1" in environments["rdlm"]["dependencies"]
    assert all("nightly" not in package for package in pip_dependencies(environments["sdtt"]))
    assert "torchdata==0.8.0" in pip_dependencies(environments["sdtt"])


def test_conda_pytorch_environments_do_not_override_its_torch_stack_with_pip():
    for environment in load_environments().values():
        assert all(
            not package.startswith(("torch==", "torchvision==", "triton=="))
            for package in pip_dependencies(environment)
        )

    for name in {"flm", "duo", "mdlm", "sdtt", "di4c"}:
        assert "torchvision=0.20.1" in load_environments()[name]["dependencies"]


def test_upstream_entrypoint_dependencies_are_explicit():
    environments = load_environments()

    assert "rich==14.2.0" in pip_dependencies(environments["flm"])
    assert "rich==13.7.1" in pip_dependencies(environments["duo"])
    assert "rich==13.7.1" in pip_dependencies(environments["mdlm"])
    assert "rich==14.2.0" in pip_dependencies(environments["candi"])


def test_known_pinned_dependency_pairs_use_compatible_versions():
    environments = load_environments()
    compatibility_matrix = {
        ("duo", "datasets", "2.15.0", "fsspec"): {"2023.10.0"},
        ("candi", "numpy", "1.24.3", "scipy"): {"1.11.4"},
    }

    for (environment, left, left_version, right), compatible_versions in (
        compatibility_matrix.items()
    ):
        packages = pinned_pip_dependencies(environments[environment])
        assert packages[left] == left_version
        assert packages[right] in compatible_versions


@pytest.fixture
def fake_conda(tmp_path):
    command = tmp_path / "fake-conda"
    command.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

arguments = sys.argv[1:]
with Path(os.environ["FAKE_CONDA_LOG"]).open("a") as log:
    log.write(json.dumps(arguments) + "\\n")

if arguments == ["env", "list", "--json"]:
    mode = os.environ.get("FAKE_ENV_LIST_MODE", "")
    if mode == "failure":
        raise SystemExit(23)
    if mode == "malformed":
        print("not-json")
        raise SystemExit(0)
    if mode == "wrong_type":
        print(json.dumps([]))
        raise SystemExit(0)
    if mode == "foreign":
        print(json.dumps({"envs": ["/foreign/dlb-langflow"]}))
        raise SystemExit(0)
    print(json.dumps({"envs": os.environ.get("FAKE_ENVS", "").split(os.pathsep)}))
    raise SystemExit(0)

if arguments == ["info", "--json"]:
    mode = os.environ.get("FAKE_INFO_MODE", "")
    if mode == "failure":
        raise SystemExit(23)
    if mode == "malformed":
        print("not-json")
        raise SystemExit(0)
    print(json.dumps({"envs_dirs": ["/opt/conda/envs"]}))
    raise SystemExit(0)

failure = os.environ.get("FAKE_FAIL_SUBSTRING", "")
if failure and any(failure in argument for argument in arguments):
    raise SystemExit(23)

if arguments[:1] == ["run"]:
    output = os.environ.get("FAKE_PROBE_OUTPUT")
    if output is not None:
        print(output)
    else:
        modules = arguments[6:]
        payload = {
            "environment": arguments[5],
            "python": "3.11",
            "torch": "2.5.1",
            "torch_cuda": "12.4",
            "cuda_available": True,
            "imports": {module: True for module in modules},
        }
        print("DLB_ENV_PROBE_V1:" + json.dumps(payload, sort_keys=True))
"""
    )
    command.chmod(0o755)
    return command


def run_script(script_name, fake_conda, tmp_path, names, **extra_environment):
    log = tmp_path / "conda.log"
    environment = {
        **os.environ,
        "DLB_CONDA": str(fake_conda),
        "DLB_ENV_NAMES": ",".join(names),
        "FAKE_CONDA_LOG": str(log),
        **extra_environment,
    }
    completed = subprocess.run(
        ["bash", str(ENVIRONMENT_DIR / script_name)],
        cwd=Path.cwd(),
        env=environment,
        text=True,
        capture_output=True,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return completed, calls


def test_create_all_creates_absent_environment_then_installs_flash_attention(
    fake_conda, tmp_path
):
    completed, calls = run_script("create_all.sh", fake_conda, tmp_path, ["dlb-flm"])

    assert completed.returncode == 0, completed.stderr
    create_call = next(call for call in calls if call[:3] == ["env", "create", "--file"])
    assert Path(create_call[3]).name == "flm.yml"
    assert [
        "run",
        "-n",
        "dlb-flm",
        "python",
        "-m",
        "pip",
        "install",
        "flash-attn==2.8.3",
        "--no-build-isolation",
    ] in calls


def test_create_all_updates_existing_environment_without_pruning(fake_conda, tmp_path):
    completed, calls = run_script(
        "create_all.sh",
        fake_conda,
        tmp_path,
        ["dlb-langflow"],
        FAKE_ENVS="/opt/conda/envs/dlb-langflow",
    )

    assert completed.returncode == 0, completed.stderr
    update_call = next(call for call in calls if call[:3] == ["env", "update", "--file"])
    assert Path(update_call[3]).name == "langflow.yml"
    assert update_call == [
        "env",
        "update",
        "--file",
        update_call[3],
        "--prune=false",
    ]


def test_create_all_reports_failures_but_continues_other_environments(fake_conda, tmp_path):
    completed, calls = run_script(
        "create_all.sh",
        fake_conda,
        tmp_path,
        ["dlb-langflow", "dlb-eval"],
        FAKE_FAIL_SUBSTRING="langflow.yml",
    )

    assert completed.returncode != 0
    assert "dlb-langflow" in completed.stderr
    assert any(
        call[:3] == ["env", "create", "--file"] and Path(call[3]).name == "eval.yml"
        for call in calls
    )


@pytest.mark.parametrize(
    ("environment_variable", "value"),
    [
        ("FAKE_ENV_LIST_MODE", "malformed"),
        ("FAKE_ENV_LIST_MODE", "wrong_type"),
        ("FAKE_ENV_LIST_MODE", "failure"),
        ("FAKE_INFO_MODE", "malformed"),
        ("FAKE_INFO_MODE", "failure"),
        ("FAKE_ENV_LIST_MODE", "foreign"),
    ],
)
def test_create_all_refuses_unreliable_environment_discovery(
    fake_conda, tmp_path, environment_variable, value
):
    completed, calls = run_script(
        "create_all.sh",
        fake_conda,
        tmp_path,
        ["dlb-langflow"],
        **{environment_variable: value},
    )

    assert completed.returncode != 0
    assert "discovery" in completed.stderr
    assert not any(call[:2] == ["env", "create"] for call in calls)


def test_create_all_distinguishes_exact_existing_name_from_absence(fake_conda, tmp_path):
    existing, existing_calls = run_script(
        "create_all.sh",
        fake_conda,
        tmp_path,
        ["dlb-langflow"],
        FAKE_ENVS="/opt/conda/envs/dlb-langflow",
    )
    absent, absent_calls = run_script(
        "create_all.sh", fake_conda, tmp_path, ["dlb-langflow"]
    )

    assert existing.returncode == 0
    assert any(call[:2] == ["env", "update"] for call in existing_calls)
    assert absent.returncode == 0
    assert any(call[:2] == ["env", "create"] for call in absent_calls)


def test_verify_all_emits_json_for_each_environment_and_fails_in_aggregate(
    fake_conda, tmp_path
):
    completed, calls = run_script(
        "verify_all.sh",
        fake_conda,
        tmp_path,
        ["dlb-langflow", "dlb-duo"],
        FAKE_FAIL_SUBSTRING="dlb-duo",
    )

    records = [json.loads(line) for line in completed.stdout.splitlines()]
    assert completed.returncode != 0
    assert records[0]["environment"] == "dlb-langflow"
    assert records[0]["cuda_available"] is True
    assert "torchvision" in calls[-1]
    assert "flash_attn" in calls[-1]
    assert "rich" in calls[-1]
    assert records[1] == {
        "environment": "dlb-duo",
        "python": None,
        "torch": None,
        "torch_cuda": None,
        "cuda_available": False,
        "imports": {},
        "error": "verification probe failed",
    }


@pytest.mark.parametrize(
    "probe_output",
    [
        "run -n dlb-langflow python",
        "noise\nDLB_ENV_PROBE_V1:{}",
        "DLB_ENV_PROBE_V1:not-json",
    ],
)
def test_verify_all_rejects_untrusted_manager_stdout(fake_conda, tmp_path, probe_output):
    completed, _ = run_script(
        "verify_all.sh",
        fake_conda,
        tmp_path,
        ["dlb-langflow"],
        FAKE_PROBE_OUTPUT=probe_output,
    )

    records = [json.loads(line) for line in completed.stdout.splitlines()]
    assert completed.returncode != 0
    assert records == [
        {
            "environment": "dlb-langflow",
            "python": None,
            "torch": None,
            "torch_cuda": None,
            "cuda_available": False,
            "imports": {},
            "error": "verification probe failed",
        }
    ]


def test_verify_all_rejects_echo_manager_output(fake_conda, tmp_path):
    completed, _ = run_script(
        "verify_all.sh", fake_conda, tmp_path, ["dlb-langflow"], DLB_CONDA="/bin/echo"
    )

    records = [json.loads(line) for line in completed.stdout.splitlines()]
    assert completed.returncode != 0
    assert records[0]["error"] == "verification probe failed"


def test_verify_all_escapes_unknown_environment_names(fake_conda, tmp_path):
    completed, _ = run_script(
        "verify_all.sh", fake_conda, tmp_path, ['bad"name']
    )

    records = [json.loads(line) for line in completed.stdout.splitlines()]
    assert completed.returncode != 0
    assert records[0]["environment"] == 'bad"name'
    assert records[0]["error"] == "unknown environment"


def test_verify_all_checks_the_complete_method_import_mapping(fake_conda, tmp_path):
    expected_imports = {
        "dlb-flm": {
            "datasets", "einops", "flash_attn", "fsspec", "hydra", "lightning",
            "omegaconf", "rich", "scipy", "timm", "tokenizers", "torchmetrics",
            "tqdm", "transformers", "triton", "wandb",
        },
        "dlb-langflow": {"einops", "huggingface_hub", "safetensors", "transformers"},
        "dlb-duo": {
            "datasets", "einops", "flash_attn", "fsspec", "h5py", "hydra",
            "lightning", "omegaconf", "rich", "timm", "tokenizers", "torchmetrics",
                "torchvision", "tqdm", "transformers", "triton", "wandb",
        },
        "dlb-mdlm": {
            "causal_conv1d", "datasets", "einops", "flash_attn", "fsspec", "hydra",
            "lightning", "mamba_ssm", "omegaconf", "rich", "timm", "transformers", "wandb",
        },
        "dlb-candi": {
            "datasets", "einops", "evaluate", "flash_attn", "fsspec", "hydra", "lightning",
            "omegaconf", "rich", "scipy", "tokenizers", "torchmetrics", "tqdm", "transformers",
        },
        "dlb-rdlm": {
                "accelerate", "datasets", "einops", "fsspec", "hydra", "numpy", "omegaconf",
                "scipy", "tokenizers", "tqdm", "transformers", "wandb",
        },
        "dlb-sdtt": {
            "datasets", "einops", "flash_attn", "fsspec", "huggingface_hub", "hydra", "lightning",
            "loguru", "mauve", "omegaconf", "pandas", "tensorboard", "timm", "tokenizers",
            "torchdata", "tqdm", "transformers", "wandb",
        },
        "dlb-di4c": {
            "datasets", "einops", "flash_attn", "fsspec", "huggingface_hub", "hydra", "lightning",
            "loguru", "mauve", "omegaconf", "pandas", "tensorboard", "timm", "tokenizers",
            "torchdata", "tqdm", "transformers", "wandb",
        },
        "dlb-eval": {"accelerate", "datasets", "evaluate", "fsspec", "mauve", "sacrebleu", "scipy", "tokenizers", "transformers"},
    }
    completed, calls = run_script(
        "verify_all.sh", fake_conda, tmp_path, list(expected_imports)
    )

    assert completed.returncode == 0, completed.stderr
    probe_calls = [call for call in calls if call[:1] == ["run"]]
    assert len(probe_calls) == len(expected_imports)
    assert {
        call[5]: set(call[6:])
        for call in probe_calls
    } == expected_imports


def test_pack_all_writes_archives_without_removing_environment(fake_conda, tmp_path):
    packer = tmp_path / "fake-conda-pack"
    packer.write_text(
        """#!/usr/bin/env python3
import pathlib
import sys
output = pathlib.Path(sys.argv[sys.argv.index("-o") + 1])
output.write_bytes(b"archive")
"""
    )
    packer.chmod(0o755)
    artifact_dir = tmp_path / "artifacts"

    completed, calls = run_script(
        "pack_all.sh",
        fake_conda,
        tmp_path,
        ["dlb-langflow"],
        DLB_CONDA_PACK=str(packer),
        DLB_ARTIFACT_DIR=str(artifact_dir),
    )

    assert completed.returncode == 0, completed.stderr
    assert (artifact_dir / "dlb-langflow.tar.gz").read_bytes() == b"archive"
    assert all("remove" not in call for call in calls)


def test_pack_all_rejects_a_non_file_destination_before_atomic_publication(
    fake_conda, tmp_path
):
    packer = tmp_path / "fake-conda-pack"
    packer.write_text(
        """#!/usr/bin/env python3
import pathlib
import sys
output = pathlib.Path(sys.argv[sys.argv.index("-o") + 1])
output.write_bytes(b"archive")
"""
    )
    packer.chmod(0o755)
    artifact_dir = tmp_path / "artifacts"
    (artifact_dir / "dlb-langflow.tar.gz").mkdir(parents=True)

    completed, _ = run_script(
        "pack_all.sh",
        fake_conda,
        tmp_path,
        ["dlb-langflow"],
        DLB_CONDA_PACK=str(packer),
        DLB_ARTIFACT_DIR=str(artifact_dir),
    )

    assert completed.returncode != 0
    assert not (artifact_dir / "dlb-langflow.tar.gz" / "dlb-langflow.tar.gz").exists()

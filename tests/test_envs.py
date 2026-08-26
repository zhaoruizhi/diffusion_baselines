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


def conda_dependencies(environment):
    return [
        dependency
        for dependency in environment["dependencies"]
        if isinstance(dependency, str)
    ]


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


def test_conda_pytorch_environments_pin_mkl_before_ijit_regression():
    for name, environment in load_environments().items():
        dependencies = conda_dependencies(environment)
        if any(dependency.startswith("pytorch=") for dependency in dependencies):
            assert "mkl<2024.1" in dependencies, name
        assert not any(
            dependency.startswith("intel-openmp=2024.0") for dependency in dependencies
        )


def test_upstream_entrypoint_dependencies_are_explicit():
    environments = load_environments()

    assert "rich==14.2.0" in pip_dependencies(environments["flm"])
    assert "rich==13.7.1" in pip_dependencies(environments["duo"])
    assert "rich==13.7.1" in pip_dependencies(environments["mdlm"])
    assert "rich==14.2.0" in pip_dependencies(environments["candi"])


def test_project_runtime_dependencies_are_available_in_every_environment():
    for name, environment in load_environments().items():
        packages = pinned_pip_dependencies(environment)
        assert packages["pydantic"] == "2.12.4", name
        assert packages["pyyaml"] == "6.0.2", name


def test_langflow_environment_includes_conditional_prompt_data_dependencies():
    """Catch conditional runs failing before LangFlow loads because datasets is absent."""

    packages = pinned_pip_dependencies(load_environments()["langflow"])

    assert packages["datasets"] == "3.6.0"
    assert packages["fsspec"] == "2025.3.0"
    assert packages["pyarrow"] == "20.0.0"


def test_python39_pydantic_environments_include_annotation_backport():
    environments = load_environments()

    for name in {"mdlm", "rdlm"}:
        packages = pinned_pip_dependencies(environments[name])
        assert packages["eval-type-backport"] == "0.2.2"


def test_sdtt_family_includes_pkg_resources_provider_for_legacy_imports():
    environments = load_environments()

    for name in {"sdtt", "di4c"}:
        assert "setuptools==80.9.0" in pip_dependencies(environments[name])
        assert "nltk==3.9.1" in pip_dependencies(environments[name])


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


def test_mdlm_defers_source_only_cuda_extensions_until_torch_is_available():
    dependencies = pip_dependencies(load_environments()["mdlm"])

    assert not any(package.startswith("causal-conv1d==") for package in dependencies)
    assert not any(package.startswith("mamba-ssm==") for package in dependencies)


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
    failure_stderr = os.environ.get("FAKE_FAIL_STDERR")
    if failure_stderr:
        print(failure_stderr, file=sys.stderr)
    raise SystemExit(23)

if arguments[:1] == ["run"]:
    if arguments[3:5] == ["python", "-c"]:
        print("2.5.1 12.4")
        raise SystemExit(0)
    if arguments[3:6] == ["python", "-m", "pip"]:
        raise SystemExit(0)
    if os.environ.get("FAKE_DROP_RUN_STDIN") and arguments[4] == "-":
        sys.stdin.read()
        raise SystemExit(0)
    if arguments[4] == "-":
        probe_source = sys.stdin.read()
    else:
        probe_source = Path(arguments[4]).read_text()
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
        if os.environ.get("FAKE_UNHEALTHY_PROBE"):
            payload["imports"][modules[0]] = False
            payload["import_errors"] = {modules[0]: "simulated import failure"}
        print("DLB_ENV_PROBE_V1:" + json.dumps(payload, sort_keys=True))
        if (
            os.environ.get("FAKE_APPEND_CONDA_DIAGNOSTIC")
            and "raise SystemExit(0)" not in probe_source
        ):
            print("CondaError: child process exited with status 1", file=sys.stderr)
            raise SystemExit(23)
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
        "flash-attn==2.7.4.post1",
        "--no-build-isolation",
    ] in calls


def test_create_all_checks_torch_import_before_flash_attention(fake_conda, tmp_path):
    completed, calls = run_script("create_all.sh", fake_conda, tmp_path, ["dlb-duo"])

    assert completed.returncode == 0, completed.stderr
    torch_import_call = next(
        call for call in calls
        if call[:5] == ["run", "-n", "dlb-duo", "python", "-c"]
        and "import torch" in call[5]
    )
    flash_install_call = [
        "run",
        "-n",
        "dlb-duo",
        "python",
        "-m",
        "pip",
        "install",
        "flash-attn==2.7.4.post1",
        "--no-build-isolation",
    ]
    assert calls.index(torch_import_call) < calls.index(flash_install_call)


def test_create_all_installs_mdlm_cuda_extensions_from_git_after_torch(
    fake_conda, tmp_path
):
    completed, calls = run_script("create_all.sh", fake_conda, tmp_path, ["dlb-mdlm"])

    assert completed.returncode == 0, completed.stderr
    torch_import_call = next(
        call for call in calls
        if call[:5] == ["run", "-n", "dlb-mdlm", "python", "-c"]
        and "import torch" in call[5]
    )
    extension_install_call = [
        "run",
        "-n",
        "dlb-mdlm",
        "python",
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "git+https://github.com/Dao-AILab/causal-conv1d.git@v1.1.3.post1",
        "git+https://github.com/state-spaces/mamba.git@v1.1.4",
    ]
    assert calls.index(torch_import_call) < calls.index(extension_install_call)


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


def test_verify_all_debug_mode_reports_probe_manager_stderr(fake_conda, tmp_path):
    completed, _ = run_script(
        "verify_all.sh",
        fake_conda,
        tmp_path,
        ["dlb-duo"],
        DLB_VERIFY_DEBUG="1",
        FAKE_FAIL_SUBSTRING="dlb-duo",
        FAKE_FAIL_STDERR="simulated native loader failure",
    )

    records = [json.loads(line) for line in completed.stdout.splitlines()]
    assert completed.returncode != 0
    assert records[0]["error"] == "verification probe failed"
    assert "conda run exited with status 23" in records[0]["diagnostic"]
    assert "simulated native loader failure" in records[0]["diagnostic"]


def test_verify_all_does_not_depend_on_manager_forwarding_stdin(fake_conda, tmp_path):
    completed, calls = run_script(
        "verify_all.sh",
        fake_conda,
        tmp_path,
        ["dlb-duo"],
        FAKE_DROP_RUN_STDIN="1",
    )

    assert completed.returncode == 0, completed.stderr
    run_call = next(call for call in calls if call[:3] == ["run", "-n", "dlb-duo"])
    assert run_call[3] == "python"
    assert run_call[4] != "-"


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


@pytest.mark.parametrize(
    "payload",
    [
        '{"environment":"dlb-langflow","python":"3.11","torch":NaN,"torch_cuda":"12.4","cuda_available":true,"imports":{"einops":true,"huggingface_hub":true,"safetensors":true,"transformers":true}}',
        '{"environment":"dlb-langflow","python":"3.11","torch":"2.5.1","torch_cuda":Infinity,"cuda_available":true,"imports":{"einops":true,"huggingface_hub":true,"safetensors":true,"transformers":true}}',
        '{"environment":"dlb-langflow","python":"3.11","torch":"2.5.1","torch_cuda":"12.4","cuda_available":true,"imports":{"einops":true,"huggingface_hub":true,"safetensors":true,"transformers":true},"unexpected":true}',
        '{"environment":"dlb-langflow","python":"3.11","torch":"2.5.1","torch_cuda":"12.4","cuda_available":true,"imports":{"einops":true,"huggingface_hub":true,"safetensors":true,"transformers":true,"unexpected":true}}',
        '{"environment":"dlb-langflow","python":"3.11","torch":"2.5.1","torch_cuda":"12.4","cuda_available":1,"imports":{"einops":true,"huggingface_hub":true,"safetensors":true,"transformers":true}}',
        '{"environment":"dlb-langflow","python":"3.11","torch":"2.5.1","torch_cuda":"12.4","cuda_available":true,"imports":{"einops":true,"einops":false,"huggingface_hub":true,"safetensors":true,"transformers":true}}',
        '{"environment":"dlb-langflow","python":"3.11","torch":"2.5.1","torch_cuda":"12.4","cuda_available":true,"imports":{"einops":true,"huggingface_hub":true,"safetensors":true}}',
    ],
    ids=[
        "nan",
        "infinity",
        "unknown-top-level-field",
        "unknown-import-field",
        "bool-as-int",
        "duplicate-requested-import",
        "missing-requested-import",
    ],
)
def test_verify_all_rejects_invalid_probe_schema(fake_conda, tmp_path, payload):
    completed, _ = run_script(
        "verify_all.sh",
        fake_conda,
        tmp_path,
        ["dlb-langflow"],
        FAKE_PROBE_OUTPUT="DLB_ENV_PROBE_V1:" + payload,
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
    assert "NaN" not in completed.stdout
    assert "Infinity" not in completed.stdout


def test_verify_all_preserves_unhealthy_probe_when_manager_adds_diagnostic(
    fake_conda, tmp_path
):
    completed, _ = run_script(
        "verify_all.sh",
        fake_conda,
        tmp_path,
        ["dlb-langflow"],
        FAKE_UNHEALTHY_PROBE="1",
        FAKE_APPEND_CONDA_DIAGNOSTIC="1",
    )

    records = [json.loads(line) for line in completed.stdout.splitlines()]
    assert completed.returncode != 0
    assert records[0]["imports"]["datasets"] is False
    assert records[0]["import_errors"] == {"datasets": "simulated import failure"}
    assert "CondaError" not in completed.stdout
    assert "CondaError" not in completed.stderr


def test_verify_all_rejects_echo_manager_output(fake_conda, tmp_path):
    completed, _ = run_script(
        "verify_all.sh", fake_conda, tmp_path, ["dlb-langflow"], DLB_CONDA="/bin/echo"
    )

    records = [json.loads(line) for line in completed.stdout.splitlines()]
    assert completed.returncode != 0
    assert records[0]["error"] == "verification probe failed"


def test_verify_all_rejects_flash_attention_without_required_torch_triton_api(
    fake_conda, tmp_path
):
    payload = {
        "environment": "dlb-flm",
        "python": "3.11",
        "torch": "2.5.1",
        "torch_cuda": "12.4",
        "cuda_available": True,
        "imports": {
            module: True
            for module in [
                "datasets",
                "einops",
                "entmax",
                "flash_attn",
                "fsspec",
                "huggingface_hub",
                "hydra",
                "lightning",
                "numpy",
                "omegaconf",
                "pydantic",
                "requests",
                "rich",
                "scipy",
                "timm",
                "tokenizers",
                "torchmetrics",
                "tqdm",
                "transformers",
                "triton",
                "wandb",
                "yaml",
            ]
        },
        "runtime_checks": {"flash_attn_torch_wrap_triton": False},
        "runtime_errors": {
            "flash_attn_torch_wrap_triton": (
                "flash-attn 2.8.3 requires torch.library.wrap_triton"
            )
        },
    }
    completed, _ = run_script(
        "verify_all.sh",
        fake_conda,
        tmp_path,
        ["dlb-flm"],
        FAKE_PROBE_OUTPUT="DLB_ENV_PROBE_V1:" + json.dumps(payload, sort_keys=True),
    )

    records = [json.loads(line) for line in completed.stdout.splitlines()]
    assert completed.returncode != 0
    assert records[0]["runtime_checks"]["flash_attn_torch_wrap_triton"] is False
    assert "wrap_triton" in records[0]["runtime_errors"]["flash_attn_torch_wrap_triton"]


def test_verify_all_escapes_unknown_environment_names(fake_conda, tmp_path):
    completed, _ = run_script(
        "verify_all.sh", fake_conda, tmp_path, ['bad"name']
    )

    records = [json.loads(line) for line in completed.stdout.splitlines()]
    assert completed.returncode != 0
    assert records[0]["environment"] == 'bad"name'
    assert records[0]["error"] == "unknown environment"


def test_verify_all_checks_the_complete_method_import_mapping(fake_conda, tmp_path):
    project_runtime_imports = {"pydantic", "yaml"}
    expected_imports = {
        "dlb-flm": {
            "datasets", "einops", "entmax", "flash_attn", "fsspec", "huggingface_hub",
            "hydra", "lightning", "numpy", "omegaconf", "requests", "rich", "scipy",
            "timm", "tokenizers", "torchmetrics", "tqdm", "transformers", "triton", "wandb",
        },
        "dlb-langflow": {
            "datasets", "einops", "fsspec", "huggingface_hub", "pyarrow",
            "safetensors", "transformers",
        },
        "dlb-duo": {
            "datasets", "einops", "flash_attn", "fsspec", "h5py", "huggingface_hub",
            "hydra", "lightning", "numpy", "omegaconf", "requests", "rich", "scipy", "timm",
            "tokenizers", "torchmetrics", "torchvision", "tqdm", "transformers", "triton", "wandb",
        },
        "dlb-mdlm": {
            "causal_conv1d", "datasets", "einops", "eval_type_backport", "flash_attn",
            "fsspec", "huggingface_hub", "hydra", "lightning", "mamba_ssm", "numpy",
            "omegaconf", "requests", "rich", "timm", "tokenizers", "torchmetrics",
            "transformers", "wandb",
        },
        "dlb-candi": {
            "datasets", "einops", "evaluate", "flash_attn", "fsspec", "huggingface_hub", "hydra",
            "lightning", "numpy", "omegaconf", "requests", "rich", "scipy", "tokenizers", "torchmetrics",
            "tqdm", "transformers",
        },
        "dlb-rdlm": {
            "accelerate", "datasets", "einops", "eval_type_backport", "fsspec",
            "huggingface_hub", "hydra", "numpy", "omegaconf", "requests", "scipy",
            "tokenizers", "tqdm", "transformers", "wandb",
        },
        "dlb-sdtt": {
            "datasets", "einops", "flash_attn", "fsspec", "huggingface_hub", "hydra", "lightning",
            "loguru", "mauve", "nltk", "numpy", "omegaconf", "pandas", "requests", "safetensors",
            "tensorboard", "timm", "tokenizers", "torchdata", "tqdm", "transformers", "wandb",
        },
        "dlb-di4c": {
            "datasets", "einops", "flash_attn", "fsspec", "huggingface_hub", "hydra", "lightning",
            "loguru", "mauve", "nltk", "numpy", "omegaconf", "pandas", "requests", "safetensors",
            "tensorboard", "timm", "tokenizers", "torchdata", "tqdm", "transformers", "wandb",
        },
        "dlb-eval": {"accelerate", "datasets", "evaluate", "fsspec", "mauve", "sacrebleu", "scipy", "tokenizers", "transformers"},
    }
    expected_imports = {
        environment: modules | project_runtime_imports
        for environment, modules in expected_imports.items()
    }
    completed, calls = run_script(
        "verify_all.sh", fake_conda, tmp_path, list(expected_imports)
    )

    assert completed.returncode == 0, completed.stderr
    requested_imports = {}
    for call in calls:
        if call[:1] != ["run"]:
            continue
        assert call[:2] == ["run", "-n"]
        assert call[3] == "python"
        assert call[4] != "-"
        manager_environment = call[2]
        probe_environment = call[5]
        assert manager_environment == probe_environment
        assert manager_environment in expected_imports
        assert manager_environment not in requested_imports
        raw_modules = call[6:]
        assert len(raw_modules) == len(set(raw_modules))
        requested_imports[manager_environment] = set(raw_modules)

    assert set(requested_imports) == set(expected_imports)
    for environment, expected_modules in expected_imports.items():
        assert requested_imports[environment] == expected_modules


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

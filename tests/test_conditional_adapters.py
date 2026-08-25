from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from dlb.adapters.candi import CANDIAdapter
from dlb.adapters.di4c import Di4CAdapter
from dlb.adapters.duo import DuoAdapter
from dlb.adapters.flm import FLMAdapter
from dlb.adapters.langflow import LangFlowAdapter
from dlb.adapters.mdlm import MDLMAdapter
from dlb.adapters.rdlm import RDLMAdapter
from dlb.adapters.sdtt import SDTTAdapter
from dlb.runner import RunRequest


ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "data/manifests/conditional-lm1b-c64.json"


def conditional_request(model: str, dataset: str = "lm1b", steps: int = 8) -> RunRequest:
    step_count = 1000 if model == "rdlm" else steps
    return RunRequest(
        run_id=f"{model}-{dataset}-steps-{step_count}",
        model_id=model,
        dataset_id=dataset,
        step_count=step_count,
        seed=42,
        sample_count=2048,
        generation_mode="conditional_prefix",
        conditioning_manifest=str(MANIFEST),
        conditioning_manifest_sha256="a" * 64,
        conditioning_config_sha256="b" * 64,
        prefix_length=64,
        evaluation_continuation_length=64,
        prompt_count=1024,
        diversity_prompt_count=256,
        completions_per_diversity_prompt=5,
        completion_schedule="c0:p0-1023;c1-4:p0-255",
        results_root=str(ROOT / "results/conditional"),
    )


def run_dir(item: RunRequest) -> Path:
    return (
        ROOT
        / "results"
        / "conditional"
        / "samples"
        / item.dataset_id
        / item.model_id
        / f"steps_{item.step_count}"
    )


def option(command: list[str], key: str) -> str:
    index = command.index(key)
    return command[index + 1]


def wrapper_option(command: list[str], key: str) -> str:
    prefix = key + "="
    values = [argument[len(prefix) :] for argument in command if argument.startswith(prefix)]
    assert len(values) == 1, (key, command)
    return values[0]


@pytest.mark.parametrize(
    ("model", "adapter"),
    [
        ("flm", FLMAdapter()),
        ("fmlm", FLMAdapter()),
        ("duo", DuoAdapter()),
        ("duo_dcd", DuoAdapter()),
        ("mdlm", MDLMAdapter()),
        ("candi", CANDIAdapter()),
        ("langflow", LangFlowAdapter()),
        ("rdlm", RDLMAdapter()),
    ],
)
def test_teacher_command_serializes_complete_conditioning_contract(model: str, adapter) -> None:
    request = conditional_request(model)

    command = adapter.render_command(request, run_dir(request), dry_run=True)

    assert wrapper_option(command, "--generation-mode") == "conditional_prefix"
    assert wrapper_option(command, "--conditioning-manifest") == str(MANIFEST)
    assert wrapper_option(command, "--conditioning-manifest-sha256") == "a" * 64
    assert wrapper_option(command, "--prefix-length") == "64"
    assert wrapper_option(command, "--prompt-count") == "1024"
    assert wrapper_option(command, "--diversity-prompt-count") == "256"
    assert wrapper_option(command, "--completions-per-diversity-prompt") == "5"
    assert wrapper_option(command, "--completion-schedule") == "c0:p0-1023;c1-4:p0-255"


@pytest.mark.parametrize(
    ("model", "adapter"),
    [
        ("mdlm_sdtt", SDTTAdapter()),
        ("mdlm_di4c", Di4CAdapter("mdlm")),
        ("duo_di4c", Di4CAdapter("duo")),
    ],
)
def test_distilled_command_serializes_complete_conditioning_contract(model: str, adapter) -> None:
    request = conditional_request(model)

    command = adapter.render_command(request, run_dir(request), dry_run=True)

    assert option(command, "--generation-mode") == "conditional_prefix"
    assert option(command, "--conditioning-manifest") == str(MANIFEST)
    assert option(command, "--conditioning-manifest-sha256") == "a" * 64
    assert option(command, "--prefix-length") == "64"
    assert option(command, "--prompt-count") == "1024"
    assert option(command, "--diversity-prompt-count") == "256"
    assert option(command, "--completions-per-diversity-prompt") == "5"
    assert option(command, "--completion-schedule") == "c0:p0-1023;c1-4:p0-255"


def test_unconditional_command_does_not_emit_conditioning_flags() -> None:
    request = replace(conditional_request("mdlm"), generation_mode="unconditional")
    request = replace(
        request,
        conditioning_manifest=None,
        conditioning_manifest_sha256=None,
        conditioning_config_sha256=None,
        prefix_length=None,
        evaluation_continuation_length=None,
        prompt_count=None,
        diversity_prompt_count=None,
        completions_per_diversity_prompt=None,
        completion_schedule=None,
        results_root=None,
        sample_count=17,
    )

    command = MDLMAdapter().render_command(
        request,
        ROOT / "results" / "samples" / "lm1b" / "mdlm" / "steps_8",
        dry_run=True,
    )

    assert not any("conditioning" in argument or "generation-mode" in argument for argument in command)

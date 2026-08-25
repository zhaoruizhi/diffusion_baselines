"""Render validated teacher-adapter commands without invoking model code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dlb.adapters.base import AdapterError, BaseTeacherAdapter
from dlb.adapters.candi import CANDIAdapter
from dlb.adapters.di4c import Di4CAdapter
from dlb.adapters.duo import DuoAdapter
from dlb.adapters.flm import FLMAdapter
from dlb.adapters.langflow import LangFlowAdapter
from dlb.adapters.mdlm import MDLMAdapter
from dlb.adapters.rdlm import RDLMAdapter
from dlb.adapters.sdtt import SDTTAdapter
from dlb.registry import load_registry, step_grid_for_model
from dlb.runner import RunRequest


ADAPTERS: dict[str, BaseTeacherAdapter] = {
    "flm": FLMAdapter(),
    "fmlm": FLMAdapter(),
    "duo": DuoAdapter(),
    "duo_dcd": DuoAdapter(),
    "mdlm": MDLMAdapter(),
    "candi": CANDIAdapter(),
    "langflow": LangFlowAdapter(),
    "rdlm": RDLMAdapter(),
    "mdlm_sdtt": SDTTAdapter(),
    "mdlm_di4c": Di4CAdapter("mdlm"),
    "duo_di4c": Di4CAdapter("duo"),
}


def _csv(value: str) -> list[str]:
    values = value.split(",")
    if not values or any(not item for item in values):
        raise argparse.ArgumentTypeError("comma-separated values must not be empty")
    return values


def _record(
    root: Path,
    model_id: str,
    dataset_id: str,
    step_count: int,
    sample_count: int,
    seed: int,
    dry_run: bool,
) -> dict[str, object]:
    registry = load_registry(root / "configs" / "experiments.yaml")
    model = registry.models.get(model_id)
    support = model.datasets.get(dataset_id) if model is not None else None
    if model is None or support is None:
        return {
            "status": "unsupported",
            "model": model_id,
            "dataset": dataset_id,
            "reason": "unknown model or dataset",
        }
    if support.status != "supported":
        return {
            "status": "unsupported",
            "model": model_id,
            "dataset": dataset_id,
            "reason": support.reason,
        }
    allowed_steps = step_grid_for_model(registry, model_id)
    if step_count not in allowed_steps:
        joined_steps = ",".join(str(step) for step in allowed_steps)
        return {
            "status": "error",
            "model": model_id,
            "dataset": dataset_id,
            "reason": (
                f"invalid step count {step_count} for {model_id}/{dataset_id}; "
                f"allowed: {joined_steps}"
            ),
        }
    adapter = ADAPTERS.get(model_id)
    if adapter is None:
        return {
            "status": "unsupported",
            "model": model_id,
            "dataset": dataset_id,
            "reason": "no command adapter is implemented for this model",
        }
    request = RunRequest(
        run_id=f"{model_id}-{dataset_id}-steps-{step_count}",
        model_id=model_id,
        dataset_id=dataset_id,
        step_count=step_count,
        seed=seed,
        sample_count=sample_count,
    )
    run_dir = (
        root
        / "results"
        / "samples"
        / dataset_id
        / model_id
        / f"steps_{step_count}"
    )
    try:
        command = adapter.render_command(request, run_dir, dry_run=dry_run)
    except (AdapterError, OSError, ValueError) as error:
        return {
            "status": "error",
            "model": model_id,
            "dataset": dataset_id,
            "reason": str(error),
        }
    return {
        "status": "supported",
        "model": model_id,
        "dataset": dataset_id,
        "steps": step_count,
        "sample_count": sample_count,
        "command": command,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--models", type=_csv, required=True)
    parser.add_argument("--datasets", type=_csv, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    exit_code = 0
    for model_id in arguments.models:
        for dataset_id in arguments.datasets:
            record = _record(
                root,
                model_id,
                dataset_id,
                arguments.steps,
                arguments.num_samples,
                arguments.seed,
                arguments.dry_run,
            )
            if record["status"] == "error":
                exit_code = 1
            print(json.dumps(record, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

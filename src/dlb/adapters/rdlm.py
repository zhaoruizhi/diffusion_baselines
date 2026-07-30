"""Adapter for the pinned official RDLM LM1B SDE sampler."""

from pathlib import Path
import sys
from typing import Iterable

from dlb.adapters.base import AdapterError, BaseTeacherAdapter, CheckpointSelection
from dlb.runner import RunRequest
from dlb.schema import SampleRecord


class RDLMAdapter(BaseTeacherAdapter):
    identity = "dlb.adapters.rdlm:v1"
    upstream = "rdlm"
    supported_models = frozenset({"rdlm"})
    teacher_families = {"rdlm": "continuous_rdlm"}
    batch_sizes = {("rdlm", "lm1b"): 8}
    _LM1B_FILES = (
        "LM1B/checkpoint.pth",
        "LM1B/config.yaml",
        "LM1B/sde.pkl",
    )

    def _asset_paths(
        self, checkpoint: CheckpointSelection
    ) -> tuple[Path, Path, Path]:
        missing = [name for name in self._LM1B_FILES if name not in checkpoint.required_files]
        if missing:
            raise AdapterError(
                "canonical RDLM resource does not bind the saved LM1B trio: "
                + ", ".join(missing)
            )
        asset_root = checkpoint.path / "LM1B"
        return (
            asset_root / "checkpoint.pth",
            asset_root / "config.yaml",
            asset_root / "sde.pkl",
        )

    def render_command(
        self, request: RunRequest, run_dir: Path, *, dry_run: bool
    ) -> list[str]:
        root, length, _ = self._validate_request(request, run_dir)
        if length != 128:
            raise AdapterError(f"RDLM LM1B requires sequence length 128, found {length}")
        checkpoint = self._resolve_checkpoint(root, request, dry_run=dry_run)
        checkpoint_path, saved_config, saved_sde = self._asset_paths(checkpoint)
        entrypoint = root / "upstreams" / "rdlm" / "main.py"
        if not entrypoint.is_file() or entrypoint.is_symlink():
            raise AdapterError(f"pinned upstream entrypoint is missing or unsafe: {entrypoint}")
        capture_path = run_dir.resolve() / "upstream_token_ids.json"
        hydra_dir = run_dir.resolve() / "upstream_hydra"
        tokenizer_name = self._load_data_contract(root, request.dataset_id)["tokenizer"]
        tokenizer_revision = self._tokenizer_revision(root, tokenizer_name)
        tokenizer_snapshot = (
            root
            / "data"
            / "raw"
            / "huggingface"
            / "hub"
            / f"models--{tokenizer_name}"
            / "snapshots"
            / tokenizer_revision
        )
        arguments = [
            sys.executable,
            "-B",
            "-u",
            "-m",
            "dlb.adapters.capture",
            f"--upstream-entrypoint={entrypoint}",
            f"--capture-path={capture_path}",
            "--capture-kind=rdlm",
            f"--saved-config-path={saved_config}",
            f"--saved-sde-path={saved_sde}",
            f"--expected-samples={request.sample_count}",
            f"--data-config-path={root / 'artifacts' / 'data.yaml'}",
            f"--downloads-manifest-path={root / 'data' / 'manifests' / 'downloads.json'}",
            f"--dataset-id={request.dataset_id}",
            f"--tokenizer-snapshot={tokenizer_snapshot}",
            "--",
            "run_mode=sample",
            "server=sample",
            "exp=sample_lm1b",
            "ngpus=1",
            "use_wandb=False",
            f"model_path={checkpoint_path}",
            f"seed={request.seed}",
            "sampling.predictor=grw",
            f"sampling.steps={request.step_count}",
            f"sampling.batch_per_gpu={min(8, request.sample_count)}",
            "eval.entropy=False",
            "eval.nll=False",
            "eval.gen_ppl=False",
            f"hydra.run.dir={hydra_dir}",
        ]
        self._validate_argv(arguments)
        return arguments

    def convert_outputs(
        self, request: RunRequest, run_dir: Path
    ) -> Iterable[SampleRecord]:
        return self._convert_capture_outputs(
            request, run_dir, retokenize_inexact_rows=True
        )


adapter = RDLMAdapter()

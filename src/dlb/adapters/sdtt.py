"""Adapter for the pinned SDTT KLD round-7 language student."""

from pathlib import Path
import sys
from typing import Iterable

from dlb.adapters.base import AdapterError, BaseTeacherAdapter
from dlb.runner import RunRequest
from dlb.schema import SampleRecord


class SDTTAdapter(BaseTeacherAdapter):
    identity = "dlb.adapters.sdtt:v1"
    upstream = "sdtt"
    supported_models = frozenset({"mdlm_sdtt"})
    teacher_families = {"mdlm_sdtt": "masked_mdlm"}
    batch_sizes = {
        ("mdlm_sdtt", "lm1b"): 16,
        ("mdlm_sdtt", "owt"): 16,
    }

    def render_command(
        self, request: RunRequest, run_dir: Path, *, dry_run: bool
    ) -> list[str]:
        root, length, batch_size = self._validate_request(request, run_dir)
        checkpoint = self._resolve_checkpoint(root, request, dry_run=dry_run)
        upstream_root = root / "upstreams" / "sdtt"
        wrapper = root / "adapters" / "sample_sdtt.py"
        if not (upstream_root / "src/sdtt/main.py").is_file() or upstream_root.is_symlink():
            raise AdapterError(f"pinned SDTT source is missing or unsafe: {upstream_root}")
        if not wrapper.is_file() or wrapper.is_symlink():
            raise AdapterError(f"SDTT sampling wrapper is missing or unsafe: {wrapper}")
        if checkpoint.path.suffix:
            checkpoint_path = checkpoint.path
            config_path = checkpoint.config_path
            if config_path is None:
                raise AdapterError("SDTT recipe does not bind a sampling config")
            if not dry_run and (
                config_path.is_symlink()
                or not config_path.is_file()
                or config_path.stat().st_size <= 0
            ):
                raise AdapterError(
                    f"SDTT recipe sampling config is missing or unsafe: {config_path}"
                )
        else:
            checkpoint_path = checkpoint.path / "model.safetensors"
            config_path = checkpoint.path / "config.json"
        tokenizer_snapshot = self._tokenizer_snapshot(root, request.dataset_id)
        arguments = [
            sys.executable,
            "-B",
            "-u",
            str(wrapper),
            "--upstream-root",
            str(upstream_root),
            "--checkpoint",
            str(checkpoint_path),
            "--config",
            str(config_path),
            "--tokenizer-snapshot",
            str(tokenizer_snapshot),
            "--output",
            str(run_dir.resolve() / "upstream_token_ids.json"),
            "--sample-count",
            str(request.sample_count),
            "--batch-size",
            str(min(batch_size, request.sample_count)),
            "--num-steps",
            str(request.step_count),
            "--seq-len",
            str(length),
            "--seed",
            str(request.seed),
            "--sampler",
            "ancestral",
            "--loss",
            "kld",
            "--round",
            "7",
            "--teacher-family",
            checkpoint.teacher_family,
            "--offline",
            "true",
        ]
        self._validate_argv(arguments)
        return arguments

    def _tokenizer_snapshot(self, root: Path, dataset_id: str) -> Path:
        tokenizer = self._load_data_contract(root, dataset_id)["tokenizer"]
        revision = self._tokenizer_revision(root, tokenizer)
        return (
            root
            / "data/raw/huggingface/hub"
            / f"models--{tokenizer}"
            / "snapshots"
            / revision
        )

    def convert_outputs(
        self, request: RunRequest, run_dir: Path
    ) -> Iterable[SampleRecord]:
        return self._convert_capture_outputs(request, run_dir)


adapter = SDTTAdapter()

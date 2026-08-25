"""Adapter for the pinned SDTT KLD round-7 language student."""

from pathlib import Path
import sys
from typing import Iterable

from dlb.adapters.base import AdapterError, BaseTeacherAdapter
from dlb.runner import RunRequest
from dlb.schema import SampleRecord


class SDTTAdapter(BaseTeacherAdapter):
    identity = "dlb.adapters.sdtt:v3"
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
        checkpoint_path = (
            checkpoint.path
            if checkpoint.path.suffix
            else checkpoint.path / "model.safetensors"
        )
        config_path = checkpoint.config_path
        if config_path is None:
            raise AdapterError("SDTT checkpoint selection does not bind a sampling config")
        checkpoint_sha256, config_sha256 = self._runtime_asset_digests(
            root,
            request,
            checkpoint_path,
            config_path,
            dry_run=dry_run,
        )
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
            "--checkpoint-sha256",
            checkpoint_sha256,
            "--config-sha256",
            config_sha256,
            "--data-config",
            str(root / "artifacts/data.yaml"),
            "--downloads-manifest",
            str(root / "data/manifests/downloads.json"),
            "--dataset",
            request.dataset_id,
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
            *self._conditional_script_flags(request),
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

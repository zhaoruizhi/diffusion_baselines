"""Adapters for masked-MDLM and uniform-Duo Di4C students."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

from dlb.adapters.base import AdapterError, BaseTeacherAdapter
from dlb.runner import RunRequest
from dlb.schema import SampleRecord


class Di4CAdapter(BaseTeacherAdapter):
    identity = "dlb.adapters.di4c:v3"
    upstream = "di4c"
    teacher_families = {
        "mdlm_di4c": "masked_mdlm",
        "duo_di4c": "uniform_duo",
    }
    batch_sizes = {
        ("mdlm_di4c", "lm1b"): 16,
        ("mdlm_di4c", "owt"): 16,
        ("duo_di4c", "lm1b"): 16,
        ("duo_di4c", "owt"): 16,
    }

    def __init__(self, teacher: str | None = None) -> None:
        if teacher not in {None, "mdlm", "duo"}:
            raise AdapterError(f"unknown Di4C teacher identity: {teacher!r}")
        self.supported_models = frozenset(
            {"mdlm_di4c", "duo_di4c"} if teacher is None else {f"{teacher}_di4c"}
        )

    def render_command(
        self, request: RunRequest, run_dir: Path, *, dry_run: bool
    ) -> list[str]:
        root, length, batch_size = self._validate_request(request, run_dir)
        checkpoint = self._resolve_checkpoint(root, request, dry_run=dry_run)
        expected_family = self.teacher_families[request.model_id]
        if checkpoint.teacher_family != expected_family:
            raise AdapterError(
                f"Di4C checkpoint teacher family {checkpoint.teacher_family!r} "
                f"does not match {expected_family!r}"
            )
        upstream_root = root / "upstreams" / "di4c" / "sdtt"
        wrapper = root / "adapters" / "sample_di4c.py"
        if not (upstream_root / "src/sdtt/main.py").is_file() or upstream_root.is_symlink():
            raise AdapterError(f"pinned Di4C language source is missing or unsafe: {upstream_root}")
        if not wrapper.is_file() or wrapper.is_symlink():
            raise AdapterError(f"Di4C sampling wrapper is missing or unsafe: {wrapper}")
        checkpoint_path = (
            checkpoint.path / "sdtt7-di4c2.ckpt"
            if not checkpoint.path.suffix
            else checkpoint.path
        )
        config_path = checkpoint.config_path
        if config_path is None:
            raise AdapterError("Di4C checkpoint selection does not bind a sampling config")
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
            "--teacher-family",
            checkpoint.teacher_family,
            "--dataset",
            request.dataset_id,
            *self._conditional_script_flags(request),
            "--offline",
            "true",
        ]
        arguments.extend(["--allow-missing-embedded-config", "true"])
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


adapter = Di4CAdapter()

"""Adapter for the pinned OpenWebText LangFlow inference release."""

from pathlib import Path
import sys
from typing import Iterable

from dlb.adapters.base import AdapterError, BaseTeacherAdapter
from dlb.runner import RunRequest
from dlb.schema import SampleRecord


class LangFlowAdapter(BaseTeacherAdapter):
    identity = "dlb.adapters.langflow:v1"
    upstream = "langflow"
    supported_models = frozenset({"langflow"})
    teacher_families = {"langflow": "continuous_langflow"}
    batch_sizes = {("langflow", "owt"): 1}

    def render_command(
        self, request: RunRequest, run_dir: Path, *, dry_run: bool
    ) -> list[str]:
        root, length, _ = self._validate_request(request, run_dir)
        checkpoint = self._resolve_checkpoint(root, request, dry_run=dry_run)
        entrypoint = root / "upstreams" / "langflow" / "inference.py"
        if not entrypoint.is_file() or entrypoint.is_symlink():
            raise AdapterError(f"pinned upstream entrypoint is missing or unsafe: {entrypoint}")
        checkpoint_path = checkpoint.path / "model.safetensors"
        output_path = run_dir.resolve() / "upstream_samples.txt"
        capture_path = run_dir.resolve() / "upstream_token_ids.json"
        arguments = [
            sys.executable,
            "-B",
            "-u",
            "-m",
            "dlb.adapters.capture",
            f"--upstream-entrypoint={entrypoint}",
            f"--capture-path={capture_path}",
            "--capture-kind=langflow",
            f"--expected-samples={request.sample_count}",
            "--",
            "--checkpoint",
            str(checkpoint_path),
            "--num_samples",
            str(request.sample_count),
            "--batch_size",
            "1",
            "--num_steps",
            str(request.step_count),
            "--seq_length",
            str(length),
            "--seed",
            str(request.seed),
            "--output",
            str(output_path),
        ]
        self._validate_argv(arguments)
        return arguments

    def convert_outputs(
        self, request: RunRequest, run_dir: Path
    ) -> Iterable[SampleRecord]:
        return self._convert_capture_outputs(request, run_dir)


adapter = LangFlowAdapter()

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
        root, _, _ = self._validate_request(request, run_dir)
        if request.step_count != 1024:
            raise AdapterError(
                "pinned LangFlow inference.py does not expose variable sampling steps; "
                "only the official 1024-step inference contract can be labeled faithfully"
            )
        checkpoint = self._resolve_checkpoint(root, request, dry_run=dry_run)
        entrypoint = root / "upstreams" / "langflow" / "inference.py"
        if not entrypoint.is_file() or entrypoint.is_symlink():
            raise AdapterError(f"pinned upstream entrypoint is missing or unsafe: {entrypoint}")
        checkpoint_path = checkpoint.path / "model.safetensors"
        capture_path = run_dir.resolve() / "upstream_token_ids.json"
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
            "--capture-kind=langflow",
            f"--expected-samples={request.sample_count}",
            f"--data-config-path={root / 'artifacts' / 'data.yaml'}",
            f"--downloads-manifest-path={root / 'data' / 'manifests' / 'downloads.json'}",
            f"--dataset-id={request.dataset_id}",
            f"--tokenizer-snapshot={tokenizer_snapshot}",
            "--",
            "--checkpoint",
            str(checkpoint_path),
            "--num_samples",
            str(request.sample_count),
        ]
        self._validate_argv(arguments)
        return arguments

    def convert_outputs(
        self, request: RunRequest, run_dir: Path
    ) -> Iterable[SampleRecord]:
        return self._convert_capture_outputs(request, run_dir)


adapter = LangFlowAdapter()

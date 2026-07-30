"""Shared validation, checkpoint selection, and output conversion for teachers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import ClassVar, Iterable

import yaml

from dlb.checkpoints import load_checkpoint_manifest
from dlb.io import atomic_json_write
from dlb.registry import load_registry
from dlb.runner import RunRequest
from dlb.schema import SampleRecord


class AdapterError(ValueError):
    """A teacher request or upstream artifact cannot satisfy the canonical contract."""


@dataclass(frozen=True)
class CheckpointSelection:
    path: Path
    teacher_family: str
    source: str
    required_files: tuple[str, ...] = ()


class BaseTeacherAdapter:
    """Strict adapter base used by the FLM-lineage and discrete teachers."""

    identity: ClassVar[str]
    upstream: ClassVar[str]
    supported_models: ClassVar[frozenset[str]]
    teacher_families: ClassVar[dict[str, str]]
    batch_sizes: ClassVar[dict[tuple[str, str], int]]

    _DATA_CONFIGS: ClassVar[dict[str, str]] = {
        "lm1b": "lm1b-wrap",
        "owt": "openwebtext-split",
    }
    _TOKENIZER_VOCAB_SIZES: ClassVar[dict[str, int]] = {
        "bert-base-uncased": 30_522,
        "gpt2": 50_257,
    }
    _CHECKPOINT_SUFFIXES: ClassVar[frozenset[str]] = frozenset(
        {".ckpt", ".pt", ".pth", ".bin", ".safetensors"}
    )

    def build_command(self, request: RunRequest, run_dir: Path) -> list[str]:
        """Build a real argv, rejecting absent canonical checkpoint bytes."""

        return self.render_command(request, run_dir, dry_run=False)

    def render_command(
        self, request: RunRequest, run_dir: Path, *, dry_run: bool
    ) -> list[str]:
        root, length, batch_size = self._validate_request(request, run_dir)
        checkpoint = self._resolve_checkpoint(root, request, dry_run=dry_run)
        entrypoint = root / "upstreams" / self.upstream / "main.py"
        if not entrypoint.is_file() or entrypoint.is_symlink():
            raise AdapterError(f"pinned upstream entrypoint is missing or unsafe: {entrypoint}")

        generated_count = math.ceil(request.sample_count / batch_size) * batch_size
        batches = generated_count // batch_size
        capture_path = run_dir.resolve() / "upstream_token_ids.json"
        upstream_output = run_dir.resolve() / "upstream_samples.json"
        hydra_dir = run_dir.resolve() / "upstream_hydra"
        arguments = [
            sys.executable,
            "-u",
            "-m",
            "dlb.adapters.capture",
            f"--upstream-entrypoint={entrypoint}",
            f"--capture-path={capture_path}",
            "--",
            "mode=sample_eval",
            f"seed={request.seed}",
            f"data={self._data_config(request)}",
            "model=small",
            f"model.length={length}",
            f"loader.batch_size={self._loader_batch_size(request)}",
            f"loader.eval_batch_size={batch_size}",
            f"sampling.num_sample_batches={batches}",
            f"sampling.steps={request.step_count}",
            f"eval.checkpoint_path={checkpoint.path}",
            "trainer.devices=1",
            f"hydra.run.dir={hydra_dir}",
            f"checkpointing.save_dir={hydra_dir}",
            "+wandb.offline=true",
        ]
        if self.upstream != "mdlm":
            arguments.append(f"eval.generated_samples_path={upstream_output}")
        arguments.extend(self._sampling_overrides(request, checkpoint))
        self._validate_argv(arguments)
        return arguments

    def convert_outputs(
        self, request: RunRequest, run_dir: Path
    ) -> Iterable[SampleRecord]:
        root, _, batch_size = self._validate_conversion_request(request, run_dir)
        expected_generated = math.ceil(request.sample_count / batch_size) * batch_size
        capture_path = run_dir.resolve() / "upstream_token_ids.json"
        capture = self._read_capture(capture_path, expected_generated)

        if self.upstream == "mdlm":
            texts = [item["text"] for item in capture]
            token_ids = [item["token_ids"] for item in capture]
            output_format = "dlb-upstream-token-capture-v1"
            token_ids_source = "upstream"
            upstream_path = capture_path
        else:
            upstream_path = run_dir.resolve() / "upstream_samples.json"
            texts = self._read_standard_output(upstream_path, expected_generated)
            self._validate_texts(texts)
            output_format = "upstream-generated-seqs-v1"
            if capture:
                captured_texts = [item["text"] for item in capture]
                if captured_texts != texts:
                    raise AdapterError("captured token batches do not match upstream generated_seqs")
                token_ids = [item["token_ids"] for item in capture]
                token_ids_source = "upstream"
            else:
                token_ids = self._retokenize(root, request.dataset_id, texts)
                token_ids_source = "retokenized"

        dataset = self._load_data_contract(root, request.dataset_id)
        tokenizer_name = dataset["tokenizer"]
        vocab_size = self._TOKENIZER_VOCAB_SIZES[tokenizer_name]
        self._validate_samples(texts, token_ids, expected_generated, vocab_size)

        records = [
            SampleRecord(
                sample_id=index,
                text=texts[index],
                token_ids=token_ids[index],
                seed=request.seed,
                generation_seconds=0.0,
            )
            for index in range(request.sample_count)
        ]
        atomic_json_write(
            run_dir.resolve() / "conversion_metadata.json",
            {
                "format": output_format,
                "upstream_output": str(upstream_path),
                "generated_samples": expected_generated,
                "requested_samples": request.sample_count,
                "trimmed_samples": expected_generated - request.sample_count,
                "trim_policy": "stable_prefix",
                "token_ids_source": token_ids_source,
                "tokenizer": tokenizer_name,
                "tokenizer_revision": self._tokenizer_revision(root, tokenizer_name),
                "generation_seconds_source": "unavailable_excluded_sentinel",
                "generation_seconds_sentinel": 0.0,
                "exclude_from_latency": True,
            },
        )
        return records

    def _validate_request(
        self, request: RunRequest, run_dir: Path
    ) -> tuple[Path, int, int]:
        root, length, batch_size = self._validate_conversion_request(request, run_dir)
        registry_path = root / "configs" / "experiments.yaml"
        if not registry_path.is_file():
            raise AdapterError(f"canonical registry is missing: {registry_path}")
        registry = load_registry(registry_path)
        try:
            model = registry.models[request.model_id]
            support = model.datasets[request.dataset_id]
        except KeyError as error:
            raise AdapterError(
                f"unknown model/dataset cell: {request.model_id}/{request.dataset_id}"
            ) from error
        if support.status != "supported":
            raise AdapterError(
                f"unsupported model/dataset cell: {request.model_id}/{request.dataset_id}"
            )
        if request.step_count not in registry.step_grids[model.category]:
            raise AdapterError(
                f"invalid step count {request.step_count} for {model.category} category"
            )
        if model.adapter != self.upstream:
            raise AdapterError(
                f"registry adapter {model.adapter!r} does not match {self.upstream!r}"
            )
        return root, length, batch_size

    def _validate_conversion_request(
        self, request: RunRequest, run_dir: Path
    ) -> tuple[Path, int, int]:
        if request.model_id not in self.supported_models:
            raise AdapterError(
                f"{type(self).__name__} does not support model {request.model_id!r}"
            )
        if request.dataset_id not in self._DATA_CONFIGS:
            raise AdapterError(f"unsupported dataset: {request.dataset_id!r}")
        for name, value, positive in (
            ("step_count", request.step_count, True),
            ("sample_count", request.sample_count, True),
            ("seed", request.seed, False),
        ):
            if type(value) is not int or (positive and value <= 0):
                qualifier = "a positive integer" if positive else "an integer"
                raise AdapterError(f"{name} must be {qualifier}")
        run_dir = run_dir.resolve()
        suffix = (
            Path("results")
            / "samples"
            / request.dataset_id
            / request.model_id
            / f"steps_{request.step_count}"
        )
        if len(run_dir.parents) < 5:
            raise AdapterError(f"run directory is not rooted at a request root: {run_dir}")
        root = run_dir.parents[4]
        if root / suffix != run_dir:
            raise AdapterError(f"run directory does not match the request: {run_dir}")
        dataset = self._load_data_contract(root, request.dataset_id)
        length = dataset.get("sequence_length")
        if type(length) is not int or length <= 0:
            raise AdapterError(f"invalid canonical sequence length for {request.dataset_id}")
        batch_size = self.batch_sizes.get((request.model_id, request.dataset_id))
        if type(batch_size) is not int or batch_size <= 0:
            raise AdapterError(
                f"missing eval batch size for {request.model_id}/{request.dataset_id}"
            )
        return root, length, batch_size

    def _resolve_checkpoint(
        self, root: Path, request: RunRequest, *, dry_run: bool
    ) -> CheckpointSelection:
        manifest_path = root / "artifacts" / "checkpoints.yaml"
        if not manifest_path.is_file():
            raise AdapterError(f"canonical checkpoint manifest is missing: {manifest_path}")
        manifest = load_checkpoint_manifest(manifest_path)
        registry = load_registry(root / "configs" / "experiments.yaml")
        support = registry.models[request.model_id].datasets[request.dataset_id]
        expected_family = self.teacher_families[request.model_id]
        coverage = manifest.coverage.get((request.model_id, request.dataset_id))
        if coverage is not None:
            resource = manifest.resources[coverage.resource]
            family = coverage.teacher_family or resource.teacher_family
            if family != expected_family:
                raise AdapterError(
                    f"checkpoint teacher family {family!r} does not match {expected_family!r}"
                )
            base = root / "checkpoints" / resource.destination
            path = base / coverage.path if coverage.path else base
            selection = CheckpointSelection(
                path=path.resolve(),
                teacher_family=family,
                source=f"resource:{coverage.resource}",
                required_files=tuple(resource.required_files),
            )
            if not dry_run:
                self._require_resource_checkpoint(selection, base, coverage.path)
            return selection
        recipe_id = support.train_recipe
        if recipe_id is None or recipe_id not in manifest.recipes:
            raise AdapterError(
                f"canonical checkpoint selection is missing for {request.model_id}/{request.dataset_id}"
            )
        recipe = manifest.recipes[recipe_id]
        if (recipe.model, recipe.dataset) != (request.model_id, request.dataset_id):
            raise AdapterError(f"checkpoint recipe {recipe_id} describes the wrong cell")
        if recipe.teacher_family != expected_family:
            raise AdapterError(
                f"checkpoint teacher family {recipe.teacher_family!r} does not match {expected_family!r}"
            )
        output = (root / recipe.output).resolve()
        path = output if dry_run else self._select_recipe_checkpoint(output)
        return CheckpointSelection(
            path=path,
            teacher_family=recipe.teacher_family,
            source=f"recipe:{recipe_id}",
        )

    def _require_resource_checkpoint(
        self, selection: CheckpointSelection, base: Path, selected_name: str | None
    ) -> None:
        path = selection.path
        if selected_name is not None:
            if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
                raise AdapterError(f"required checkpoint file is missing or unsafe: {path}")
            return
        if path.is_symlink() or not path.is_dir():
            raise AdapterError(f"required checkpoint directory is missing or unsafe: {path}")
        for relative in selection.required_files:
            required = base / relative
            if required.is_symlink() or not required.is_file() or required.stat().st_size <= 0:
                raise AdapterError(f"required checkpoint file is missing or unsafe: {required}")

    def _select_recipe_checkpoint(self, output: Path) -> Path:
        if output.is_symlink() or not output.exists():
            raise AdapterError(f"recipe checkpoint output is missing or unsafe: {output}")
        if output.is_file():
            if output.stat().st_size <= 0:
                raise AdapterError(f"recipe checkpoint output is empty: {output}")
            return output
        candidates = sorted(
            path
            for path in output.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and path.suffix.lower() in self._CHECKPOINT_SUFFIXES
            and path.stat().st_size > 0
        )
        preferred = [path for path in candidates if path.name == "last.ckpt"]
        if len(preferred) == 1:
            return preferred[0]
        if len(candidates) != 1:
            raise AdapterError(
                f"recipe output must contain exactly one selectable checkpoint, found {len(candidates)}"
            )
        return candidates[0]

    def _read_standard_output(self, path: Path, expected: int) -> list[str]:
        value = self._read_json(path, "upstream sample output")
        if not isinstance(value, dict) or set(value) != {
            "generative_ppl",
            "entropy",
            "generated_seqs",
        }:
            raise AdapterError("unexpected upstream output format")
        texts = value["generated_seqs"]
        if not isinstance(texts, list):
            raise AdapterError("unexpected upstream generated_seqs format")
        if len(texts) != expected:
            raise AdapterError(f"expected {expected} generated samples, found {len(texts)}")
        return texts

    def _read_capture(self, path: Path, expected: int) -> list[dict[str, object]]:
        if not path.exists() and self.upstream != "mdlm":
            return []
        value = self._read_json(path, "token capture")
        if (
            not isinstance(value, dict)
            or set(value) != {"schema", "samples"}
            or value.get("schema") != "dlb-upstream-token-capture-v1"
            or not isinstance(value.get("samples"), list)
        ):
            raise AdapterError("unexpected capture format")
        samples = value["samples"]
        if len(samples) != expected:
            raise AdapterError(f"expected {expected} generated samples, found {len(samples)}")
        seen: set[int] = set()
        normalized: list[dict[str, object]] = []
        for index, item in enumerate(samples):
            if not isinstance(item, dict) or set(item) != {"sample_id", "text", "token_ids"}:
                raise AdapterError(f"unexpected capture record format at index {index}")
            sample_id = item["sample_id"]
            if type(sample_id) is not int:
                raise AdapterError(f"invalid sample_id at index {index}")
            if sample_id in seen:
                raise AdapterError(f"duplicate sample_id {sample_id}")
            seen.add(sample_id)
            if sample_id != index:
                raise AdapterError(f"expected sample_id {index}, found {sample_id}")
            normalized.append(item)
        return normalized

    def _validate_samples(
        self, texts: list[object], token_ids: list[object], expected: int, vocab_size: int
    ) -> None:
        if len(texts) != expected or len(token_ids) != expected:
            raise AdapterError(
                f"expected {expected} generated samples, found {min(len(texts), len(token_ids))}"
            )
        self._validate_texts(texts)
        for index, (_, tokens) in enumerate(zip(texts, token_ids, strict=True)):
            if not isinstance(tokens, list) or not tokens:
                raise AdapterError(f"empty token_ids at sample {index}")
            for token in tokens:
                if type(token) is not int or token < 0 or token >= vocab_size:
                    raise AdapterError(f"invalid token at sample {index}: {token!r}")

    @staticmethod
    def _validate_texts(texts: list[object]) -> None:
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise AdapterError(f"empty text at sample {index}")

    def _retokenize(
        self, root: Path, dataset_id: str, texts: list[str]
    ) -> list[list[int]]:
        contract = self._load_data_contract(root, dataset_id)
        tokenizer_name = contract["tokenizer"]
        revision = self._tokenizer_revision(root, tokenizer_name)
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise AdapterError(
                "transformers is required to retokenize upstream text without captured IDs"
            ) from error
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, revision=revision, local_files_only=True
            )
        except Exception as error:
            raise AdapterError(
                f"pinned tokenizer {tokenizer_name}@{revision} is not available locally"
            ) from error
        return [
            list(tokenizer.encode(text, add_special_tokens=False))
            for text in texts
        ]

    def _load_data_contract(self, root: Path, dataset_id: str) -> dict[str, object]:
        path = root / "artifacts" / "data.yaml"
        if not path.is_file():
            raise AdapterError(f"canonical data manifest is missing: {path}")
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        try:
            dataset = document["datasets"][dataset_id]
        except (KeyError, TypeError) as error:
            raise AdapterError(f"canonical data contract is missing for {dataset_id}") from error
        if not isinstance(dataset, dict):
            raise AdapterError(f"invalid canonical data contract for {dataset_id}")
        tokenizer = dataset.get("tokenizer")
        if tokenizer not in self._TOKENIZER_VOCAB_SIZES:
            raise AdapterError(f"unsupported canonical tokenizer: {tokenizer!r}")
        return dataset

    def _tokenizer_revision(self, root: Path, tokenizer_name: str) -> str:
        document = yaml.safe_load(
            (root / "artifacts" / "data.yaml").read_text(encoding="utf-8")
        )
        try:
            revision = document["models"][tokenizer_name]
        except (KeyError, TypeError) as error:
            raise AdapterError(f"missing pinned tokenizer revision for {tokenizer_name}") from error
        if not isinstance(revision, str) or len(revision) != 40:
            raise AdapterError(f"invalid pinned tokenizer revision for {tokenizer_name}")
        return revision

    @staticmethod
    def _read_json(path: Path, label: str) -> object:
        if path.is_symlink() or not path.is_file():
            raise AdapterError(f"{label} is missing or unsafe: {path}")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AdapterError(f"{label} is invalid JSON: {path}") from error

    @staticmethod
    def _validate_argv(arguments: list[str]) -> None:
        if not arguments:
            raise AdapterError("adapter command is empty")
        for argument in arguments:
            if not isinstance(argument, str) or not argument:
                raise AdapterError("adapter command contains an empty argument")
            if "${" in argument or "{" in argument or "}" in argument or "None" in argument:
                raise AdapterError(f"adapter command contains an unresolved argument: {argument}")
        for argument in arguments:
            if argument.startswith("eval.checkpoint_path=") and not argument.split("=", 1)[1]:
                raise AdapterError("checkpoint path must not be empty")

    def _loader_batch_size(self, request: RunRequest) -> int:
        del request
        return 2

    def _data_config(self, request: RunRequest) -> str:
        return self._DATA_CONFIGS[request.dataset_id]

    def _sampling_overrides(
        self, request: RunRequest, checkpoint: CheckpointSelection
    ) -> list[str]:
        raise NotImplementedError

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
from dlb.io import atomic_json_write, sha256_file
from dlb.registry import load_registry
from dlb.runner import RunRequest, _resolve_checkpoint_provenance
from dlb.schema import SampleRecord


class AdapterError(ValueError):
    """A teacher request or upstream artifact cannot satisfy the canonical contract."""


@dataclass(frozen=True)
class CheckpointSelection:
    path: Path
    teacher_family: str
    source: str
    required_files: tuple[str, ...] = ()
    config_path: Path | None = None


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

    def benchmark_hook(self, request: RunRequest) -> str:
        """Name the concrete loaded-model sampling boundary used on the server."""

        del request
        return {
            "mdlm": "mdlm._sample",
            "langflow": "langflow.generate_samples",
            "rdlm": "rdlm.sampling_fn",
            "sdtt": "distilled.model.sample",
            "di4c": "distilled.model.sample",
        }.get(self.upstream, "teacher.generate_samples")

    def author_precision_policy(self, request: RunRequest) -> dict[str, str]:
        """Return the audited inference policy of this pinned upstream sampler."""

        policies = {
            "flm": "flm:checkpoint_config_and_loaded_runtime_code_bound",
            "duo": "duo:pinned_internal_bf16_autocast_with_fp32_sensitive_ops",
            "mdlm": "mdlm:pinned_internal_bf16_autocast_with_fp32_sensitive_ops",
            "candi": "candi:pinned_internal_bf16_autocast_with_fp32_sensitive_ops",
            "langflow": "langflow:pinned_internal_bf16_autocast_with_fp32_sensitive_ops",
            "rdlm": "rdlm:pinned_internal_bf16_autocast_with_fp32_sensitive_ops",
            "sdtt": "sdtt:pinned_bf16_mixed_config_and_internal_autocast",
            "di4c": "di4c:pinned_bf16_mixed_config_and_internal_autocast",
        }
        precision = (
            "resolved-from-checkpoint-config-at-execution"
            if request.model_id in {"flm", "fmlm"}
            else "bf16-mixed"
        )
        return {
            "precision": precision,
            "precision_policy": policies[self.upstream],
            "precision_evidence": (
                "static_policy_bound_to_checkpoint_and_runtime_code_"
                "not_runtime_autocast_observation"
            ),
        }

    def render_benchmark_command(
        self,
        request: RunRequest,
        run_dir: Path,
        *,
        output: Path,
        metadata_path: Path,
        precision: str,
        dry_run: bool,
    ) -> list[str]:
        """Inject timing into this adapter's real loaded-model sampler command."""

        if request.sample_count != 1:
            raise AdapterError("primary latency benchmark requires sample_count=1")
        if precision != "author":
            raise AdapterError("benchmark precision must use the pinned author policy")
        output = output.resolve()
        metadata_path = metadata_path.resolve()
        if output.is_symlink() or metadata_path.is_symlink():
            raise AdapterError("benchmark output paths must not be symlinks")
        arguments = self.render_command(request, run_dir, dry_run=dry_run)
        marker = "-m"
        capture_command = marker in arguments and "dlb.adapters.capture" in arguments
        if capture_command:
            separator = arguments.index("--")
            arguments[separator:separator] = [
                f"--benchmark-output={output}",
                f"--benchmark-metadata={metadata_path}",
                f"--benchmark-precision={precision}",
            ]
            replacements = {
                "loader.eval_batch_size": "1",
                "sampling.num_sample_batches": "1",
                "sampling.batch_per_gpu": "1",
                "--expected-samples": "1",
            }
            for key, value in replacements.items():
                prefix = key + "="
                for index, argument in enumerate(arguments):
                    if argument.startswith(prefix):
                        arguments[index] = prefix + value
            for key in ("--num_samples", "--batch_size"):
                if key in arguments:
                    arguments[arguments.index(key) + 1] = "1"
        else:
            arguments.extend(
                [
                    "--benchmark-output",
                    str(output),
                    "--benchmark-metadata",
                    str(metadata_path),
                    "--benchmark-precision",
                    precision,
                ]
            )
            for key in ("--sample-count", "--batch-size"):
                if key in arguments:
                    arguments[arguments.index(key) + 1] = "1"
        self._validate_argv(arguments)
        return arguments

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
        root, sequence_length, batch_size = self._validate_conversion_request(
            request, run_dir
        )
        self._require_conversion_provenance(root, request)
        expected_generated = math.ceil(request.sample_count / batch_size) * batch_size
        capture_path = run_dir.resolve() / "upstream_token_ids.json"
        capture = self._read_capture(capture_path, expected_generated)

        if self.upstream == "mdlm":
            texts = [item["text"] for item in capture]
            token_ids = [item["token_ids"] for item in capture]
            output_format = "dlb-upstream-token-capture-v1"
            token_ids_source = "upstream"
            token_ids_transformation = {
                "max_length": sequence_length,
                "operation": "validated_exact_length",
            }
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
                token_ids_transformation = {
                    "max_length": sequence_length,
                    "operation": "validated_exact_length",
                }
            else:
                token_ids, token_ids_transformation = self._retokenize(
                    root, request.dataset_id, texts, sequence_length
                )
                token_ids_source = "retokenized"

        dataset = self._load_data_contract(root, request.dataset_id)
        tokenizer_name = dataset["tokenizer"]
        vocab_size = self._TOKENIZER_VOCAB_SIZES[tokenizer_name]
        self._validate_samples(
            texts, token_ids, expected_generated, vocab_size, sequence_length
        )

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
                "token_ids_transformation": token_ids_transformation,
                "tokenizer": tokenizer_name,
                "tokenizer_revision": self._tokenizer_revision(root, tokenizer_name),
                "checkpoint_sha256": request.checkpoint_sha256,
                "checkpoint_lock_id": request.checkpoint_lock_id,
                "checkpoint_selection": request.checkpoint_selection,
                "teacher_family": request.checkpoint_teacher_family,
                "generation_seconds_source": "unavailable_excluded_sentinel",
                "generation_seconds_sentinel": 0.0,
                "exclude_from_latency": True,
            },
        )
        return records

    def _convert_capture_outputs(
        self,
        request: RunRequest,
        run_dir: Path,
        *,
        retokenize_inexact_rows: bool = False,
    ) -> Iterable[SampleRecord]:
        """Convert an exact project capture produced around a pinned sampler."""

        root, sequence_length, _ = self._validate_conversion_request(request, run_dir)
        self._require_conversion_provenance(root, request)
        capture_path = run_dir.resolve() / "upstream_token_ids.json"
        capture = self._read_capture(capture_path, request.sample_count)
        if not capture:
            raise AdapterError(f"token capture is missing or empty: {capture_path}")
        texts = [item["text"] for item in capture]
        upstream_token_ids = [item["token_ids"] for item in capture]
        token_ids = upstream_token_ids
        token_ids_source = "upstream"
        token_ids_transformation: dict[str, object] = {
            "max_length": sequence_length,
            "operation": "validated_exact_length",
        }
        if retokenize_inexact_rows and any(
            not isinstance(tokens, list) or len(tokens) != sequence_length
            for tokens in upstream_token_ids
        ):
            token_ids, token_ids_transformation = self._retokenize(
                root, request.dataset_id, texts, sequence_length
            )
            token_ids_transformation = {
                **token_ids_transformation,
                "reason": "upstream_ids_not_canonical_length",
            }
            token_ids_source = "retokenized"

        dataset = self._load_data_contract(root, request.dataset_id)
        tokenizer_name = dataset["tokenizer"]
        self._validate_samples(
            texts,
            token_ids,
            request.sample_count,
            self._TOKENIZER_VOCAB_SIZES[tokenizer_name],
            sequence_length,
        )
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
                "format": "dlb-upstream-token-capture-v1",
                "upstream_output": str(capture_path),
                "generated_samples": request.sample_count,
                "requested_samples": request.sample_count,
                "trimmed_samples": 0,
                "trim_policy": "none_exact_count",
                "token_ids_source": token_ids_source,
                "token_ids_transformation": token_ids_transformation,
                "tokenizer": tokenizer_name,
                "tokenizer_revision": self._tokenizer_revision(root, tokenizer_name),
                "checkpoint_sha256": request.checkpoint_sha256,
                "checkpoint_lock_id": request.checkpoint_lock_id,
                "checkpoint_selection": request.checkpoint_selection,
                "teacher_family": request.checkpoint_teacher_family,
                "generation_seconds_source": "unavailable_excluded_sentinel",
                "generation_seconds_sentinel": 0.0,
                "exclude_from_latency": True,
            },
        )
        return records

    def _require_conversion_provenance(self, root: Path, request: RunRequest) -> None:
        if (
            not isinstance(request.checkpoint_sha256, str)
            or len(request.checkpoint_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in request.checkpoint_sha256
            )
            or not isinstance(request.checkpoint_lock_id, str)
            or not request.checkpoint_lock_id
            or not isinstance(request.checkpoint_selection, dict)
            or not request.checkpoint_selection
            or not isinstance(request.checkpoint_teacher_family, str)
            or not request.checkpoint_teacher_family
        ):
            raise AdapterError("runner-resolved checkpoint provenance is required")
        registry = load_registry(root / "configs" / "experiments.yaml")
        try:
            support = registry.models[request.model_id].datasets[request.dataset_id]
            canonical = _resolve_checkpoint_provenance(
                root, request, support.train_recipe
            )
        except (KeyError, OSError, ValueError) as error:
            raise AdapterError(f"canonical checkpoint provenance is invalid: {error}") from error
        expected_family = self.teacher_families[request.model_id]
        if canonical.teacher_family != expected_family:
            raise AdapterError(
                f"canonical checkpoint teacher family {canonical.teacher_family!r} "
                f"does not match {expected_family!r}"
            )
        if request.checkpoint_sha256 != canonical.sha256:
            raise AdapterError("checkpoint SHA differs from canonical checkpoint provenance")
        if request.checkpoint_lock_id != canonical.lock_id:
            raise AdapterError("checkpoint lock ID differs from canonical checkpoint provenance")
        if request.checkpoint_selection != canonical.selection:
            raise AdapterError("checkpoint selection differs from canonical checkpoint provenance")
        if request.checkpoint_teacher_family != canonical.teacher_family:
            raise AdapterError(
                "checkpoint teacher family differs from canonical checkpoint provenance"
            )

    def _runtime_asset_digests(
        self,
        root: Path,
        request: RunRequest,
        checkpoint_path: Path,
        config_path: Path,
        *,
        dry_run: bool,
    ) -> tuple[str, str]:
        """Authenticate selected bytes before a server wrapper may deserialize them."""

        if dry_run:
            return "dry-run-unverified", "dry-run-unverified"
        self._require_conversion_provenance(root, request)
        checkpoint_digest = sha256_file(checkpoint_path)
        config_digest = sha256_file(config_path)
        selection = request.checkpoint_selection or {}
        resource_id = selection.get("resource")
        if isinstance(resource_id, str):
            lock_path = root / "artifacts" / "checkpoint_lock.json"
            try:
                lock = json.loads(lock_path.read_text(encoding="utf-8"))
                record = lock["resources"][resource_id]
                files = record["files"]
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
                raise AdapterError("canonical checkpoint lock is invalid") from error
            if not isinstance(files, list):
                raise AdapterError("canonical checkpoint lock file inventory is invalid")
            expected = {
                str(item.get("path")): item.get("sha256")
                for item in files
                if isinstance(item, dict)
            }
            for path, digest, label in (
                (checkpoint_path, checkpoint_digest, "student checkpoint"),
                (config_path, config_digest, "sampling config"),
            ):
                relative = path.relative_to(root).as_posix()
                if selection.get("sampling_config_source") == "project" and path == config_path:
                    locked_digest = selection.get("sampling_config_sha256")
                else:
                    locked_digest = expected.get(relative)
                if locked_digest != digest:
                    raise AdapterError(
                        f"{label} SHA differs from canonical checkpoint provenance: {path}"
                    )
        return checkpoint_digest, config_digest

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
        sample_suffix = (
            Path("samples")
            / request.dataset_id
            / request.model_id
            / f"steps_{request.step_count}"
        )
        if request.results_root is not None:
            results_root = Path(request.results_root).resolve()
            if results_root.is_symlink():
                raise AdapterError(f"results root is unsafe: {results_root}")
            root = self._project_root_for_results_root(results_root)
            if results_root / sample_suffix != run_dir:
                raise AdapterError(f"run directory does not match the request: {run_dir}")
        else:
            suffix = Path("results") / sample_suffix
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

    def _project_root_for_results_root(self, results_root: Path) -> Path:
        for candidate in results_root.parents:
            if (
                (candidate / "configs" / "experiments.yaml").is_file()
                and (candidate / "artifacts" / "data.yaml").is_file()
            ):
                return candidate.resolve()
        raise AdapterError(
            f"results root is not under a canonical project root: {results_root}"
        )

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
            config_path: Path | None = None
            config_sha256: str | None = None
            if coverage.sampling_config is not None:
                if coverage.sampling_config_source == "resource":
                    config_path = (base / coverage.sampling_config).absolute()
                else:
                    config_path = (root / coverage.sampling_config).absolute()
                    config_sha256 = coverage.sampling_config_sha256
                    if (
                        config_path.is_symlink()
                        or not config_path.is_file()
                        or sha256_file(config_path) != config_sha256
                    ):
                        raise AdapterError(
                            f"project sampling config is missing or differs from manifest: {config_path}"
                        )
            selection = CheckpointSelection(
                path=path.resolve(),
                teacher_family=family,
                source=f"resource:{coverage.resource}",
                required_files=tuple(resource.required_files),
                config_path=config_path,
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
        output = (root / recipe.output).absolute()
        if recipe.sampling_checkpoint is not None:
            path = (output / recipe.sampling_checkpoint).absolute()
            if not dry_run:
                relative = Path(recipe.sampling_checkpoint)
                ancestors = [output]
                current = output
                for part in relative.parts[:-1]:
                    current = current / part
                    ancestors.append(current)
                if (
                    any(ancestor.is_symlink() for ancestor in ancestors)
                    or path.is_symlink()
                    or not path.is_file()
                    or path.stat().st_size <= 0
                ):
                    raise AdapterError(
                        f"recipe sampling checkpoint is missing or unsafe: {path}"
                    )
        else:
            path = output if dry_run else self._select_recipe_checkpoint(output)
        config_path = (
            (output / recipe.sampling_config).absolute()
            if recipe.sampling_config is not None
            else None
        )
        if not dry_run and config_path is not None:
            relative = Path(recipe.sampling_config)
            ancestors = [output]
            current = output
            for part in relative.parts[:-1]:
                current = current / part
                ancestors.append(current)
            if (
                any(ancestor.is_symlink() for ancestor in ancestors)
                or config_path.is_symlink()
                or not config_path.is_file()
                or config_path.stat().st_size <= 0
            ):
                raise AdapterError(
                    f"recipe sampling config is missing or unsafe: {config_path}"
                )
        return CheckpointSelection(
            path=path,
            teacher_family=recipe.teacher_family,
            source=f"recipe:{recipe_id}",
            config_path=config_path,
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
        self,
        texts: list[object],
        token_ids: list[object],
        expected: int,
        vocab_size: int,
        sequence_length: int,
    ) -> None:
        if len(texts) != expected or len(token_ids) != expected:
            raise AdapterError(
                f"expected {expected} generated samples, found {min(len(texts), len(token_ids))}"
            )
        self._validate_texts(texts)
        for index, (_, tokens) in enumerate(zip(texts, token_ids)):
            if not isinstance(tokens, list) or not tokens:
                raise AdapterError(f"empty token_ids at sample {index}")
            if len(tokens) != sequence_length:
                raise AdapterError(
                    f"expected {sequence_length} tokens at sample {index}, found {len(tokens)}"
                )
            for token in tokens:
                if type(token) is not int or token < 0 or token >= vocab_size:
                    raise AdapterError(f"invalid token at sample {index}: {token!r}")

    @staticmethod
    def _validate_texts(texts: list[object]) -> None:
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise AdapterError(f"empty text at sample {index}")

    def _retokenize(
        self, root: Path, dataset_id: str, texts: list[str], sequence_length: int
    ) -> tuple[list[list[int]], dict[str, object]]:
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
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is None:
                raise AdapterError(f"canonical tokenizer {tokenizer_name} has no padding token")
            tokenizer.pad_token = tokenizer.eos_token
        settings = {
            "add_special_tokens": False,
            "padding": "max_length",
            "truncation": True,
            "max_length": sequence_length,
            "return_attention_mask": False,
        }
        encoded = tokenizer(texts, **settings)
        try:
            token_ids = [list(tokens) for tokens in encoded["input_ids"]]
        except (KeyError, TypeError) as error:
            raise AdapterError("pinned tokenizer returned invalid input_ids") from error
        return token_ids, {
            "add_special_tokens": False,
            "max_length": sequence_length,
            "padding": "max_length",
            "pad_token_id": tokenizer.pad_token_id,
            "truncation": True,
        }

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

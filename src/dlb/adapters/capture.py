"""Run a pinned upstream entrypoint while capturing returned token-ID samples."""

from __future__ import annotations

import builtins
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Callable

import yaml

from dlb.io import atomic_json_write
from dlb.timing import benchmark_and_publish


@dataclass(frozen=True)
class CaptureInvocation:
    entrypoint: Path
    capture_path: Path
    kind: str = "teacher"
    expected_samples: int | None = None
    saved_config_path: Path | None = None
    saved_sde_path: Path | None = None
    data_config_path: Path | None = None
    downloads_manifest_path: Path | None = None
    dataset_id: str | None = None
    tokenizer_snapshot: Path | None = None
    benchmark_output: Path | None = None
    benchmark_metadata: Path | None = None
    benchmark_precision: str | None = None


@dataclass(frozen=True)
class TokenizerBinding:
    tokenizer_id: str
    revision: str
    snapshot: Path


def _split_arguments(argv: list[str]) -> tuple[CaptureInvocation, list[str]]:
    try:
        separator = argv.index("--")
    except ValueError as error:
        raise ValueError("capture wrapper requires -- before upstream arguments") from error
    wrapper = argv[:separator]
    forwarded = argv[separator + 1 :]
    allowed = {
        "--upstream-entrypoint",
        "--capture-path",
        "--capture-kind",
        "--expected-samples",
        "--saved-config-path",
        "--saved-sde-path",
        "--data-config-path",
        "--downloads-manifest-path",
        "--dataset-id",
        "--tokenizer-snapshot",
        "--benchmark-output",
        "--benchmark-metadata",
        "--benchmark-precision",
    }
    values: dict[str, str] = {}
    for argument in wrapper:
        key, marker, value = argument.partition("=")
        if not marker or key not in allowed or not value:
            raise ValueError(f"invalid capture-wrapper argument: {argument!r}")
        if key in values:
            raise ValueError(f"duplicate capture-wrapper argument: {key}")
        values[key] = value
    required = {"--upstream-entrypoint", "--capture-path"}
    if not required.issubset(values):
        raise ValueError("capture wrapper requires entrypoint and capture path")

    entrypoint = Path(values["--upstream-entrypoint"])
    capture_path = Path(values["--capture-path"])
    if not entrypoint.is_absolute() or entrypoint.is_symlink() or not entrypoint.is_file():
        raise ValueError(f"upstream entrypoint is missing or unsafe: {entrypoint}")
    if not capture_path.is_absolute():
        raise ValueError("capture path must be absolute")
    kind = values.get("--capture-kind", "teacher")
    if kind not in {"teacher", "langflow", "rdlm"}:
        raise ValueError(f"unsupported capture kind: {kind!r}")
    expected_text = values.get("--expected-samples")
    if expected_text is None:
        expected_samples = None
    else:
        try:
            expected_samples = int(expected_text)
        except ValueError as error:
            raise ValueError("expected sample count must be an integer") from error
        if expected_samples <= 0:
            raise ValueError("expected sample count must be positive")
    saved_config_path = (
        Path(values["--saved-config-path"])
        if "--saved-config-path" in values
        else None
    )
    saved_sde_path = (
        Path(values["--saved-sde-path"])
        if "--saved-sde-path" in values
        else None
    )
    data_config_path = (
        Path(values["--data-config-path"])
        if "--data-config-path" in values
        else None
    )
    downloads_manifest_path = (
        Path(values["--downloads-manifest-path"])
        if "--downloads-manifest-path" in values
        else None
    )
    dataset_id = values.get("--dataset-id")
    tokenizer_snapshot = (
        Path(values["--tokenizer-snapshot"])
        if "--tokenizer-snapshot" in values
        else None
    )
    benchmark_output = (
        Path(values["--benchmark-output"])
        if "--benchmark-output" in values
        else None
    )
    benchmark_metadata = (
        Path(values["--benchmark-metadata"])
        if "--benchmark-metadata" in values
        else None
    )
    benchmark_precision = values.get("--benchmark-precision")
    benchmark_fields = (benchmark_output, benchmark_metadata, benchmark_precision)
    if any(value is not None for value in benchmark_fields) and not all(
        value is not None for value in benchmark_fields
    ):
        raise ValueError("benchmark capture arguments must be provided together")
    if benchmark_output is not None:
        if not benchmark_output.is_absolute() or not benchmark_metadata.is_absolute():
            raise ValueError("benchmark paths must be absolute")
        if benchmark_precision != "author":
            raise ValueError("benchmark precision is invalid")
    if kind in {"langflow", "rdlm"} and expected_samples is None:
        raise ValueError(f"{kind} capture requires an expected sample count")
    if kind == "rdlm" and (saved_config_path is None or saved_sde_path is None):
        raise ValueError("RDLM capture requires saved config and SDE paths")
    tokenizer_fields = (
        data_config_path,
        downloads_manifest_path,
        dataset_id,
        tokenizer_snapshot,
    )
    if kind in {"langflow", "rdlm"} and any(value is None for value in tokenizer_fields):
        raise ValueError(f"{kind} capture requires a locked tokenizer snapshot")
    return (
        CaptureInvocation(
            entrypoint=entrypoint,
            capture_path=capture_path,
            kind=kind,
            expected_samples=expected_samples,
            saved_config_path=saved_config_path,
            saved_sde_path=saved_sde_path,
            data_config_path=data_config_path,
            downloads_manifest_path=downloads_manifest_path,
            dataset_id=dataset_id,
            tokenizer_snapshot=tokenizer_snapshot,
            benchmark_output=benchmark_output,
            benchmark_metadata=benchmark_metadata,
            benchmark_precision=benchmark_precision,
        ),
        forwarded,
    )


def _require_absolute_file(path: Path | None, label: str) -> Path:
    if path is None or not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or unsafe: {path}")
    return path


def _load_tokenizer_binding(invocation: CaptureInvocation) -> TokenizerBinding:
    """Resolve one local snapshot against both immutable project data records."""

    config_path = _require_absolute_file(invocation.data_config_path, "data config")
    downloads_path = _require_absolute_file(
        invocation.downloads_manifest_path, "download manifest"
    )
    try:
        configuration = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        downloads = json.loads(downloads_path.read_text(encoding="utf-8"))
        dataset = configuration["datasets"][invocation.dataset_id]
        tokenizer_id = dataset["tokenizer"]
        revision = configuration["models"][tokenizer_id]
        record = downloads["models"][tokenizer_id]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("locked tokenizer metadata is invalid") from error
    if downloads.get("schema_version") != 1:
        raise ValueError("download manifest schema is not supported")
    if not isinstance(tokenizer_id, str) or not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("data config tokenizer binding is invalid")
    if record.get("repo_id") != tokenizer_id or record.get("revision") != revision:
        raise ValueError("download manifest tokenizer binding differs from data config")
    recorded_snapshot = record.get("snapshot_path")
    if not isinstance(recorded_snapshot, str) or not recorded_snapshot:
        raise ValueError("download manifest tokenizer snapshot is missing")
    snapshot_path = Path(recorded_snapshot)
    if not snapshot_path.is_absolute():
        snapshot_path = config_path.parents[1] / snapshot_path
    requested_snapshot = invocation.tokenizer_snapshot
    if requested_snapshot is None or not requested_snapshot.is_absolute():
        raise ValueError("tokenizer snapshot path must be absolute")
    if requested_snapshot != snapshot_path:
        raise ValueError("capture tokenizer snapshot differs from download manifest")
    parts = snapshot_path.parts
    try:
        recorded_revision = parts[parts.index("snapshots") + 1]
    except (ValueError, IndexError) as error:
        raise ValueError("download manifest tokenizer snapshot path is invalid") from error
    if recorded_revision != revision:
        raise ValueError("download manifest tokenizer snapshot revision differs from data config")
    if snapshot_path.is_symlink() or not snapshot_path.is_dir():
        raise ValueError(f"locked tokenizer snapshot is missing or unsafe: {snapshot_path}")
    return TokenizerBinding(tokenizer_id, revision, snapshot_path)


@contextmanager
def _offline_huggingface():
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update({name: "1" for name in names})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _adapt_hf_masked_lm_backbone(model: object) -> object:
    forward = getattr(model, "forward", None)
    if not callable(forward):
        return model
    try:
        parameters = inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return model
    if {"x", "sigma"} <= set(parameters):
        return model
    if not {"input_ids", "timesteps"} <= set(parameters):
        return model
    try:
        import torch
    except ImportError:
        module_base = object
    else:
        module_base = torch.nn.Module

    class KeywordBackboneAdapter(module_base):
        def __init__(self, inner: object) -> None:
            if module_base is not object:
                super().__init__()
            self.inner = inner

        def forward(self, *args, **kwargs):
            if "x" in kwargs:
                kwargs["input_ids"] = kwargs.pop("x")
            elif args:
                kwargs["input_ids"] = args[0]
                args = args[1:]
            if "sigma" in kwargs:
                kwargs["timesteps"] = kwargs.pop("sigma")
            elif args:
                kwargs["timesteps"] = args[0]
                args = args[1:]
            kwargs.pop("class_cond", None)
            kwargs.pop("weights", None)
            call = self.inner if callable(self.inner) else self.inner.forward
            output = call(*args, **kwargs)
            if hasattr(output, "logits"):
                return output.logits
            if isinstance(output, tuple) and output:
                return output[0]
            return output

        def __getattr__(self, name: str) -> object:
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.inner, name)

    if module_base is object:
        KeywordBackboneAdapter.__call__ = KeywordBackboneAdapter.forward
    return KeywordBackboneAdapter(model)


@contextmanager
def _patched_hf_masked_lm_backbone():
    try:
        transformers = importlib.import_module("transformers")
    except ImportError:
        yield
        return
    auto_model = getattr(transformers, "AutoModelForMaskedLM", None)
    original = getattr(auto_model, "from_pretrained", None)
    if not callable(original):
        yield
        return

    def from_pretrained(*args, **kwargs):
        return _adapt_hf_masked_lm_backbone(original(*args, **kwargs))

    auto_model.from_pretrained = from_pretrained
    try:
        yield
    finally:
        auto_model.from_pretrained = original


def _load_entrypoint(path: Path) -> ModuleType:
    sys.path.insert(0, str(path.parent))
    module_name = "dlb_pinned_upstream_main"
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ValueError(f"cannot load upstream entrypoint: {path}")
    module = importlib.util.module_from_spec(specification)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


def _sample_owner(module: ModuleType) -> type:
    diffusion = getattr(module, "diffusion", None)
    if diffusion is not None and hasattr(diffusion, "Diffusion"):
        return diffusion.Diffusion
    algo = getattr(module, "algo", None)
    trainer_base = getattr(algo, "trainer_base", None)
    owner = getattr(trainer_base, "TrainerBase", None)
    if isinstance(owner, type):
        return owner
    raise ValueError("upstream entrypoint has no supported sampling owner")


def _write_capture(
    path: Path,
    texts: list[object],
    token_rows: list[object],
    expected: int | None,
) -> None:
    if len(texts) != len(token_rows):
        raise ValueError("upstream token and text sample counts differ")
    if expected is not None and len(texts) != expected:
        raise ValueError(f"expected {expected} captured samples, found {len(texts)}")
    samples: list[dict[str, object]] = []
    for index, (text, tokens) in enumerate(zip(texts, token_rows, strict=True)):
        if not isinstance(text, str):
            raise ValueError(f"captured text {index} is not a string")
        if not isinstance(tokens, list):
            raise ValueError(f"captured token row {index} is not a list")
        samples.append({"sample_id": index, "text": text, "token_ids": tokens})
    atomic_json_write(path, {"schema": "dlb-upstream-token-capture-v1", "samples": samples})


def _rows(value: object, *, label: str = "token rows") -> list[object]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if not isinstance(value, list):
        raise ValueError(f"{label} returned invalid token rows")
    rows: list[object] = []
    for row in value:
        if hasattr(row, "detach"):
            row = row.detach().cpu().tolist()
        elif not isinstance(row, list):
            row = list(row)
        rows.append(row)
    return rows


def _run_main(module: ModuleType, entrypoint: Path, forwarded: list[str]) -> None:
    previous_argv = sys.argv
    try:
        sys.argv = [
            str(entrypoint),
            *_forwarded_with_hydra_config_path(entrypoint, forwarded),
        ]
        module.main()
    finally:
        sys.argv = previous_argv


def _forwarded_with_hydra_config_path(entrypoint: Path, forwarded: list[str]) -> list[str]:
    if any(
        argument in {"--config-path", "-cp"}
        or argument.startswith("--config-path=")
        or argument.startswith("-cp=")
        for argument in forwarded
    ):
        return forwarded
    config_dir = entrypoint.parent / "configs"
    if config_dir.is_dir() and not config_dir.is_symlink():
        return [f"--config-path={config_dir.resolve()}", *forwarded]
    return forwarded


def _capture_teacher(
    module: ModuleType, invocation: CaptureInvocation, forwarded: list[str]
) -> None:
    owner = _sample_owner(module)
    original = owner.restore_model_and_sample
    captured: list[dict[str, object]] = []

    def capture(self, *args, **kwargs):
        if invocation.benchmark_output is not None:
            num_steps = kwargs.get("num_steps", args[0] if args else None)
            eps = kwargs.get("eps", 1e-5)
            if type(num_steps) is not int or num_steps <= 0:
                raise ValueError("benchmark sampler requires a positive num_steps")
            if invocation.entrypoint.parent.name == "mdlm":
                if self.ema:
                    parameters = tuple(
                        list(self.backbone.parameters()) + list(self.noise.parameters())
                    )
                    self.ema.store(parameters)
                    self.ema.copy_to(parameters)
                self.backbone.eval()
                self.noise.eval()
                generate_one = lambda: self._sample(num_steps=num_steps, eps=eps)
            else:
                self._eval_mode()
                generate_one = lambda: self.generate_samples(
                    num_samples=1, num_steps=num_steps, eps=eps
                )
            try:
                return benchmark_and_publish(
                    generate_one,
                    model=self,
                    output=invocation.benchmark_output,
                    metadata_path=invocation.benchmark_metadata,
                    precision=invocation.benchmark_precision,
                )
            finally:
                if invocation.entrypoint.parent.name == "mdlm":
                    if self.ema:
                        self.ema.restore(parameters)
                    self.backbone.train()
                    self.noise.train()
                else:
                    self._train_mode()
        result = original(self, *args, **kwargs)
        token_rows = _rows(result, label="teacher sampler")
        text_rows = list(self.tokenizer.batch_decode(result))
        if len(token_rows) != len(text_rows):
            raise ValueError("upstream token and text batch sizes differ")
        for tokens, text in zip(token_rows, text_rows, strict=True):
            captured.append(
                {"sample_id": len(captured), "text": text, "token_ids": tokens}
            )
        return result

    owner.restore_model_and_sample = capture
    with _patched_hf_masked_lm_backbone():
        _run_main(module, invocation.entrypoint, forwarded)
    atomic_json_write(
        invocation.capture_path,
        {"schema": "dlb-upstream-token-capture-v1", "samples": captured},
    )


def _capture_langflow(
    module: ModuleType,
    invocation: CaptureInvocation,
    forwarded: list[str],
    tokenizer: TokenizerBinding,
) -> None:
    owner = getattr(module, "LangFlow", None)
    auto_tokenizer = getattr(module, "AutoTokenizer", None)
    if not isinstance(owner, type) or auto_tokenizer is None:
        raise ValueError("LangFlow entrypoint lacks its model or tokenizer")
    original_generate = owner.generate_samples
    token_rows: list[object] = []
    texts: list[object] = []

    def generate(self, *args, **kwargs):
        if invocation.benchmark_output is None:
            result = original_generate(self, *args, **kwargs)
        else:
            benchmark_kwargs = dict(kwargs)
            benchmark_kwargs["num_samples"] = 1
            result = benchmark_and_publish(
                lambda: original_generate(self, *args, **benchmark_kwargs),
                model=self,
                output=invocation.benchmark_output,
                metadata_path=invocation.benchmark_metadata,
                precision=invocation.benchmark_precision,
            )
        rows = result.detach().cpu().tolist()
        if not isinstance(rows, list):
            raise ValueError("LangFlow sampler returned invalid token rows")
        token_rows.extend(rows)
        return result

    class TokenizerProxy:
        def __init__(self, tokenizer: object) -> None:
            self._tokenizer = tokenizer

        def __getattr__(self, name: str) -> object:
            return getattr(self._tokenizer, name)

        def batch_decode(self, *args, **kwargs):
            decoded = list(self._tokenizer.batch_decode(*args, **kwargs))
            texts[:] = decoded
            return decoded

    class AutoTokenizerProxy:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            del args
            kwargs.pop("revision", None)
            kwargs["local_files_only"] = True
            loaded = auto_tokenizer.from_pretrained(str(tokenizer.snapshot), **kwargs)
            return TokenizerProxy(loaded)

    owner.generate_samples = generate
    module.AutoTokenizer = AutoTokenizerProxy
    with _offline_huggingface():
        _run_main(module, invocation.entrypoint, forwarded)
    _write_capture(
        invocation.capture_path,
        texts,
        token_rows,
        invocation.expected_samples,
    )


def _model_path(forwarded: list[str]) -> Path:
    prefix = "model_path="
    values = [argument[len(prefix) :] for argument in forwarded if argument.startswith(prefix)]
    if len(values) != 1 or not values[0]:
        raise ValueError("RDLM capture requires one model_path override")
    path = Path(values[0])
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"RDLM checkpoint is missing or unsafe: {path}")
    return path


def _positive_override(forwarded: list[str], key: str) -> int:
    prefix = key + "="
    values = [argument[len(prefix) :] for argument in forwarded if argument.startswith(prefix)]
    if len(values) != 1:
        raise ValueError(f"RDLM capture requires one {key} override")
    try:
        value = int(values[0])
    except ValueError as error:
        raise ValueError(f"RDLM {key} must be an integer") from error
    if value <= 0:
        raise ValueError(f"RDLM {key} must be positive")
    return value


def _require_saved_file(path: Path | None, label: str) -> Path:
    if (
        path is None
        or not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size <= 0
    ):
        raise ValueError(f"saved RDLM {label} is missing or unsafe: {path}")
    return path


def _rdlm_benchmark_sampling_fn(
    sampling_fn: Callable[[object], object], invocation: CaptureInvocation
) -> Callable[[object], object]:
    """Publish one benchmark the first time RDLM reaches its loaded sampling_fn."""

    if invocation.benchmark_output is None:
        return sampling_fn
    published = False

    def sample(model: object) -> object:
        nonlocal published
        if published:
            return sampling_fn(model)
        published = True
        return benchmark_and_publish(
            lambda: sampling_fn(model),
            model=model,
            output=invocation.benchmark_output,
            metadata_path=invocation.benchmark_metadata,
            precision=invocation.benchmark_precision,
        )

    return sample


def _capture_rdlm(
    module: ModuleType, invocation: CaptureInvocation, forwarded: list[str]
) -> None:
    from omegaconf import OmegaConf

    tokenizer = _load_tokenizer_binding(invocation)
    saved_config_path = _require_saved_file(invocation.saved_config_path, "config")
    saved_sde_path = _require_saved_file(invocation.saved_sde_path, "SDE")
    checkpoint_path = _model_path(forwarded)
    expected = invocation.expected_samples
    if type(expected) is not int or expected <= 0:
        raise ValueError("RDLM capture requires a positive expected sample count")
    batch_size = _positive_override(forwarded, "sampling.batch_per_gpu")
    sample_batches = math.ceil(expected / batch_size)
    saved_config = OmegaConf.load(saved_config_path)
    if OmegaConf.select(saved_config, "model.length") != 128:
        raise ValueError("saved RDLM config does not declare LM1B model length 128")
    if OmegaConf.select(saved_config, "data.train") != "lm1b":
        raise ValueError("saved RDLM config is not the LM1B training config")

    run_sample = importlib.import_module("run_sample")
    data_module = getattr(run_sample, "data", None)
    transformers_module = getattr(data_module, "transformers", None)
    original_bert_tokenizer = getattr(transformers_module, "BertTokenizer", None)
    original_auto_tokenizer = getattr(data_module, "AutoTokenizer", None)
    if original_bert_tokenizer is None or original_auto_tokenizer is None:
        raise ValueError("RDLM data module lacks its tokenizer constructors")
    original_torch = run_sample.torch
    original_open: Callable[..., object] = builtins.open
    original_tqdm = run_sample.tqdm
    original_instantiate = run_sample.instantiate
    original_shift_factory = run_sample.sutils.find_bos_and_shift_fn
    original_sampling_factory = (
        run_sample.sampling.get_sampling_fn
        if invocation.benchmark_output is not None
        else None
    )
    texts: list[object] = []
    token_rows: list[object] = []
    saved_config_used = False
    saved_sde_opened = False
    saved_sde_used = False

    def locked_tokenizer_proxy(original):
        class LocalTokenizerProxy:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                del args
                kwargs.pop("revision", None)
                kwargs["local_files_only"] = True
                return original.from_pretrained(str(tokenizer.snapshot), **kwargs)

        return LocalTokenizerProxy

    class TorchProxy:
        def __getattr__(self, name: str) -> object:
            return getattr(original_torch, name)

        def load(self, path, *args, **kwargs):
            nonlocal saved_config_used
            state = original_torch.load(path, *args, **kwargs)
            if Path(path).resolve() != checkpoint_path.resolve():
                return state
            if not isinstance(state, dict):
                raise ValueError("RDLM checkpoint state is not a mapping")
            saved_config_used = True
            return {**state, "config": saved_config}

    def routed_open(path, mode="r", *args, **kwargs):
        nonlocal saved_sde_opened
        requested = Path(path)
        if "r" in mode and "b" in mode and requested.name == "sde.pkl":
            saved_sde_opened = True
            return original_open(saved_sde_path, mode, *args, **kwargs)
        return original_open(path, mode, *args, **kwargs)

    def instantiate(config, *args, **kwargs):
        nonlocal saved_sde_used
        is_sde = {"manifold", "scheduler", "prior_dist"}.issubset(kwargs)
        if is_sde and "preprocessed" not in kwargs:
            raise ValueError("saved SDE preprocessing was not used by RDLM")
        if is_sde:
            saved_sde_used = True
        return original_instantiate(config, *args, **kwargs)

    def exact_sample_batches(iterable, *args, **kwargs):
        del iterable
        return original_tqdm(range(sample_batches), *args, **kwargs)

    def benchmark_sampling_factory(*args, **kwargs):
        if original_sampling_factory is None:
            raise ValueError("RDLM benchmark sampling factory is unavailable")
        sampling_fn = original_sampling_factory(*args, **kwargs)
        return _rdlm_benchmark_sampling_fn(sampling_fn, invocation)

    def capture_shift_factory(*args, **kwargs):
        shift = original_shift_factory(*args, **kwargs)

        def capture_shift(samples):
            sentences, shifted = shift(samples)
            texts[:] = list(sentences)
            token_rows[:] = _rows(shifted)
            return sentences, shifted

        return capture_shift

    def inline_spawn(function, args=(), nprocs=1, join=True, **kwargs):
        del join, kwargs
        if nprocs != 1:
            raise ValueError("RDLM capture requires ngpus=1")
        return function(0, *args)

    run_sample.torch = TorchProxy()
    data_module.transformers.BertTokenizer = locked_tokenizer_proxy(
        original_bert_tokenizer
    )
    data_module.AutoTokenizer = locked_tokenizer_proxy(original_auto_tokenizer)
    run_sample.open = routed_open
    run_sample.tqdm = exact_sample_batches
    if invocation.benchmark_output is not None:
        run_sample.sampling.get_sampling_fn = benchmark_sampling_factory
    run_sample.instantiate = instantiate
    run_sample.sutils.find_bos_and_shift_fn = capture_shift_factory
    module.mp.spawn = inline_spawn
    with _offline_huggingface():
        _run_main(module, invocation.entrypoint, forwarded)
    if not saved_config_used:
        raise ValueError("saved RDLM config was not used")
    if not saved_sde_opened or not saved_sde_used:
        raise ValueError("saved RDLM SDE was not used")
    if len(texts) < expected or len(token_rows) < expected:
        raise ValueError(
            f"expected {expected} captured samples, found {min(len(texts), len(token_rows))}"
        )
    _write_capture(
        invocation.capture_path,
        texts[:expected],
        token_rows[:expected],
        expected,
    )


def main(argv: list[str] | None = None) -> int:
    invocation, forwarded = _split_arguments(
        list(sys.argv[1:] if argv is None else argv)
    )
    module = _load_entrypoint(invocation.entrypoint)
    if invocation.kind == "langflow":
        tokenizer = _load_tokenizer_binding(invocation)
        _capture_langflow(module, invocation, forwarded, tokenizer)
    elif invocation.kind == "rdlm":
        _capture_rdlm(module, invocation, forwarded)
    else:
        _capture_teacher(module, invocation, forwarded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

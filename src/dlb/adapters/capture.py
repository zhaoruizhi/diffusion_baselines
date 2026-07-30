"""Run a pinned upstream entrypoint while capturing returned token-ID samples."""

from __future__ import annotations

import builtins
from dataclasses import dataclass
import importlib
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Callable

from dlb.io import atomic_json_write


@dataclass(frozen=True)
class CaptureInvocation:
    entrypoint: Path
    capture_path: Path
    kind: str = "teacher"
    expected_samples: int | None = None
    saved_config_path: Path | None = None
    saved_sde_path: Path | None = None


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
    if kind in {"langflow", "rdlm"} and expected_samples is None:
        raise ValueError(f"{kind} capture requires an expected sample count")
    if kind == "rdlm" and (saved_config_path is None or saved_sde_path is None):
        raise ValueError("RDLM capture requires saved config and SDE paths")
    return (
        CaptureInvocation(
            entrypoint=entrypoint,
            capture_path=capture_path,
            kind=kind,
            expected_samples=expected_samples,
            saved_config_path=saved_config_path,
            saved_sde_path=saved_sde_path,
        ),
        forwarded,
    )


def _load_entrypoint(path: Path) -> ModuleType:
    sys.path.insert(0, str(path.parent))
    specification = importlib.util.spec_from_file_location("dlb_pinned_upstream_main", path)
    if specification is None or specification.loader is None:
        raise ValueError(f"cannot load upstream entrypoint: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
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


def _run_main(module: ModuleType, entrypoint: Path, forwarded: list[str]) -> None:
    previous_argv = sys.argv
    try:
        sys.argv = [str(entrypoint), *forwarded]
        module.main()
    finally:
        sys.argv = previous_argv


def _capture_teacher(
    module: ModuleType, invocation: CaptureInvocation, forwarded: list[str]
) -> None:
    owner = _sample_owner(module)
    original = owner.restore_model_and_sample
    captured: list[dict[str, object]] = []

    def capture(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        token_rows = result.detach().cpu().tolist()
        text_rows = list(self.tokenizer.batch_decode(result))
        if len(token_rows) != len(text_rows):
            raise ValueError("upstream token and text batch sizes differ")
        for tokens, text in zip(token_rows, text_rows, strict=True):
            captured.append(
                {"sample_id": len(captured), "text": text, "token_ids": tokens}
            )
        return result

    owner.restore_model_and_sample = capture
    _run_main(module, invocation.entrypoint, forwarded)
    atomic_json_write(
        invocation.capture_path,
        {"schema": "dlb-upstream-token-capture-v1", "samples": captured},
    )


def _capture_langflow(
    module: ModuleType, invocation: CaptureInvocation, forwarded: list[str]
) -> None:
    owner = getattr(module, "LangFlow", None)
    auto_tokenizer = getattr(module, "AutoTokenizer", None)
    if not isinstance(owner, type) or auto_tokenizer is None:
        raise ValueError("LangFlow entrypoint lacks its model or tokenizer")
    original_generate = owner.generate_samples
    token_rows: list[object] = []
    texts: list[object] = []

    def generate(self, *args, **kwargs):
        result = original_generate(self, *args, **kwargs)
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
            return TokenizerProxy(auto_tokenizer.from_pretrained(*args, **kwargs))

    owner.generate_samples = generate
    module.AutoTokenizer = AutoTokenizerProxy
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


def _rows(value: object) -> list[object]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if not isinstance(value, list):
        raise ValueError("RDLM shift/decode returned invalid token rows")
    rows: list[object] = []
    for row in value:
        if hasattr(row, "detach"):
            row = row.detach().cpu().tolist()
        elif not isinstance(row, list):
            row = list(row)
        rows.append(row)
    return rows


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


def _capture_rdlm(
    module: ModuleType, invocation: CaptureInvocation, forwarded: list[str]
) -> None:
    from omegaconf import OmegaConf

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
    original_torch = run_sample.torch
    original_open: Callable[..., object] = builtins.open
    original_tqdm = run_sample.tqdm
    original_instantiate = run_sample.instantiate
    original_shift_factory = run_sample.sutils.find_bos_and_shift_fn
    texts: list[object] = []
    token_rows: list[object] = []
    saved_config_used = False
    saved_sde_opened = False
    saved_sde_used = False

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
    run_sample.open = routed_open
    run_sample.tqdm = exact_sample_batches
    run_sample.instantiate = instantiate
    run_sample.sutils.find_bos_and_shift_fn = capture_shift_factory
    module.mp.spawn = inline_spawn
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
        _capture_langflow(module, invocation, forwarded)
    elif invocation.kind == "rdlm":
        _capture_rdlm(module, invocation, forwarded)
    else:
        _capture_teacher(module, invocation, forwarded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

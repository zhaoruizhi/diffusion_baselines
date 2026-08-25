"""Run Di4C language distillation with project-controlled student initialization."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


_INIT_MODES = frozenset({"hf_small", "teacher", "scratch"})


def _install_upstream_path() -> Path:
    source = Path.cwd()
    sdtt_src = source / "sdtt" / "src"
    if not (sdtt_src / "sdtt" / "main.py").is_file():
        raise RuntimeError(f"Di4C SDTT source is missing: {sdtt_src}")
    sys.path.insert(0, str(sdtt_src))
    sys.path.insert(0, str(source))
    return source


def _student_init_mode(config: Any) -> str:
    mode = config.get("dlb_student_init", "hf_small")
    if mode not in _INIT_MODES:
        raise ValueError(
            f"unsupported dlb_student_init={mode!r}; expected one of {sorted(_INIT_MODES)}"
        )
    return str(mode)


def _eval_expression(expression: Any) -> Any:
    return eval(str(expression), {"__builtins__": {}}, {})


def _device_count() -> int:
    try:
        import torch
    except ImportError:
        return 0
    return int(torch.cuda.device_count())


def _register_omegaconf_resolvers() -> None:
    from omegaconf import OmegaConf

    resolvers = {
        "cwd": lambda: str(Path.cwd()),
        "device_count": _device_count,
        "div_up": lambda numerator, denominator: (
            int(numerator) + int(denominator) - 1
        )
        // int(denominator),
        "eval": _eval_expression,
    }
    for name, resolver in resolvers.items():
        try:
            OmegaConf.register_new_resolver(name, resolver, replace=True)
        except TypeError:
            OmegaConf.clear_resolver(name)
            OmegaConf.register_new_resolver(name, resolver)


def _teacher_initialized_student(config: Any):
    from sdtt.loading_utils import get_diffusion
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.name)
    return get_diffusion(config, tokenizer)


def _scratch_initialized_student(config: Any):
    from sdtt.models.loading_utils import get_backbone

    model = _teacher_initialized_student(config)
    model.backbone = get_backbone(config, model.vocab_size)
    if hasattr(model.backbone, "is_di4c"):
        model.backbone.is_di4c = True
    model.init_ema()
    return model


def _patch_student_loader() -> None:
    import sdtt

    original = sdtt.load_small_student
    if getattr(original, "_dlb_student_init_patch", False):
        return

    def load_small_student(
        loss: str = "kld",
        round: int = 1,
        config: Any | None = None,
        **kwargs: Any,
    ):
        if config is None:
            try:
                return original(loss=loss, round=round, config=config, **kwargs)
            except TypeError:
                if kwargs:
                    return original(loss=loss, round=round, config=config)
                raise

        mode = _student_init_mode(config)
        if mode == "hf_small":
            try:
                return original(loss=loss, round=round, config=config, **kwargs)
            except TypeError:
                if kwargs:
                    return original(loss=loss, round=round, config=config)
                raise
        if mode == "teacher":
            return _teacher_initialized_student(config)
        if mode == "scratch":
            return _scratch_initialized_student(config)
        raise AssertionError(f"unreachable Di4C student init mode: {mode}")

    load_small_student._dlb_student_init_patch = True  # type: ignore[attr-defined]
    sdtt.load_small_student = load_small_student


def main() -> int:
    _install_upstream_path()
    _register_omegaconf_resolvers()
    import hydra

    _patch_student_loader()
    import sdtt.main as upstream_main

    @hydra.main(
        version_base=None,
        config_path="../upstreams/di4c/sdtt/src/sdtt/configs",
        config_name="config",
    )
    def run(config: Any) -> None:
        upstream_main.main.__wrapped__(config)

    return int(run() or 0)


if __name__ == "__main__":
    raise SystemExit(main())

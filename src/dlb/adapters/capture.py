"""Run a pinned upstream entrypoint while capturing every returned token-ID batch."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

from dlb.io import atomic_json_write


def _split_arguments(argv: list[str]) -> tuple[Path, Path, list[str]]:
    try:
        separator = argv.index("--")
    except ValueError as error:
        raise ValueError("capture wrapper requires -- before upstream arguments") from error
    wrapper = argv[:separator]
    forwarded = argv[separator + 1 :]
    values: dict[str, str] = {}
    for argument in wrapper:
        key, marker, value = argument.partition("=")
        if not marker or key not in {"--upstream-entrypoint", "--capture-path"} or not value:
            raise ValueError(f"invalid capture-wrapper argument: {argument!r}")
        if key in values:
            raise ValueError(f"duplicate capture-wrapper argument: {key}")
        values[key] = value
    if set(values) != {"--upstream-entrypoint", "--capture-path"}:
        raise ValueError("capture wrapper requires entrypoint and capture path")
    entrypoint = Path(values["--upstream-entrypoint"])
    capture_path = Path(values["--capture-path"])
    if not entrypoint.is_absolute() or entrypoint.is_symlink() or not entrypoint.is_file():
        raise ValueError(f"upstream entrypoint is missing or unsafe: {entrypoint}")
    if not capture_path.is_absolute():
        raise ValueError("capture path must be absolute")
    return entrypoint, capture_path, forwarded


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


def main(argv: list[str] | None = None) -> int:
    entrypoint, capture_path, forwarded = _split_arguments(
        list(sys.argv[1:] if argv is None else argv)
    )
    module = _load_entrypoint(entrypoint)
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
                {
                    "sample_id": len(captured),
                    "text": text,
                    "token_ids": tokens,
                }
            )
        return result

    owner.restore_model_and_sample = capture
    sys.argv = [str(entrypoint), *forwarded]
    module.main()
    atomic_json_write(
        capture_path,
        {"schema": "dlb-upstream-token-capture-v1", "samples": captured},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

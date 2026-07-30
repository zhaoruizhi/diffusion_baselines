"""Server-only runtime shared by the SDTT and Di4C sampling entrypoints."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import sys
import tempfile


def require_file(path: Path, label: str) -> Path:
    path = path.absolute()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or unsafe: {path}")
    return path


def require_directory(path: Path, label: str) -> Path:
    path = path.absolute()
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} is missing or unsafe: {path}")
    return path


def parse_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError("boolean values must be 'true' or 'false'")


@contextmanager
def offline_huggingface(enabled: bool):
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    if enabled:
        os.environ.update({name: "1" for name in names})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def install_upstream(upstream_root: Path) -> None:
    source = require_directory(upstream_root / "src", "upstream Python source")
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy

        numpy.random.seed(seed % (2**32))
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(snapshot: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(require_directory(snapshot, "locked tokenizer snapshot")),
        local_files_only=True,
        trust_remote_code=False,
    )


def load_config(path: Path):
    from omegaconf import OmegaConf

    return OmegaConf.load(require_file(path, "model config"))


def checkpoint_state(path: Path) -> tuple[dict[str, object], object | None]:
    path = require_file(path, "student checkpoint")
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(path))), None
    import torch

    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("student checkpoint is not a mapping")
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise ValueError("student checkpoint state_dict is invalid")
    config = None
    hyperparameters = payload.get("hyper_parameters")
    if isinstance(hyperparameters, dict):
        config = hyperparameters.get("config")
    return dict(state), config


def instantiate_without_training_teacher(model_type, config, tokenizer):
    """Instantiate the released student architecture without loading a network teacher."""

    original = model_type.prepare_teacher_and_student

    def sampling_only(self):
        self.teacher = []

    model_type.prepare_teacher_and_student = sampling_only
    try:
        return model_type(config, tokenizer, verbose=False)
    finally:
        model_type.prepare_teacher_and_student = original


def normalize_state_dict(state: dict[str, object]) -> dict[str, object]:
    return {key.replace("_orig_mod.", ""): value for key, value in state.items()}


def write_capture_atomic(
    output: Path,
    *,
    model,
    tokenizer,
    sample_count: int,
    batch_size: int,
    num_steps: int,
    seq_len: int,
    sampler: str,
) -> None:
    """Sample exact batches, immediately move IDs to CPU, and atomically publish JSON."""

    if min(sample_count, batch_size, num_steps, seq_len) <= 0:
        raise ValueError("sampling counts and lengths must be positive")
    if sampler != "ancestral":
        raise ValueError("distilled language baselines require ancestral sampling")
    output = output.absolute()
    if not output.is_absolute() or output.is_symlink():
        raise ValueError(f"output path is unsafe: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    temporary = Path(temporary_name)
    written = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write('{"schema":"dlb-upstream-token-capture-v1","samples":[')
            while written < sample_count:
                current = min(batch_size, sample_count - written)
                result = model.sample(
                    n_samples=current,
                    num_steps=num_steps,
                    seq_len=seq_len,
                    sampler=sampler,
                    verbose=False,
                )
                rows = result.detach().cpu().tolist()
                if not isinstance(rows, list) or len(rows) != current:
                    raise ValueError("upstream sampler returned the wrong batch size")
                texts = list(tokenizer.batch_decode(rows))
                if len(texts) != current:
                    raise ValueError("tokenizer returned the wrong decoded batch size")
                for row, text in zip(rows, texts, strict=True):
                    if not isinstance(row, list) or len(row) != seq_len:
                        raise ValueError("upstream sampler returned a noncanonical sequence")
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError("upstream tokenizer returned empty text")
                    if written:
                        handle.write(",")
                    json.dump(
                        {"sample_id": written, "text": text, "token_ids": row},
                        handle,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    written += 1
                del result, rows, texts
            handle.write("]}")
            handle.flush()
            os.fsync(handle.fileno())
        if written != sample_count:
            raise ValueError(f"expected {sample_count} samples, wrote {written}")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def configure_for_sampling(config, *, tokenizer_snapshot: Path, seq_len: int) -> None:
    config.tokenizer.name = str(tokenizer_snapshot)
    config.model.length = seq_len
    config.data_preprocess.seq_len = seq_len
    config.mode = "sample"
    config.compile = False


def materialize_model(
    *,
    model_type,
    config,
    tokenizer,
    state: dict[str, object],
    strict: bool,
):
    model = instantiate_without_training_teacher(model_type, config, tokenizer)
    result = model.load_state_dict(normalize_state_dict(state), strict=strict)
    if not strict:
        missing = [key for key in result.missing_keys if not key.startswith("teacher.")]
        unexpected = [key for key in result.unexpected_keys if not key.startswith("teacher.")]
        if missing or unexpected:
            raise ValueError(
                "student checkpoint does not match the configured architecture: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
    model.eval()
    if not __import__("torch").cuda.is_available():
        raise RuntimeError("distilled sampling requires a CUDA server")
    model.cuda()
    return model

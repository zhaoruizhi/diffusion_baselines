"""Small, independent ELF experiment helpers (no GPU imports)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "b29d8833609e9ab7f67cd9da39435ac5cea04837"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def condition_token_ids(row, tokenizer):
    """Keep released IDs exactly; raw text follows the author's JSONL loader."""
    ids = row.get("condition_input_ids")
    if ids is None:
        ids = tokenizer(row["input"], add_special_tokens=False, verbose=False)["input_ids"]
    vocab_size = len(tokenizer)
    if not isinstance(ids, list) or any(type(t) is not int or t < 0 or t >= vocab_size for t in ids):
        raise ValueError(f"Invalid condition IDs at source_id={row.get('id')}")
    return ids


def preflight_conditions(rows, tokenizer, task, max_input_length, max_length):
    """Validate the whole requested input before loading weights; never drop rows."""
    prepared, failures = [], []
    for index, row in enumerate(rows):
        ids = condition_token_ids(row, tokenizer)
        if task == "owt-prefix":
            fits = 0 < len(ids) < max_length - 64
        else:
            ids = ids[:max_input_length]
            fits = bool(ids)
        if not fits:
            failures.append({"row_index": index, "source_id": row.get("id"), "tokens": len(ids)})
        prepared.append(ids)
    if failures:
        raise ValueError(
            f"Invalid/empty ELF conditions: {len(failures)}/{len(rows)}; "
            f"first affected rows (zero-based): {failures[:20]}. No samples were dropped. "
            "Run scripts/audit_elf_inputs.py to compare raw and official inputs; "
            "do not invent source text or delete rows to bypass this check."
        )
    return prepared


def require_server():
    if sys.platform != "linux":
        raise RuntimeError("ELF downloads and GPU experiments must run on the Linux server.")


def source_path(root):
    path = Path(root) / "upstreams/elf"
    commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True)
    if commit != SOURCE_COMMIT or dirty.strip():
        raise ValueError("ELF upstream must be clean and match the locked commit")
    return path


def default_sampling(task, steps):
    if steps < 1:
        raise ValueError("steps must be positive")
    if task in ("wmt14", "xsum"):
        return {"sampling_method": "ode", "cfg": 2.0, "sc_cfg": 1.0, "gamma": 0.0}
    # 8/16/32: paper Appendix D.2; 64: release configuration.
    # Other budgets are explicitly an extension of our benchmark.
    return {"sampling_method": "sde", "cfg": 1.0, "sc_cfg": 3.0,
            "gamma": 2.0 if steps <= 16 else 1.5 if steps == 32 else 1.0}


def manifest(root):
    path = Path(root) / "data/elf/assets.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"ELF asset manifest is missing: {path}. "
            "First complete `python scripts/prepare_elf.py assets` successfully; "
            "a failed download does not publish assets.json. Do not create it manually."
        )
    data = json.loads(path.read_text())
    if data["lock_sha256"] != sha256(Path(root) / "artifacts/elf_lock.json"):
        raise ValueError("ELF lock changed since asset preparation")
    return data


def asset(root, name):
    record = manifest(root)["assets"][name]
    directory = Path(root) / record["path"]
    for relative, digest in record["sha256"].items():
        if sha256(directory / relative) != digest:
            raise ValueError(f"ELF asset changed: {name}/{relative}")
    return directory


def offline():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"

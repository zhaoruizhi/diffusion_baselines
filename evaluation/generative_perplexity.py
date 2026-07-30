"""Corpus-token-weighted generative perplexity with an offline HF scorer."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import yaml


@dataclass(frozen=True)
class PPLResult:
    perplexity: float
    total_nll: float
    valid_token_count: int
    sample_count: int = 0
    batch_size: int = 0
    context_length: int = 1024
    truncation: str = "right"
    model_id: str = "gpt2-large"
    model_revision: str = ""
    tokenizer_id: str = "gpt2"
    tokenizer_revision: str = ""
    aggregation: str = "sum_nll_over_sum_valid_next_tokens"


@dataclass(frozen=True)
class GPT2Assets:
    model_id: str
    model_revision: str
    model_path: Path
    tokenizer_id: str
    tokenizer_revision: str
    tokenizer_path: Path


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _safe_snapshot(root: Path, relative: object, revision: str, name: str) -> Path:
    if type(relative) is not str or not relative:
        raise ValueError(f"{name} snapshot_path is invalid")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"{name} snapshot_path must be repository-relative")
    raw_candidate = root / relative_path
    cursor = root
    for part in relative_path.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{name} snapshot traverses a symlink: {cursor}")
    candidate = raw_candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} snapshot escapes repository root") from error
    if not candidate.is_dir():
        raise FileNotFoundError(f"{name} local snapshot is missing: {candidate}")
    if candidate.name != revision:
        raise ValueError(f"{name} snapshot directory does not match pinned revision")
    return candidate


def resolve_gpt2_assets(root: Path) -> GPT2Assets:
    """Resolve GPT-2 tokenizer/model snapshots without any Hub fallback."""

    root = root.resolve()
    configuration_path = root / "artifacts" / "data.yaml"
    downloads_path = root / "data" / "manifests" / "downloads.json"
    configuration = yaml.safe_load(configuration_path.read_text(encoding="utf-8"))
    downloads = json.loads(
        downloads_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
    )
    if not isinstance(downloads, dict) or downloads.get("schema_version") != 1:
        raise ValueError("downloads manifest schema_version must be 1")
    try:
        pinned = configuration["models"]
        downloaded = downloads["models"]
    except (KeyError, TypeError) as error:
        raise ValueError("data/download manifests do not contain model records") from error

    resolved: dict[str, tuple[str, Path]] = {}
    for name in ("gpt2-large", "gpt2"):
        try:
            revision = pinned[name]
            record = downloaded[name]
        except (KeyError, TypeError) as error:
            raise ValueError(f"missing pinned {name} snapshot record") from error
        if type(revision) is not str or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError(
                f"{name} pinned revision must be 40 lowercase hexadecimal characters"
            )
        if not isinstance(record, dict):
            raise ValueError(f"{name} download record is invalid")
        if record.get("repo_id") != name:
            raise ValueError(f"{name} repo_id does not match pinned model")
        if record.get("revision") != revision:
            raise ValueError(f"{name} revision does not match pinned configuration")
        resolved[name] = (
            revision,
            _safe_snapshot(root, record.get("snapshot_path"), revision, name),
        )
    return GPT2Assets(
        model_id="gpt2-large",
        model_revision=resolved["gpt2-large"][0],
        model_path=resolved["gpt2-large"][1],
        tokenizer_id="gpt2",
        tokenizer_revision=resolved["gpt2"][0],
        tokenizer_path=resolved["gpt2"][1],
    )


def load_offline_gpt2_large(
    assets: GPT2Assets, *, device: str = "cuda"
) -> tuple[object, object]:
    """Load only the already-resolved snapshots, with network access disabled."""

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(assets.tokenizer_path), local_files_only=True
    )
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("GPT-2 tokenizer has neither padding nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(assets.model_path), local_files_only=True
    )
    model = model.to(device).eval()
    return model, tokenizer


def aggregate_nll(parts: Iterable[tuple[float, int]]) -> PPLResult:
    """Exponentiate once after summing finite NLL/token-count parts."""

    total_nll = 0.0
    total_tokens = 0
    found = False
    for nll, count in parts:
        found = True
        if type(count) is not int or count <= 0:
            raise ValueError("valid next-token count must be a positive integer")
        if type(nll) not in (int, float) or not math.isfinite(float(nll)) or nll < 0:
            raise ValueError("NLL must be finite and non-negative")
        total_nll += float(nll)
        total_tokens += count
    if not found or total_tokens == 0:
        raise ValueError("no valid next-token targets were evaluated")
    if not math.isfinite(total_nll):
        raise ValueError("total NLL is not finite")
    try:
        perplexity = math.exp(total_nll / total_tokens)
    except OverflowError as error:
        raise ValueError("perplexity is not finite") from error
    if not math.isfinite(perplexity):
        raise ValueError("perplexity is not finite")
    return PPLResult(
        perplexity=perplexity,
        total_nll=total_nll,
        valid_token_count=total_tokens,
    )


class TorchCausalLMScorer:
    """Efficiently reduce causal-LM logits without transferring them to the CPU."""

    def __init__(self, model: object, *, device: str | None = None) -> None:
        self.model = model
        self.device = device

    def eval(self) -> "TorchCausalLMScorer":
        self.model.eval()  # type: ignore[attr-defined]
        return self

    def score_batch(
        self, input_ids: Sequence[Sequence[int]], attention_mask: Sequence[Sequence[int]]
    ) -> tuple[float, int]:
        import torch
        import torch.nn.functional as functional

        device = self.device
        if device is None:
            try:
                device = str(next(self.model.parameters()).device)  # type: ignore[attr-defined]
            except StopIteration:
                device = "cpu"
        ids = torch.as_tensor(input_ids, dtype=torch.long, device=device)
        mask = torch.as_tensor(attention_mask, dtype=torch.bool, device=device)
        with torch.inference_mode():
            outputs = self.model(input_ids=ids, attention_mask=mask)  # type: ignore[operator]
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = ids[:, 1:].clone()
            valid = mask[:, 1:]
            shifted_labels.masked_fill_(~valid, -100)
            nll = functional.cross_entropy(
                shifted_logits.reshape(-1, shifted_logits.size(-1)),
                shifted_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
        return float(nll.item()), int(valid.sum().item())


def _rows(value: object, key: str) -> list[list[int]]:
    try:
        rows = value[key]  # type: ignore[index]
    except (KeyError, TypeError) as error:
        raise ValueError(f"tokenizer result is missing {key}") from error
    if not isinstance(rows, (list, tuple)):
        raise ValueError(f"tokenizer {key} must be a batch of rows")
    normalized: list[list[int]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or not all(type(item) is int for item in row):
            raise ValueError(f"tokenizer {key} row {index} is invalid")
        normalized.append(list(row))
    return normalized


def compute_gen_ppl(
    texts: Sequence[str],
    model: object,
    tokenizer: object,
    *,
    batch_size: int = 8,
    max_length: int = 1024,
    model_revision: str = "",
    tokenizer_revision: str = "",
    device: str | None = None,
) -> PPLResult:
    """Retokenize texts and compute corpus-level GPT-2 causal perplexity."""

    if not texts or any(type(text) is not str or not text.strip() for text in texts):
        raise ValueError("texts must contain non-empty strings")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if type(max_length) is not int or max_length < 2:
        raise ValueError("max_length must be at least two")
    if max_length > 1024:
        raise ValueError("GPT-2 context length must not exceed 1024")
    scorer = (
        model
        if callable(getattr(model, "score_batch", None))
        else TorchCausalLMScorer(model, device=device)
    )
    scorer.eval()  # type: ignore[attr-defined]
    parts: list[tuple[float, int]] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(  # type: ignore[operator]
            batch,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        input_ids = _rows(encoded, "input_ids")
        attention_mask = _rows(encoded, "attention_mask")
        if len(input_ids) != len(batch) or len(attention_mask) != len(batch):
            raise ValueError("tokenizer silently changed the batch size")
        if any(len(ids) != len(mask) for ids, mask in zip(input_ids, attention_mask, strict=True)):
            raise ValueError("tokenizer input and attention-mask shapes differ")
        if any(value not in (0, 1) for row in attention_mask for value in row):
            raise ValueError("tokenizer attention mask must contain only zero or one")
        expected_count = 0
        for batch_index, row in enumerate(attention_mask):
            valid_length = sum(row)
            if row != [1] * valid_length + [0] * (len(row) - valid_length):
                raise ValueError("tokenizer attention mask must be right-padded")
            if valid_length < 2:
                raise ValueError(
                    f"sample {start + batch_index} has no valid next-token target"
                )
            expected_count += valid_length - 1
        nll, count = scorer.score_batch(input_ids, attention_mask)  # type: ignore[attr-defined]
        if count != expected_count:
            raise ValueError(
                "scorer valid-token count differs from attention mask: "
                f"expected {expected_count}, reported {count}"
            )
        if count:
            parts.append((nll, count))
        elif nll != 0:
            raise ValueError("scorer returned NLL without valid next-token targets")
    aggregated = aggregate_nll(parts)
    return PPLResult(
        perplexity=aggregated.perplexity,
        total_nll=aggregated.total_nll,
        valid_token_count=aggregated.valid_token_count,
        sample_count=len(texts),
        batch_size=batch_size,
        context_length=max_length,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
    )

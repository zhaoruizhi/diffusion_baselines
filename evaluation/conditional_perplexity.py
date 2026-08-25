"""Prompt-excluded GPT-2 perplexity for fixed-prefix conditional samples."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Sequence

import yaml

from dlb.schema import ConditionalSampleRecord

from .generative_perplexity import PPLResult, _rows, _safe_snapshot, aggregate_nll


@dataclass(frozen=True)
class TokenizerAssets:
    tokenizer_id: str
    tokenizer_revision: str
    tokenizer_path: Path


@dataclass(frozen=True)
class ConditionalText:
    prefix: str
    generated_suffix: str
    reference_suffix: str
    prefix_and_generated_suffix: str


class PromptExcludedCausalLMScorer:
    """Reduce causal-LM logits only at target positions after the prompt."""

    def __init__(self, model: object, *, device: str | None = None) -> None:
        self.model = model
        self.device = device

    def eval(self) -> "PromptExcludedCausalLMScorer":
        self.model.eval()  # type: ignore[attr-defined]
        return self

    def score_prompt_excluded_batch(
        self,
        input_ids: Sequence[Sequence[int]],
        attention_mask: Sequence[Sequence[int]],
        target_mask: Sequence[Sequence[int]],
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
        targets = torch.as_tensor(target_mask, dtype=torch.bool, device=device)
        with torch.inference_mode():
            outputs = self.model(input_ids=ids, attention_mask=mask)  # type: ignore[operator]
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = ids[:, 1:].clone()
            valid = mask[:, 1:] & targets[:, 1:]
            shifted_labels.masked_fill_(~valid, -100)
            nll = functional.cross_entropy(
                shifted_logits.reshape(-1, shifted_logits.size(-1)),
                shifted_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
        return float(nll.item()), int(valid.sum().item())


def resolve_dataset_tokenizer_assets(root: Path, dataset: str) -> TokenizerAssets:
    """Resolve the already-downloaded source tokenizer used by a dataset."""

    root = root.resolve()
    configuration_path = root / "artifacts" / "data.yaml"
    downloads_path = root / "data" / "manifests" / "downloads.json"
    configuration = yaml.safe_load(configuration_path.read_text(encoding="utf-8"))
    downloads = json.loads(downloads_path.read_text(encoding="utf-8"))
    if not isinstance(downloads, dict) or downloads.get("schema_version") != 1:
        raise ValueError("downloads manifest schema_version must be 1")
    try:
        tokenizer_id = configuration["datasets"][dataset]["tokenizer"]
        revision = configuration["models"][tokenizer_id]
        record = downloads["models"][tokenizer_id]
    except (KeyError, TypeError) as error:
        raise ValueError(f"missing tokenizer binding for dataset {dataset!r}") from error
    if not isinstance(tokenizer_id, str) or not isinstance(revision, str):
        raise ValueError("dataset tokenizer binding is invalid")
    if record.get("repo_id") != tokenizer_id or record.get("revision") != revision:
        raise ValueError("dataset tokenizer download record differs from data config")
    return TokenizerAssets(
        tokenizer_id=tokenizer_id,
        tokenizer_revision=revision,
        tokenizer_path=_safe_snapshot(root, record.get("snapshot_path"), revision, tokenizer_id),
    )


def load_offline_tokenizer(assets: TokenizerAssets) -> object:
    """Load one local tokenizer snapshot without any Hub fallback."""

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(assets.tokenizer_path), local_files_only=True)


def _decode(tokenizer: object, token_ids: Sequence[int]) -> str:
    text = tokenizer.decode(  # type: ignore[operator]
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=True,
    )
    if not isinstance(text, str) or not text.strip():
        raise ValueError("conditional decoded text is empty")
    return text


def conditional_texts(
    records: Sequence[ConditionalSampleRecord],
    dataset_tokenizer: object,
    *,
    continuation_length: int = 64,
) -> list[ConditionalText]:
    """Decode exactly prefix plus the first evaluation continuation tokens."""

    if not records:
        raise ValueError("conditional records must not be empty")
    result: list[ConditionalText] = []
    for index, record in enumerate(records):
        generated = record.continuation_token_ids[:continuation_length]
        if len(generated) != continuation_length:
            raise ValueError(f"record {index} has too few generated continuation tokens")
        if len(record.reference_token_ids) != continuation_length:
            raise ValueError(f"record {index} has the wrong reference continuation length")
        prefix = list(record.prefix_token_ids)
        result.append(
            ConditionalText(
                prefix=_decode(dataset_tokenizer, prefix),
                generated_suffix=_decode(dataset_tokenizer, generated),
                reference_suffix=_decode(dataset_tokenizer, record.reference_token_ids),
                prefix_and_generated_suffix=_decode(dataset_tokenizer, [*prefix, *generated]),
            )
        )
    return result


def _right_padded(mask: list[int]) -> bool:
    valid = sum(mask)
    return mask == [1] * valid + [0] * (len(mask) - valid)


def compute_conditional_gen_ppl(
    records: Sequence[ConditionalSampleRecord],
    model: object,
    tokenizer: object,
    dataset_tokenizer: object,
    *,
    batch_size: int = 8,
    max_length: int = 1024,
    continuation_length: int = 64,
    model_revision: str = "",
    tokenizer_revision: str = "",
    device: str | None = None,
) -> PPLResult:
    """Score prefix+continuation text while excluding prompt tokens from NLL."""

    texts = conditional_texts(
        records,
        dataset_tokenizer,
        continuation_length=continuation_length,
    )
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if type(max_length) is not int or max_length < 2 or max_length > 1024:
        raise ValueError("max_length must be in [2, 1024]")
    scorer = (
        model
        if callable(getattr(model, "score_prompt_excluded_batch", None))
        else PromptExcludedCausalLMScorer(model, device=device)
    )
    scorer.eval()  # type: ignore[attr-defined]
    parts: list[tuple[float, int]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        prefix_encoded = tokenizer(  # type: ignore[operator]
            [item.prefix for item in batch],
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        full_encoded = tokenizer(  # type: ignore[operator]
            [item.prefix_and_generated_suffix for item in batch],
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        input_ids = _rows(full_encoded, "input_ids")
        attention_mask = _rows(full_encoded, "attention_mask")
        prefix_masks = _rows(prefix_encoded, "attention_mask")
        if len(input_ids) != len(batch) or len(attention_mask) != len(batch):
            raise ValueError("tokenizer silently changed the conditional batch size")
        target_masks: list[list[int]] = []
        expected = 0
        for batch_index, (ids, mask, prefix_mask) in enumerate(
            zip(input_ids, attention_mask, prefix_masks, strict=True)
        ):
            if len(ids) != len(mask):
                raise ValueError("tokenizer input and attention-mask shapes differ")
            if any(value not in (0, 1) for value in mask) or not _right_padded(mask):
                raise ValueError("tokenizer attention mask must be right-padded")
            prefix_length = sum(prefix_mask)
            valid_length = sum(mask)
            if prefix_length <= 0 or prefix_length >= valid_length:
                raise ValueError(
                    f"sample {start + batch_index} has no prompt-excluded targets"
                )
            row = [0] * len(mask)
            for index in range(prefix_length, valid_length):
                row[index] = 1
            target_masks.append(row)
            expected += sum(row[1:])
        nll, count = scorer.score_prompt_excluded_batch(  # type: ignore[attr-defined]
            input_ids,
            attention_mask,
            target_masks,
        )
        if count != expected:
            raise ValueError(
                "scorer valid-token count differs from prompt-excluded mask: "
                f"expected {expected}, reported {count}"
            )
        parts.append((nll, count))
    aggregated = aggregate_nll(parts)
    return PPLResult(
        perplexity=aggregated.perplexity,
        total_nll=aggregated.total_nll,
        valid_token_count=aggregated.valid_token_count,
        sample_count=len(records),
        batch_size=batch_size,
        context_length=max_length,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
    )

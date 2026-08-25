"""CLI for evaluating fixed-prefix conditional generation artifacts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Sequence

from dlb.io import (
    atomic_json_write,
    expected_conditional_schedule,
    read_conditional_samples,
    sha256_file,
    validate_conditional_samples,
)
from dlb.schema import ConditionalSampleRecord

from .conditional_perplexity import (
    TokenizerAssets,
    conditional_texts,
    compute_conditional_gen_ppl,
    load_offline_tokenizer,
    resolve_dataset_tokenizer_assets,
)
from .generative_perplexity import (
    load_offline_gpt2_large,
    resolve_gpt2_assets,
)
from .self_bleu import SelfBleuConfig, compute_self_bleu
from .unigram_entropy import mean_unigram_entropy


PRODUCTION_SAMPLE_COUNT = 2048
EVALUATION_CONTINUATION_LENGTH = 64
DATASET_CONTRACTS = {
    "lm1b": {"sequence_length": 128, "vocab_size": 30_522, "padding_ids": frozenset({0})},
    "owt": {"sequence_length": 1024, "vocab_size": 50_257, "padding_ids": frozenset()},
}
METRIC_NAMES = (
    "conditional_gen_ppl",
    "mauve_suffix",
    "entropy",
    "self_bleu",
    "prefix_exact_match",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--metrics", default=",".join(METRIC_NAMES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", choices=tuple(DATASET_CONTRACTS), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mauve-device-id", type=int, default=0)
    return parser.parse_args(argv)


def _metrics(value: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if not names:
        raise ValueError("at least one metric is required")
    unknown = sorted(set(names) - set(METRIC_NAMES))
    if unknown:
        raise ValueError(f"unknown metrics: {', '.join(unknown)}")
    if len(set(names)) != len(names):
        raise ValueError("metric names must not be duplicated")
    return names


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_records(path: Path) -> list[ConditionalSampleRecord]:
    return list(read_conditional_samples(path))


def _suffix_token_records(
    records: Sequence[ConditionalSampleRecord], *, reference: bool
) -> list[dict[str, object]]:
    key = "reference_token_ids" if reference else "continuation_token_ids"
    rows = []
    for record in records:
        tokens = list(getattr(record, key))[:EVALUATION_CONTINUATION_LENGTH]
        rows.append({"token_ids": tokens})
    return rows


def _compute_mauve(
    generated: Sequence[str],
    reference: Sequence[str],
    *,
    device_id: int,
) -> dict[str, object]:
    try:
        import mauve
    except ImportError as error:
        raise RuntimeError("mauve is required for conditional MAUVE evaluation") from error
    result = mauve.compute_mauve(
        p_text=list(generated),
        q_text=list(reference),
        device_id=device_id,
        verbose=False,
    )
    score = float(result.mauve)
    if not math.isfinite(score) or score < 0 or score > 1:
        raise ValueError("MAUVE score is outside [0, 1]")
    return {
        "score": score,
        "sample_count": len(generated),
        "comparison": "generated_suffix_vs_reference_suffix",
        "slice": "source_tokens_64_128_only",
        "prefix_included": False,
    }


def _gpt2_tokenized_self_bleu(
    records: Sequence[ConditionalSampleRecord],
    suffix_texts: Sequence[str],
    tokenizer: object,
) -> dict[str, object]:
    groups: dict[int, list[str]] = {}
    for record, suffix in zip(records, suffix_texts, strict=True):
        if record.prompt_id >= 256:
            continue
        encoded = tokenizer(  # type: ignore[operator]
            suffix,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        token_ids = encoded["input_ids"]
        if not isinstance(token_ids, list) or not all(type(token) is int for token in token_ids):
            raise ValueError("GPT-2 tokenizer returned invalid Self-BLEU tokens")
        groups.setdefault(record.prompt_id, []).append(" ".join(map(str, token_ids)))
    if sorted(groups) != list(range(256)) or any(len(group) != 5 for group in groups.values()):
        raise ValueError("conditional Self-BLEU requires the canonical 256 prompts x 5 completions")
    scores = [
        compute_self_bleu(groups[prompt_id], SelfBleuConfig()).score
        for prompt_id in range(256)
    ]
    return {
        "score": math.fsum(scores) / len(scores),
        "prompt_count": 256,
        "completions_per_prompt": 5,
        "sample_count": 256 * 5,
        "tokenization": "gpt2",
        "ngram_order": 4,
        "weights": (0.25, 0.25, 0.25, 0.25),
        "smoothing": "chen_cherry_method1",
        "aggregation": "mean_prompt_group_self_bleu",
        "slice": "generated_source_tokens_64_128_only",
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    root = args.root.resolve()
    metric_names = _metrics(args.metrics)
    contract = DATASET_CONTRACTS[args.dataset]
    sample_count = validate_conditional_samples(
        args.samples,
        expected=PRODUCTION_SAMPLE_COUNT,
        schedule=expected_conditional_schedule(),
        sequence_length=int(contract["sequence_length"]),
        vocab_size=int(contract["vocab_size"]),
    )
    records = _load_records(args.samples)
    if sample_count != len(records):
        raise ValueError("conditional sample count changed while reading records")
    result_metrics: dict[str, object] = {}
    dataset_tokenizer = None
    decoded = None

    if any(name in metric_names for name in ("conditional_gen_ppl", "mauve_suffix", "self_bleu")):
        dataset_assets = resolve_dataset_tokenizer_assets(root, args.dataset)
        dataset_tokenizer = load_offline_tokenizer(dataset_assets)
        decoded = conditional_texts(
            records,
            dataset_tokenizer,
            continuation_length=EVALUATION_CONTINUATION_LENGTH,
        )

    if "conditional_gen_ppl" in metric_names:
        assets = resolve_gpt2_assets(root)
        model, gpt2_tokenizer = load_offline_gpt2_large(assets, device=args.device)
        ppl = compute_conditional_gen_ppl(
            records,
            model,
            gpt2_tokenizer,
            dataset_tokenizer,
            batch_size=args.batch_size,
            max_length=args.context_length,
            continuation_length=EVALUATION_CONTINUATION_LENGTH,
            model_revision=assets.model_revision,
            tokenizer_revision=assets.tokenizer_revision,
            device=args.device,
        )
        result_metrics["conditional_generative_perplexity"] = {
            **asdict(ppl),
            "model_snapshot": _display_path(assets.model_path, root),
            "tokenizer_snapshot": _display_path(assets.tokenizer_path, root),
            "prompt_loss_excluded": True,
            "scored_slice": "source_tokens_64_128_only",
            "offline": True,
        }

    if "mauve_suffix" in metric_names:
        assert decoded is not None
        result_metrics["mauve_suffix"] = _compute_mauve(
            [item.generated_suffix for item in decoded],
            [item.reference_suffix for item in decoded],
            device_id=args.mauve_device_id,
        )

    if "entropy" in metric_names:
        generated_entropy = mean_unigram_entropy(
            _suffix_token_records(records, reference=False),
            contract["padding_ids"],
        )
        reference_entropy = mean_unigram_entropy(
            _suffix_token_records(records, reference=True),
            contract["padding_ids"],
        )
        result_metrics["sample_entropy"] = {
            "generated": asdict(generated_entropy),
            "reference": asdict(reference_entropy),
            "generated_minus_reference": (
                generated_entropy.mean_entropy - reference_entropy.mean_entropy
            ),
            "slice": "source_tokens_64_128_only",
        }

    if "self_bleu" in metric_names:
        assert decoded is not None
        assets = resolve_gpt2_assets(root)
        gpt2_tokenizer = load_offline_tokenizer(
            TokenizerAssets(
                tokenizer_id=assets.tokenizer_id,
                tokenizer_revision=assets.tokenizer_revision,
                tokenizer_path=assets.tokenizer_path,
            )
        )
        result_metrics["self_bleu"] = _gpt2_tokenized_self_bleu(
            records,
            [item.generated_suffix for item in decoded],
            gpt2_tokenizer,
        )

    if "prefix_exact_match" in metric_names:
        exact = sum(1 for record in records if record.prefix_exact_match is True)
        result_metrics["prefix_exact_match"] = {
            "rate": exact / len(records),
            "matched": exact,
            "sample_count": len(records),
            "required_rate": 1.0,
        }

    return {
        "schema_version": 1,
        "protocol": "c64_zs_v1",
        "samples": _display_path(args.samples, root),
        "samples_sha256": sha256_file(args.samples),
        "sample_count": sample_count,
        "production_sample_count": PRODUCTION_SAMPLE_COUNT,
        "dataset": args.dataset,
        "prefix_length": 64,
        "evaluation_continuation_length": EVALUATION_CONTINUATION_LENGTH,
        "requested_metrics": list(metric_names),
        "metrics": result_metrics,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    document = evaluate(args)
    atomic_json_write(args.output, document)
    print(f"WROTE {args.output} samples={document['sample_count']} conditional=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

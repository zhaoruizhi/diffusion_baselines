"""CLI for atomically evaluating canonical sample JSONL artifacts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

from dlb.io import atomic_json_write, sha256_file, validate_samples
from dlb.schema import SampleRecord

from .generative_perplexity import (
    compute_gen_ppl,
    load_offline_gpt2_large,
    resolve_gpt2_assets,
)
from .self_bleu import SelfBleuConfig, compute_self_bleu
from .unigram_entropy import mean_unigram_entropy


PRODUCTION_SAMPLE_COUNT = 1024
METRIC_NAMES = ("gen_ppl", "entropy", "self_bleu")
DATASET_PADDING_IDS = {"lm1b": frozenset({0}), "owt": frozenset()}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--metrics", default=",".join(METRIC_NAMES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", choices=tuple(DATASET_PADDING_IDS))
    parser.add_argument("--special-id", type=int, action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="allow a non-production sample count and label the artifact partial",
    )
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


def _load_records(path: Path) -> list[SampleRecord]:
    records = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            records.append(SampleRecord.model_validate(json.loads(line)))
    return records


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    root = args.root.resolve()
    metric_names = _metrics(args.metrics)
    expected = None if args.allow_partial else PRODUCTION_SAMPLE_COUNT
    sample_count = validate_samples(args.samples, expected=expected)
    if sample_count < 1:
        raise ValueError("sample artifact is empty")
    partial = sample_count != PRODUCTION_SAMPLE_COUNT
    if partial and not args.allow_partial:
        raise ValueError("partial input requires --allow-partial")
    if not partial and "entropy" in metric_names and args.dataset is None:
        raise ValueError("production entropy evaluation requires --dataset")
    records = _load_records(args.samples)
    texts = [record.text for record in records]
    result_metrics: dict[str, object] = {}

    if "gen_ppl" in metric_names:
        assets = resolve_gpt2_assets(root)
        model, tokenizer = load_offline_gpt2_large(assets, device=args.device)
        ppl = compute_gen_ppl(
            texts,
            model,
            tokenizer,
            batch_size=args.batch_size,
            max_length=args.context_length,
            model_revision=assets.model_revision,
            tokenizer_revision=assets.tokenizer_revision,
            device=args.device,
        )
        result_metrics["generative_perplexity"] = {
            **asdict(ppl),
            "model_snapshot": _display_path(assets.model_path, root),
            "tokenizer_snapshot": _display_path(assets.tokenizer_path, root),
            "offline": True,
        }

    excluded = set(args.special_id)
    exclusion_source = "explicit_cli"
    if args.dataset is not None:
        excluded.update(DATASET_PADDING_IDS[args.dataset])
        exclusion_source = "dataset_contract_plus_explicit_cli"
    if "entropy" in metric_names:
        entropy = mean_unigram_entropy(records, excluded)
        result_metrics["unigram_entropy"] = {
            **asdict(entropy),
            "special_token_policy": "exclude_documented_padding_only_preserve_bos_eos",
            "exclusion_source": exclusion_source,
        }

    if "self_bleu" in metric_names:
        result_metrics["self_bleu"] = asdict(
            compute_self_bleu(texts, SelfBleuConfig())
        )

    return {
        "schema_version": 1,
        "samples": _display_path(args.samples, root),
        "samples_sha256": sha256_file(args.samples),
        "sample_count": sample_count,
        "production_sample_count": PRODUCTION_SAMPLE_COUNT,
        "partial": partial,
        "dataset": args.dataset,
        "requested_metrics": list(metric_names),
        "metrics": result_metrics,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    document = evaluate(args)
    atomic_json_write(args.output, document)
    print(
        f"WROTE {args.output} samples={document['sample_count']} "
        f"partial={str(document['partial']).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

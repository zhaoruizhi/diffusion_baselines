#!/usr/bin/env python3
"""Sample a pinned Di4C language student from local server assets only."""

from __future__ import annotations

import argparse
from pathlib import Path

from _distilled_runtime import (
    benchmark_model,
    checkpoint_state,
    configure_for_sampling,
    install_upstream,
    load_config,
    load_tokenizer,
    load_tokenizer_binding,
    materialize_model,
    offline_huggingface,
    parse_bool,
    require_directory,
    require_file,
    seed_everything,
    validate_embedded_config,
    validate_sampling_config,
    write_capture_atomic,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--downloads-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-steps", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sampler", choices=("ancestral",), required=True)
    parser.add_argument(
        "--teacher-family", choices=("masked_mdlm", "uniform_duo"), required=True
    )
    parser.add_argument("--dataset", choices=("lm1b", "owt"), required=True)
    parser.add_argument("--generation-mode", choices=("unconditional", "conditional_prefix"), default="unconditional")
    parser.add_argument("--conditioning-manifest", type=Path)
    parser.add_argument("--conditioning-manifest-sha256")
    parser.add_argument("--conditioning-config-sha256")
    parser.add_argument("--prefix-length", type=int)
    parser.add_argument("--evaluation-continuation-length", type=int)
    parser.add_argument("--prompt-count", type=int)
    parser.add_argument("--diversity-prompt-count", type=int)
    parser.add_argument("--completions-per-diversity-prompt", type=int)
    parser.add_argument("--completion-schedule")
    parser.add_argument("--offline", type=parse_bool, required=True)
    parser.add_argument(
        "--allow-missing-embedded-config", type=parse_bool, default=False
    )
    parser.add_argument("--benchmark-output", type=Path)
    parser.add_argument("--benchmark-metadata", type=Path)
    parser.add_argument("--benchmark-precision", choices=("author",))
    args = parser.parse_args(argv)
    benchmark_values = (args.benchmark_output, args.benchmark_metadata, args.benchmark_precision)
    if any(value is not None for value in benchmark_values) and not all(
        value is not None for value in benchmark_values
    ):
        parser.error("benchmark arguments must be provided together")

    upstream = require_directory(args.upstream_root, "pinned Di4C language source")
    require_file(upstream / "src/sdtt/main.py", "pinned Di4C entrypoint")
    install_upstream(upstream)
    checkpoint = require_file(args.checkpoint, "Di4C student checkpoint")
    config_path = require_file(args.config, "Di4C sampling config")
    binding = load_tokenizer_binding(
        args.data_config,
        args.downloads_manifest,
        args.dataset,
        args.tokenizer_snapshot,
    )
    with offline_huggingface(args.offline):
        from sdtt.core.distill.multi_round_sdtt import MultiRoundSDTT

        config = load_config(config_path, args.config_sha256)
        state, embedded_config = checkpoint_state(
            checkpoint, args.checkpoint_sha256
        )
        validate_sampling_config(
            config,
            binding=binding,
            sequence_length=args.seq_len,
            require_di4c=True,
        )
        validate_embedded_config(
            config,
            embedded_config,
            allow_missing_embedded_config=args.allow_missing_embedded_config,
        )
        configure_for_sampling(
            config, tokenizer_snapshot=binding.snapshot, seq_len=args.seq_len
        )
        config.parameterization.checkpoint_path = str(checkpoint)
        tokenizer = load_tokenizer(binding.snapshot)
        seed_everything(args.seed)
        model = materialize_model(
            model_type=MultiRoundSDTT,
            config=config,
            tokenizer=tokenizer,
            state=state,
            strict=False,
        )
        if args.benchmark_output is not None:
            benchmark_model(
                model=model,
                output=args.benchmark_output,
                metadata_path=args.benchmark_metadata,
                precision=args.benchmark_precision,
                num_steps=args.num_steps,
                seq_len=args.seq_len,
                sampler=args.sampler,
                generation_mode=args.generation_mode,
                conditioning_manifest=args.conditioning_manifest,
                conditioning_manifest_sha256=args.conditioning_manifest_sha256,
                prompt_count=args.prompt_count,
                diversity_prompt_count=args.diversity_prompt_count,
                completions_per_diversity_prompt=args.completions_per_diversity_prompt,
            )
        else:
            write_capture_atomic(
                args.output,
                model=model,
                tokenizer=tokenizer,
                sample_count=args.sample_count,
                batch_size=args.batch_size,
                num_steps=args.num_steps,
                seq_len=args.seq_len,
                sampler=args.sampler,
                generation_mode=args.generation_mode,
                conditioning_manifest=args.conditioning_manifest,
                conditioning_manifest_sha256=args.conditioning_manifest_sha256,
                prompt_count=args.prompt_count,
                diversity_prompt_count=args.diversity_prompt_count,
                completions_per_diversity_prompt=args.completions_per_diversity_prompt,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

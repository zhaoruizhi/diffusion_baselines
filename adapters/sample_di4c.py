#!/usr/bin/env python3
"""Sample a pinned Di4C language student from local server assets only."""

from __future__ import annotations

import argparse
from pathlib import Path

from _distilled_runtime import (
    checkpoint_state,
    configure_for_sampling,
    install_upstream,
    load_config,
    load_tokenizer,
    materialize_model,
    offline_huggingface,
    parse_bool,
    require_directory,
    require_file,
    seed_everything,
    write_capture_atomic,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    parser.add_argument("--offline", type=parse_bool, required=True)
    args = parser.parse_args(argv)

    upstream = require_directory(args.upstream_root, "pinned Di4C language source")
    entrypoint = require_file(upstream / "src/sdtt/main.py", "pinned Di4C entrypoint")
    install_upstream(upstream)
    checkpoint = require_file(args.checkpoint, "Di4C student checkpoint")
    tokenizer_snapshot = require_directory(args.tokenizer_snapshot, "locked tokenizer")
    with offline_huggingface(args.offline):
        from omegaconf import OmegaConf
        from sdtt.core.distill.multi_round_sdtt import MultiRoundSDTT

        state, embedded_config = checkpoint_state(checkpoint)
        config = (
            OmegaConf.create(embedded_config)
            if embedded_config is not None
            else load_config(entrypoint.parent / "configs/config.yaml")
        )
        configure_for_sampling(
            config, tokenizer_snapshot=tokenizer_snapshot, seq_len=args.seq_len
        )
        config.is_di4c = True
        config.is_teacher_di4c = True
        config.parameterization.start_from_hf = False
        config.parameterization.checkpoint_path = str(checkpoint)
        tokenizer = load_tokenizer(tokenizer_snapshot)
        seed_everything(args.seed)
        model = materialize_model(
            model_type=MultiRoundSDTT,
            config=config,
            tokenizer=tokenizer,
            state=state,
            strict=False,
        )
        write_capture_atomic(
            args.output,
            model=model,
            tokenizer=tokenizer,
            sample_count=args.sample_count,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            seq_len=args.seq_len,
            sampler=args.sampler,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

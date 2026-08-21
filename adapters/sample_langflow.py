#!/usr/bin/env python3
"""Dataset-configurable LangFlow sampler used by the DLB capture wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = ROOT / "upstreams" / "langflow"
if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from langflow import LangFlow, LangFlowConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate samples with pinned LangFlow")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--seq_length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")
    parser.add_argument("--tokenizer", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    config_dir = checkpoint.parent

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = LangFlowConfig.from_pretrained(str(config_dir))
    model = LangFlow(config)
    state_dict = load_file(str(checkpoint), device=str(device))
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    all_samples = []
    remaining = args.num_samples
    with torch.no_grad():
        while remaining > 0:
            batch = min(args.batch_size, remaining)
            samples = model.generate_samples(
                num_samples=batch,
                seq_length=args.seq_length,
                num_steps=args.num_steps,
                device=device,
            )
            all_samples.append(samples)
            remaining -= batch

    generated = torch.cat(all_samples, dim=0)
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    for index, text in enumerate(texts):
        print(f"\n--- Sample {index + 1} ---")
        print(text[:500] + ("..." if len(text) > 500 else ""))

    if args.output is not None:
        with open(args.output, "w", encoding="utf-8") as output_file:
            for index, text in enumerate(texts):
                output_file.write(f"--- Sample {index + 1} ---\n{text}\n\n")


if __name__ == "__main__":
    main()

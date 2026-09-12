#!/usr/bin/env python3
"""Independent, offline ELF-B generation and synchronized CUDA timing.

Uses the locked official model and sampler without editing upstream sources.
Outputs intentionally do not claim compatibility with the legacy DLB registry.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

from elf_common import (ROOT, SOURCE_COMMIT, asset, default_sampling, manifest, offline,
                        preflight_conditions, read_jsonl, require_server, sha256, source_path, write_json)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["generate", "timing"])
    p.add_argument("--task", choices=["owt", "owt-prefix", "wmt14", "xsum"], required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--input", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-samples", type=int, default=1024,
                   help="Prompt count; 0 means the entire conditional input file")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampler", choices=["ode", "sde"])
    p.add_argument("--cfg", type=float)
    p.add_argument("--sc-cfg", type=float)
    p.add_argument("--gamma", type=float)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--diversity", action="store_true", help="C64: four extra completions for first 256 prompts")
    p.add_argument("--timing-prompt-index", type=int, default=0)
    p.add_argument("--warmups", type=int, default=5)
    p.add_argument("--repeats", type=int, default=32)
    args = p.parse_args(argv)
    if args.steps < 1 or args.batch_size < 1 or args.num_samples < 0:
        p.error("steps/batch-size must be positive and num-samples nonnegative")
    if (args.task != "owt") != (args.input is not None):
        p.error("--input is required exactly for conditional tasks")
    if args.task == "owt" and args.num_samples == 0:
        p.error("unconditional generation needs an explicit positive sample count")
    if args.diversity and (args.task != "owt-prefix" or args.mode != "generate"):
        p.error("--diversity is only for owt-prefix generation")
    if args.mode == "timing" and args.batch_size != 1:
        p.error("primary timing requires --batch-size 1")
    if args.warmups < 1 or args.repeats < 2:
        p.error("use at least one warmup and two measured repeats")
    return args


def main():
    args = parse_args()
    require_server()
    offline()
    root = args.root.resolve()
    upstream = source_path(root)
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(upstream / "src"), str(root / "src")]
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from configs.config import load_config_from_yaml, SamplingConfig
    from modules.model import ELF_models
    from modules.t5_encoder import get_encoder
    from utils.encoder_utils import encode_text, build_self_attn_cond_masks
    from utils.generation_utils import _generate_samples_single_batch, _dlm_decode_batch, shift_left, mask_after_eos
    from utils.sampling_utils import get_sampling_steps
    from dlb.timing import benchmark, cuda_runtime_metadata

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU supporting bf16 is required for the author precision policy")
    if args.output.exists():
        raise FileExistsError(f"Use a fresh output directory: {args.output}")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    checkpoint_task = "owt" if args.task == "owt-prefix" else args.task
    config_task = "de-en" if checkpoint_task == "wmt14" else checkpoint_task
    config_path = upstream / f"src/configs/training_configs/train_{config_task}_ELF-B.yml"
    # The official loader resolves sampling YAML relative to upstream cwd.
    import os
    original_cwd = Path.cwd()
    os.chdir(upstream)
    try:
        config = load_config_from_yaml(str(config_path))
    finally:
        os.chdir(original_cwd)
    config.use_bf16 = True
    config.use_compile = args.compile
    config.use_wandb = False
    config.hf_repo_id = None
    chosen = default_sampling(args.task, args.steps)
    for key, value in (("sampling_method", args.sampler), ("cfg", args.cfg),
                       ("sc_cfg", args.sc_cfg), ("gamma", args.gamma)):
        if value is not None:
            chosen[key] = value
    if args.task in ("owt", "owt-prefix") and chosen["cfg"] != 1:
        raise ValueError("OWT checkpoint uses SC-CFG only; input CFG must be 1")
    if any(not math.isfinite(chosen[k]) or chosen[k] < 0 for k in ("cfg", "sc_cfg", "gamma")):
        raise ValueError("CFG/SC-CFG/gamma must be finite and nonnegative")
    sc = SamplingConfig(sampling_method=chosen["sampling_method"],
                        num_sampling_steps=[args.steps], cfgs=[chosen["cfg"]],
                        self_cond_cfg_scales=[chosen["sc_cfg"]], time_schedule="logit_normal",
                        sde_gamma=chosen["gamma"])
    t5_path = asset(root, "t5")
    tokenizer = AutoTokenizer.from_pretrained(str(t5_path), local_files_only=True)
    rows = []
    input_manifest = None
    if args.input:
        args.input = args.input.resolve()
        input_manifest = json.loads(args.input.with_suffix(".manifest.json").read_text())
        if input_manifest["sha256"] != sha256(args.input) or input_manifest["task"] != args.task:
            raise ValueError("Input manifest does not match task/file")
        rows = read_jsonl(args.input)
        if not rows or len(rows) != input_manifest["count"] or len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Input counts/IDs do not match manifest or input is empty")
        if args.num_samples > len(rows) and args.mode == "generate":
            raise ValueError("Requested more prompts than input rows")
        if args.mode == "timing":
            if not 0 <= args.timing_prompt_index < len(rows):
                raise ValueError("Timing prompt index out of range")
            selected = [rows[0], rows[args.timing_prompt_index]]
        else:
            selected = rows[:args.num_samples] if args.num_samples else rows
        print(f"Checking {len(selected)} conditions before loading ELF weights: {args.input}", flush=True)
        tokens = preflight_conditions(selected, tokenizer, args.task, config.max_input_length, config.max_length)
        for row, ids in zip(selected, tokens):
            row["_elf_condition_ids"] = ids
    args.output.mkdir(parents=True)
    print(f"ELF {args.mode}: task={args.task}, steps={args.steps}, "
          f"split={(input_manifest or {}).get('split', 'unconditional')}, output={args.output}", flush=True)
    checkpoint_dir = asset(root, checkpoint_task)
    assets = manifest(root)
    checkpoint = checkpoint_dir / assets["assets"][checkpoint_task]["checkpoint"]
    # Inference only: use the exact EMA state without constructing an optimizer.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not payload.get("ema_params1"):
        raise ValueError("The released EMA parameters are required")
    model = ELF_models["ELF-B"](
        text_encoder_dim=512, max_length=config.max_length,
        attn_drop=config.attn_dropout, proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens, num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=tokenizer.vocab_size, num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim)
    model.load_state_dict(payload["ema_params1"], strict=True)
    model = model.to(device).eval().requires_grad_(False)
    del payload
    encoder = None
    if args.input:
        _, encoder = get_encoder(str(t5_path), torch.float32)
        encoder = encoder.to(device).eval().requires_grad_(False)
    def prepare(batch_rows):
        # Encode only the visible source. References never enter the encoder.
        batch = len(batch_rows)
        ids = torch.full((batch, config.max_length), tokenizer.pad_token_id, device=device, dtype=torch.long)
        lens = []
        for i, row in enumerate(batch_rows):
            condition = row["_elf_condition_ids"]
            ids[i, :len(condition)] = torch.tensor(condition, device=device)
            lens.append(len(condition))
        mask = (np.arange(config.max_length)[None, :] < np.array(lens)[:, None])
        enc_mask, _, _ = build_self_attn_cond_masks(mask, mask, xp=np)
        return (ids, torch.as_tensor(enc_mask, device=device),
                torch.as_tensor(mask, dtype=torch.float32, device=device),
                torch.tensor(lens, device=device))

    def encode(prepared):
        return encode_text(input_ids=prepared[0], attention_mask=prepared[1], encoder=encoder,
                           latent_mean=config.latent_mean, latent_std=config.latent_std).float()

    @torch.inference_mode()
    def sample(batch, prepared=None, cached=None):
        cond = cached if cached is not None else encode(prepared) if prepared is not None else None
        times = get_sampling_steps(args.steps, "logit_normal", config.denoiser_p_mean,
                                   config.denoiser_p_std, device=device, dtype=torch.float32)
        z = torch.randn((batch, config.max_length, 512), device=device) * config.denoiser_noise_scale
        latent = _generate_samples_single_batch(model=model, generator=generator, z=z, t_steps=times,
                    cond_seq=cond, cond_seq_mask=prepared[2] if prepared is not None else None,
                    config=config, sampling_config=sc, cfg_scale=chosen["cfg"],
                    self_cond_cfg_scale=chosen["sc_cfg"])
        predicted = _dlm_decode_batch(latent, model, 1.0, config, chosen["sc_cfg"])
        if prepared is not None:
            # Fixed prefix is observed data, restored also in the discrete artifact.
            predicted = torch.where(prepared[2].bool(), prepared[0], predicted)
            predicted = shift_left(predicted, prepared[3], tokenizer.pad_token_id)
            if args.task != "owt-prefix":
                predicted = predicted[:, :config.max_length - config.max_input_length]
        return predicted

    # A separate untimed preflight observes every real model input. Verify both
    # clean prefix projection and the actual NFE, including the neural decoder.
    pilot = prepare([rows[0]]) if rows else None
    pilot_cond = encode(pilot) if pilot is not None else None
    observed = {"calls": 0, "prefix_checked_calls": 0}
    def check_forward(module, inputs, kwargs):
        observed["calls"] += 1
        x = inputs[0]
        if not torch.isfinite(x).all():
            raise ValueError("Non-finite ELF model input")
        if pilot is not None:
            mask = pilot[2].bool()
            visible = x[..., :512][mask]
            clean = pilot_cond[mask]
            if not torch.equal(visible, clean) and not (chosen["cfg"] != 1 and torch.count_nonzero(visible) == 0):
                raise ValueError("ELF prefix was not clamped before a model forward")
            observed["prefix_checked_calls"] += 1
    hook = model.register_forward_pre_hook(check_forward, with_kwargs=True)
    try:
        sample(1, pilot, pilot_cond)
    finally:
        hook.remove()
    expected_calls = args.steps * (1 if chosen["cfg"] == 1 else 2) + 1
    if observed["calls"] != expected_calls:
        raise ValueError(f"Unexpected ELF NFE: {observed['calls']} != {expected_calls}")
    # Pilot must not alter production sample assignment.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    generator.manual_seed(args.seed)
    if args.compile:
        model = torch.compile(model)

    metadata = {
        "schema": "dlb-elf-v1", "task": args.task, "model": "ELF-B", "source_commit": SOURCE_COMMIT,
        "checkpoint_task": checkpoint_task, "checkpoint_selection": "ema_params1",
        "checkpoint_sha256": sha256(checkpoint), "assets_manifest_sha256": sha256(root / "data/elf/assets.json"),
        "source_config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)),
        "helper_sha256": sha256(Path(__file__).with_name("elf_common.py")),
        "lock_sha256": sha256(root / "artifacts/elf_lock.json"),
        "steps": args.steps, "sampling": chosen, "seed": args.seed, "batch_size": args.batch_size,
        "precision": "fp32 parameters/latents; official bf16 forward autocast; fp32 output heads",
        "matmul_precision": "high", "compile": args.compile,
        "model_canvas_t5_tokens": config.max_length, "input": input_manifest,
        "protocol": "c64_text_t5_v1" if args.task == "owt-prefix" else "elf_native_v1",
        "conditioning": "zero_shot_ood_native_projection" if args.task == "owt-prefix" else "task_trained" if rows else "none",
        "elf_forward_calls_per_sample": args.steps * (1 if chosen["cfg"] == 1 else 2) + 1,
        "condition_encoder_calls_per_sample": int(bool(rows)),
        "untimed_preflight": observed,
        **cuda_runtime_metadata(),
    }
    write_json(args.output / "request.json", metadata)
    if args.mode == "timing":
        if rows and not 0 <= args.timing_prompt_index < len(rows):
            raise ValueError("Timing prompt index out of range")
        prepared = prepare([rows[args.timing_prompt_index]]) if rows else None
        cached = encode(prepared) if prepared is not None else None
        results = {}
        for name in (["sampler_cached_condition", "encoder_plus_sampler"] if rows else ["sampler"]):
            def generate():
                return sample(1, prepared, cached if name == "sampler_cached_condition" else None)
            torch.cuda.reset_peak_memory_stats()
            result = benchmark(generate, torch.cuda.synchronize, warmups=args.warmups,
                               repeats=args.repeats, batch_size=1)
            results[name] = {**asdict(result), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                             "includes": ["noise_and_schedule", "denoising", "final_neural_decode", "argmax", "conditional_projection"]}
        write_json(args.output / "timing.json", {**metadata, "prompt_index": args.timing_prompt_index,
                   "condition_length": prepared[3].tolist() if prepared is not None else [], "results": results})
        print(json.dumps(results, indent=2))
        return
    count = (args.num_samples or len(rows)) if rows else args.num_samples
    schedule = [(i, 0) for i in range(count)]
    if args.diversity:
        schedule.extend((i, completion) for completion in range(1, 5) for i in range(min(256, count)))
    temporary = args.output / "samples.jsonl.tmp"
    start = time.perf_counter()
    with temporary.open("w") as handle:
        for offset in range(0, len(schedule), args.batch_size):
            shard = schedule[offset:offset + args.batch_size]
            batch_rows = [rows[i] for i, _ in shard] if rows else []
            prepared = prepare(batch_rows) if rows else None
            predicted = sample(len(shard), prepared)
            predicted = mask_after_eos(predicted, tokenizer.eos_token_id, tokenizer.pad_token_id).cpu().tolist()
            for j, ((i, completion), ids) in enumerate(zip(shard, predicted)):
                if args.task == "owt-prefix":
                    ids = ids[:config.max_length - int(prepared[3][j].item())]
                text = tokenizer.decode(ids, skip_special_tokens=True)
                row = {"id": offset + j, "prompt_id": i, "completion_id": completion,
                       "generated": text, "token_ids": ids}
                if rows:
                    row.update(input=rows[i]["input"], reference=rows[i]["output"], source_id=rows[i]["id"],
                               condition_t5_length=int(prepared[3][j].item()))
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{offset + len(shard)}/{len(schedule)} samples", flush=True)
    temporary.replace(args.output / "samples.jsonl")
    write_json(args.output / "generation.json", {**metadata, "sample_count": len(schedule),
               "prompt_count": count, "diversity": args.diversity,
               "samples_sha256": sha256(args.output / "samples.jsonl"),
               "wall_seconds_including_io_not_primary_latency": time.perf_counter() - start})


if __name__ == "__main__":
    main()

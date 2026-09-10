#!/usr/bin/env python3
"""Score independent ELF artifacts with pinned, offline evaluation assets."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import sys

from elf_common import ROOT, asset, manifest, offline, read_jsonl, require_server, sha256, write_json


def validate_records(rows, meta):
    if len(rows) != meta["sample_count"]:
        raise ValueError("Sample count differs from generation manifest")
    n = meta["prompt_count"]
    expected = [(i, 0) for i in range(n)]
    if meta.get("diversity"):
        expected += [(i, c) for c in range(1, 5) for i in range(min(256, n))]
    actual = [(r["prompt_id"], r["completion_id"]) for r in rows]
    if actual != expected or [r["id"] for r in rows] != list(range(len(rows))):
        raise ValueError("Missing/duplicated/out-of-order prompt completions")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--mauve", action="store_true")
    args = p.parse_args()
    require_server()
    offline()
    root = args.root.resolve()
    sys.path[:0] = [str(root), str(root / "src")]
    meta = json.loads((args.run / "generation.json").read_text())
    samples = args.run / "samples.jsonl"
    if sha256(samples) != meta["samples_sha256"]:
        raise ValueError("Sample file changed since generation")
    if sha256(root / "data/elf/assets.json") != meta["assets_manifest_sha256"]:
        raise ValueError("Asset manifest changed since generation")
    rows = read_jsonl(samples)
    validate_records(rows, meta)
    primary = [r for r in rows if r["completion_id"] == 0]
    result = {"schema": "dlb-elf-metrics-v1", "task": meta["task"],
              "generation_manifest_sha256": sha256(args.run / "generation.json"),
              "samples_sha256": sha256(samples), "sample_count": len(primary),
              "evaluator_sha256": sha256(Path(__file__)), "metrics": {}}
    metrics = result["metrics"]
    if meta["task"] in ("wmt14", "xsum"):
        import sacrebleu
        from rouge_score import rouge_scorer
        hypotheses, refs = [r["generated"] for r in primary], [r["reference"] for r in primary]
        # Same metric options as official utils/metrics_utils.py.
        bleu = sacrebleu.metrics.BLEU(lowercase=True, effective_order=True)
        metrics["bleu"] = bleu.corpus_score(hypotheses, [refs]).score
        metrics["bleu_signature"] = str(bleu.get_signature())
        metrics["empty_fraction"] = sum(not x.strip() for x in hypotheses) / len(hypotheses)
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = [scorer.score(ref, hyp) for ref, hyp in zip(refs, hypotheses)]
        for key in ("rouge1", "rouge2", "rougeL"):
            values = [s[key].fmeasure * 100 for s in scores]
            metrics[key] = statistics.fmean(values)
            metrics[key + "_sem"] = statistics.pstdev(values) / math.sqrt(len(values))
        metrics["reference_policy"] = "full raw reference; no target truncation; ROUGE F1 stemmer=true"
    else:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from evaluation.generative_perplexity import compute_gen_ppl, aggregate_nll
        from evaluation.conditional_perplexity import PromptExcludedCausalLMScorer
        from evaluation.unigram_entropy import unigram_entropy
        from evaluation.self_bleu import compute_self_bleu
        gpt_path = asset(root, "gpt2")
        scorer_path = asset(root, "gpt2-large")
        tok = AutoTokenizer.from_pretrained(str(gpt_path), local_files_only=True)
        tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        tok.truncation_side = "right"
        tokenized = [tok(r["generated"], add_special_tokens=False)["input_ids"] for r in rows]
        lengths = [len(ids) for ids in tokenized[:len(primary)]]
        metrics["gpt2_length_mean"] = statistics.fmean(lengths)
        metrics["gpt2_length_min"] = min(lengths)
        metrics["empty_fraction"] = sum(n == 0 for n in lengths) / len(lengths)
        metrics["below_64_tokens_fraction"] = sum(n < 64 for n in lengths) / len(lengths)
        metrics["above_1024_tokens_fraction"] = sum(n > 1024 for n in lengths) / len(lengths)
        limit = 64 if meta["task"] == "owt-prefix" else 1024
        tokenized = [ids[:limit] for ids in tokenized]
        # Never improve apparent PPL by silently removing degenerate generations.
        if any(len(ids) < 2 for ids in tokenized):
            result["valid"] = False
            result["reason"] = "At least one completion has fewer than two GPT-2 tokens; no samples were dropped"
            write_json(args.run / "metrics.json", result)
            raise ValueError(result["reason"])
        metrics["entropy_gpt2_nats"] = statistics.fmean(unigram_entropy(ids) for ids in tokenized[:len(primary)])
        result["evaluator_assets"] = {k: manifest(root)["assets"][k] for k in ("gpt2", "gpt2-large")}
        model = AutoModelForCausalLM.from_pretrained(str(scorer_path), local_files_only=True).cuda().eval()
        if meta["task"] == "owt":
            ppl = compute_gen_ppl([r["generated"] for r in primary], model, tok,
                                  batch_size=args.batch_size, device="cuda",
                                  model_revision=result["evaluator_assets"]["gpt2-large"]["revision"],
                                  tokenizer_revision=result["evaluator_assets"]["gpt2"]["revision"])
            metrics["generative_ppl"] = asdict(ppl)
        else:
            scorer = PromptExcludedCausalLMScorer(model, device="cuda")
            def conditional_ppl(targets):
                parts = []
                for start in range(0, len(primary), args.batch_size):
                    batch = primary[start:start + args.batch_size]
                    combined, loss_masks = [], []
                    for j, row in enumerate(batch):
                        prefix = tok(row["input"], add_special_tokens=False)["input_ids"]
                        target = targets[start + j]
                        if not prefix or len(prefix) + len(target) > 1024:
                            raise ValueError("Conditional PPL input exceeds context or has empty prefix")
                        combined.append(prefix + target)
                        loss_masks.append([0] * len(prefix) + [1] * len(target))
                    width = max(map(len, combined))
                    attn = [[1] * len(x) + [0] * (width - len(x)) for x in combined]
                    masks = [x + [0] * (width - len(x)) for x in loss_masks]
                    ids = [x + [tok.pad_token_id] * (width - len(x)) for x in combined]
                    parts.append(scorer.score_prompt_excluded_batch(ids, attn, masks))
                return asdict(aggregate_nll(parts))
            metrics["conditional_ppl"] = conditional_ppl(tokenized[:len(primary)])
            references = [tok(r["reference"], add_special_tokens=False)["input_ids"][:64] for r in primary]
            metrics["reference_conditional_ppl"] = conditional_ppl(references)
            metrics["conditional_ppl_policy"] = "separate GPT-2 prefix/suffix tokenization; prefix targets masked"
            metrics["entropy_all_completions_gpt2_nats"] = statistics.fmean(unigram_entropy(ids) for ids in tokenized)
            if meta.get("diversity"):
                groups = defaultdict(list)
                for row, ids in zip(rows, tokenized):
                    if row["prompt_id"] < min(256, meta["prompt_count"]):
                        groups[row["prompt_id"]].append(" ".join(map(str, ids)))
                metrics["grouped_self_bleu"] = statistics.fmean(compute_self_bleu(group).score for group in groups.values())
            if args.mauve:
                import mauve
                del scorer, model
                torch.cuda.empty_cache()
                score = mauve.compute_mauve(
                    p_text=[tok.decode(ids) for ids in tokenized[:len(primary)]],
                    q_text=[tok.decode(ids) for ids in references], featurize_model_name=str(scorer_path),
                    device_id=0, max_text_length=64, seed=42, verbose=False)
                metrics["mauve"] = float(score.mauve)
        metrics["entropy_policy"] = "mean per-sample natural-log entropy on retokenized GPT-2 suffix/text; same PPL length cap"
    result["valid"] = True
    write_json(args.run / "metrics.json", result)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

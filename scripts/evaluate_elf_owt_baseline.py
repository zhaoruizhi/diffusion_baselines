#!/usr/bin/env python3
"""Rescore public ELF OWT outputs against the existing C64 baseline (no generation)."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import statistics
import sys

from elf_common import ROOT, asset, manifest, offline, read_jsonl, require_server, sha256, write_json
from run_elf_remaining import build_matrix, reusable_result


@dataclass
class C64Record:
    """Token-record interface consumed by the existing baseline evaluator."""
    prompt_id: int
    completion_id: int
    prefix_token_ids: list[int]
    continuation_token_ids: list[int]
    reference_token_ids: list[int]


def c64_records(rows, prompts, tokenized, tokenizer):
    by_id = {p['prompt_id']: p for p in prompts}
    records = []
    for row, ids in zip(rows, tokenized, strict=True):
        source = by_id[row['prompt_id']]
        prefix, reference = source['prefix_token_ids'], source['reference_token_ids']
        if len(prefix) != 64 or len(reference) != 64:
            raise ValueError('Canonical prompt/reference must contain exactly 64 GPT-2 tokens')
        # Prove the saved ELF condition/reference came from this exact prompt.
        if row['input'] != tokenizer.decode(prefix, skip_special_tokens=False):
            raise ValueError('Saved ELF input differs from canonical C64 prompt')
        if row['reference'] != tokenizer.decode(reference, skip_special_tokens=False):
            raise ValueError('Saved ELF reference differs from canonical C64 reference')
        if len(ids) < 64:
            raise ValueError('Short response cannot satisfy the 64-token scoring contract')
        records.append(C64Record(row['prompt_id'], row['completion_id'], prefix, ids[:64], reference))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--task', choices=['owt', 'owt-prefix'], required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    root = args.root.resolve()
    sys.path[:0] = [str(root), str(root / 'src')]
    output = args.output or root / 'results' / f'elf-owt-baseline-eval-{args.task}'
    assets = manifest(root)['assets']
    asset_hash = sha256(root / 'data/elf/assets.json')
    cells, blocked = build_matrix(root, [args.task])
    if blocked:
        raise ValueError(f'Invalid inputs: {blocked}')
    runs = []
    for cell in cells:
        found, rejected = reusable_result(root, cell, asset_hash, assets)
        if not found:
            raise ValueError(f'Missing verified generation: {cell.key}; first run run_elf_remaining.py; rejected={rejected}')
        path = Path(found['run'])
        runs.append((cell, path))
        print(f'RESCORE {cell.key}: {path}', flush=True)
    if args.plan:
        print(f'{len(runs)} cells; no generation, no training; output={output}')
        return
    require_server()
    offline()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from evaluation.generative_perplexity import compute_gen_ppl
    from evaluation.conditional_perplexity import compute_conditional_gen_ppl, conditional_texts
    from evaluation.conditional_evaluate import _gpt2_tokenized_self_bleu
    from evaluation.unigram_entropy import unigram_entropy
    tok = AutoTokenizer.from_pretrained(str(asset(root, 'gpt2')), local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = tok.truncation_side = 'right'
    prompts = read_jsonl(root / 'data/conditional/owt-c64/prompts.jsonl') if args.task == 'owt-prefix' else []
    dependencies = [Path(__file__), root / 'evaluation/generative_perplexity.py',
                    root / 'evaluation/conditional_perplexity.py', root / 'evaluation/conditional_evaluate.py',
                    root / 'evaluation/unigram_entropy.py', root / 'evaluation/self_bleu.py']
    evaluator = {p.name: sha256(p) for p in dependencies}
    model = None
    summary = []
    for cell, path in runs:
        destination = output / f'steps_{cell.steps}' / 'metrics.json'
        provenance = {'protocol': 'elf_owt_public_baseline_eval_v1', 'task': args.task,
                      'steps': cell.steps, 'source_run': str(path),
                      'samples_sha256': sha256(path / 'samples.jsonl'),
                      'generation_manifest_sha256': sha256(path / 'generation.json'),
                      'assets_manifest_sha256': asset_hash, 'evaluator': evaluator,
                      'batch_size': args.batch_size,
                      'prompts_sha256': sha256(root / 'data/conditional/owt-c64/prompts.jsonl') if prompts else None}
        cached = json.loads(destination.read_text()) if destination.exists() else {}
        if all(cached.get(k) == v for k, v in provenance.items()) and type(cached.get('valid')) is bool:
            result = cached
            print(f'SKIP verified evaluation: {destination}', flush=True)
        else:
            rows = read_jsonl(path / 'samples.jsonl')
            ids = [tok(row['generated'], add_special_tokens=False)['input_ids'] for row in rows]
            lengths = list(map(len, ids))
            limit = 64 if prompts else 1024
            short = [row['id'] for row, tokens in zip(rows, ids) if len(tokens) < (64 if prompts else 2)]
            result = {**provenance, 'sample_count': len(rows), 'prompt_count': 1024,
                      'metrics': {}, 'valid': not short,
                      'length_diagnostics': {'mean': statistics.fmean(lengths), 'min': min(lengths),
                                             'max': max(lengths), 'below_64_count': sum(n < 64 for n in lengths),
                                             'above_1024_count': sum(n > 1024 for n in lengths)},
                      'short_sample_ids': short,
                      'native_tokenizer': 't5-small', 'scoring_tokenizer': 'gpt2',
                      'scoring_limit': limit, 'completion_scope': 'all_saved_completions',
                      'output_policy': 'released ELF EOS-trimmed decoded text; GPT-2 retokenized; no fabricated padding'}
            if short:
                result['reason'] = 'Short outputs: cannot score the requested contract; none dropped, padded, or resampled'
            else:
                sliced = [tokens[:limit] for tokens in ids]
                result['metrics']['entropy_gpt2_nats'] = statistics.fmean(unigram_entropy(tokens) for tokens in sliced)
                records = c64_records(rows, prompts, ids, tok) if prompts else None
                if model is None:
                    model = AutoModelForCausalLM.from_pretrained(str(asset(root, 'gpt2-large')), local_files_only=True).cuda().eval()
                kwargs = dict(batch_size=args.batch_size, device='cuda',
                              model_revision=assets['gpt2-large']['revision'], tokenizer_revision=assets['gpt2']['revision'])
                if prompts:
                    ppl = compute_conditional_gen_ppl(records, model, tok, tok, **kwargs)
                    result['metrics']['conditional_ppl'] = asdict(ppl)
                    texts = conditional_texts(records, tok)
                    result['metrics']['grouped_self_bleu'] = _gpt2_tokenized_self_bleu(records, [t.generated_suffix for t in texts], tok)
                    result['ppl_policy'] = 'existing compute_conditional_gen_ppl; canonical prefix IDs; decode joined IDs then retokenize; prompt loss excluded'
                else:
                    ppl = compute_gen_ppl([row['generated'] for row in rows], model, tok, **kwargs)
                    result['metrics']['generative_ppl'] = asdict(ppl)
                if ppl.sample_count != len(rows):
                    raise ValueError('PPL silently changed completion count')
            write_json(destination, result)
        metrics = result['metrics']
        ppl = metrics.get('conditional_ppl', metrics.get('generative_ppl', {})).get('perplexity')
        summary.append({'task': args.task, 'steps': cell.steps, 'sample_count': result['sample_count'],
                        'valid': result['valid'], 'ppl': ppl, 'entropy': metrics.get('entropy_gpt2_nats'),
                        'short_count': len(result['short_sample_ids']), 'metrics_path': str(destination)})
        print(json.dumps(summary[-1]), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'summary.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f'SUMMARY: {output / "summary.csv"}', flush=True)
    if any(not row['valid'] for row in summary):
        raise SystemExit(2)


if __name__ == '__main__':
    main()

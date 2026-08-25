# Zero-Shot Conditional Generation Benchmark Design

**Date:** 2026-08-25

**Status:** Approved in conversation; awaiting written-spec review

**Protocol ID:** `c64_zs_v1`

## 1. Goal

Add a conditional-generation benchmark for every currently supported baseline by reusing its existing unconditional checkpoint. The benchmark fixes the first 64 real tokens of a held-out sequence as a prefix and generates the continuation without training, fine-tuning, or altering checkpoints.

The primary experiment is:

- sample 1,024 held-out sequences from each dataset with the existing model-native tokenizer and processed validation split;
- use tokens `[0:64]` as an immutable prefix;
- run each baseline's normal reverse process for the remaining canvas;
- evaluate the generated continuation, using tokens `[64:128]` as the aligned 64-token reference;
- generate five continuations for the first 256 prompts for diversity metrics.

This protocol is deliberately zero-shot. It is a fair test of whether each trained unconditional generator can respect a hard prefix at inference time, but it is not equivalent to conditional fine-tuning used by some papers.

## 2. Scope and compatibility

The current unconditional benchmark remains unchanged: its matrix, result paths, request identity, sample schema, scripts, and aggregate outputs keep their existing behavior.

Conditional generation is an additive benchmark with separate configuration, schemas, artifacts, output paths, evaluation, and aggregation. It covers the same 137 supported model/dataset/step tasks as the current matrix. The existing unsupported RDLM/OWT model/dataset cell remains in a separate explicit unsupported inventory rather than being expanded into step tasks or silently dropped.

No upstream baseline repository is modified. Integration stays in this project's adapters and launch wrappers.

## 3. Experiment definition

### 3.1 Datasets and canvases

| Dataset | Existing processed split | Tokenizer | Model canvas | Fixed prefix | Evaluated reference |
|---|---|---|---:|---:|---:|
| LM1B | `data/processed/lm1b-bert-128/validation` | existing pinned BERT tokenizer | 128 | 64 | next 64 |
| OWT | `data/processed/owt-gpt2-1024/validation` | existing pinned GPT-2 tokenizer | 1024 | 64 | next 64 |

For OWT, generation runs on the full 1,024-token model canvas. The generated suffix canvas therefore has 960 tokens, while common quality metrics evaluate the first 64 generated suffix tokens against tokens `[64:128]`. Long-form artifacts retain the entire generated suffix so later long-context analyses do not require rerunning models.

The current processed validation rows are packed sequences and may contain EOS-separated document boundaries. The benchmark uses those rows uniformly and records provenance; it does not introduce document-boundary filtering or rebuild the datasets.

### 3.2 Deterministic prompt selection

Prompt preparation creates:

- `data/conditional/lm1b-c64/prompts.jsonl`
- `data/conditional/owt-c64/prompts.jsonl`
- `data/manifests/conditional-lm1b-c64.json`
- `data/manifests/conditional-owt-c64.json`

Selection uses seed 42 and a deterministic SHA-256-driven ordering of source indices. Exactly 1,024 unique rows are selected per dataset. Each prompt record contains:

- `prompt_id` from 0 through 1,023;
- `source_index` in the processed validation dataset;
- exactly 64 `prefix_token_ids`;
- exactly 64 `reference_token_ids`;
- a SHA-256 digest of the source sequence.

The sidecar manifest binds the source dataset manifest digest, dataset identifier, tokenizer name and revision, source split, selection algorithm and seed, sequence count, lengths, and prompt-file digest. Prompt verification recomputes all digests and validates lengths and token bounds before any GPU process is launched.

### 3.3 Completion schedule

Each supported matrix cell produces exactly 2,048 samples:

- completion 0 for prompts 0–1,023: 1,024 samples used for aligned quality and reference metrics;
- completions 1–4 for prompts 0–255: 1,024 additional samples used with completion 0 for five-way diversity evaluation.

Existing configured batch sizes divide both 1,024 and 256. Runners must reject incompatible batch sizes rather than over-generate and trim outputs, because trimming changes seed-to-sample assignment.

## 4. Configuration and request identity

Add `configs/conditional.yaml` with a versioned configuration equivalent to:

```yaml
schema_version: 1
protocol: c64_zs_v1
selection_seed: 42
sampling_seed: 42
prompt_count: 1024
prefix_length: 64
evaluation_continuation_length: 64
diversity_prompt_count: 256
completions_per_diversity_prompt: 5
datasets:
  lm1b:
    model_length: 128
  owt:
    model_length: 1024
```

Extend the shared `RunRequest` with explicit conditional fields:

- `generation_mode`: `unconditional` or `conditional_prefix`;
- conditioning manifest path and SHA-256 digest;
- prefix and evaluated-continuation lengths;
- total prompt count and diversity prompt count;
- completion index/range represented by the launched shard.

All new fields participate in run identity, resume checks, metadata, and provenance. Their unconditional defaults reproduce current request identities and behavior.

## 5. Matrix and output layout

Add `src/dlb/conditional_matrix.py` with schema `dlb-conditional-generation-matrix-v1`. It derives the conditional cells from the existing supported baseline grid while assigning conditional paths, request fields, and protocol metadata. It must preserve the exact supported/unsupported accounting of the unconditional matrix.

Conditional output lives only under `results/conditional/`:

```text
results/conditional/
  matrix/generation.tsv
  samples/<dataset>/<model>/steps_<N>/
  metrics/<dataset>/<model>/steps_<N>/
  timing/<dataset>/<model>/steps_<N>/
  summary/
```

This separation prevents old aggregation scripts from accidentally mixing unconditional and conditional samples.

## 6. Conditional sample schema

Keep the existing unconditional `SampleRecord` semantics intact. Add a separate versioned `ConditionalSampleRecord` with:

- `sample_id`, `prompt_id`, `completion_id`, and `source_index`;
- prefix, generated-continuation, aligned-reference, and full token IDs;
- decoded prefix, generated-continuation, aligned-reference, and full text;
- sample seed and generation time;
- `prefix_exact_match`, which must be true for publication.

Schema validators enforce:

- prefix length equals 64;
- reference length equals 64;
- full token IDs start with the stored prefix;
- continuation token IDs equal the suffix slice of full token IDs;
- prompt/completion IDs follow the required schedule;
- all token IDs are in the recorded tokenizer vocabulary;
- `prefix_exact_match` is true.

Add dedicated conditional readers, validators, and atomic writers in the I/O layer. A shard is published only after its full record count, schedule, and prefix invariants pass.

## 7. Conditioning semantics

The universal invariant is simple: the first 64 positions are real observed tokens and may never be changed. The suffix starts from each model's normal prior and follows its normal reverse process.

The earlier adapter categories refer to how the latent state is represented, not to different experimental tasks:

- **Absorbing-mask models** such as MDLM initialize suffix positions as mask tokens. Prefix positions are initialized to real tokens and clamped back to those tokens after every reverse update.
- **Uniform categorical models** such as Duo initialize suffix positions from their original categorical prior. Prefix transitions are disabled by clamping the real prefix before and after each transition.
- **Continuous/flow models** such as FLM, FMLM, and LangFlow initialize suffix positions from their original continuous prior. Prefix positions are represented by the model's exact clean one-hot vectors or token embeddings, restored before every model evaluation, and excluded from state evolution.
- **Hybrid/manifold models** such as CANDI and RDLM retain their original suffix prior, while both the discrete prefix identity and any continuous/manifold representation of that prefix are projected back to the clean endpoint at each relevant step.

Thus `clamp` means enforcing a hard inpainting constraint throughout sampling, not merely overwriting the final output. A final token-level clamp and exact-match assertion provide defense in depth.

### 7.1 Adapter-specific hooks

Use native conditioning/projection APIs where they exist; otherwise install a scoped runtime hook through a context manager and restore original functions after sampling.

| Family | Baselines | Required hook |
|---|---|---|
| Absorbing mask | MDLM | inject prefix in `_sample_prior`; preserve it through absorbing updates; final clamp |
| Uniform categorical | Duo, Duo+DCD | initialize suffix from original uniform prior; wrap `_ancestral_update` with pre/post prefix clamp |
| Continuous vocabulary flow | FLM, FMLM | before every `forward`, replace the first 64 states with exact prefix one-hot states; final token clamp |
| Embedding flow | LangFlow | project the first 64 positions of both `noisy_embeds` and `x_self_cond` to clean prefix embeddings before every forward call; final clamp |
| Native hybrid | CANDI | pass native `prompt_tokens` and `prompt_mask`; still assert output prefix |
| Native manifold | RDLM | pass a prefix projection through native `get_sampling_fn(..., proj_fn=...)` using the clean one-hot manifold endpoint |
| Native projected samplers | SDTT variants, Di4C variants | supply the native `project_fn` at every sampling step for both masked and Duo-family variants |

Runtime instrumentation belongs in a new `src/dlb/adapters/conditional_runtime.py`. Shared request/capture code and every relevant adapter receive the uniform conditional arguments. Server-backed launchers in `src/dlb/adapters/_distilled_runtime.py`, `sample_sdtt.py`, `sample_di4c.py`, and `sample_langflow.py` must serialize and validate the same fields.

Metadata records `conditioning_implementation` as `native_projection` or `zero_shot_runtime_projection`. FLM/FMLM/LangFlow results must be labeled zero-shot/OOD conditioning; in particular, they must not be described as reproducing a paper's conditionally fine-tuned setting.

## 8. Evaluation

Add `evaluation/conditional_evaluate.py` as the cell-level evaluator and `evaluation/mauve_score.py` for a pinned, local-only MAUVE computation. Refactor existing generation-PPL, Self-BLEU, and entropy utilities into reusable functions without changing their unconditional CLI behavior.

Each metric is computed on generated continuation tokens only unless explicitly stated:

- **Conditional generation PPL:** tokenize prefix and continuation independently with the pinned local GPT-2 evaluator, concatenate IDs, and mask all prefix targets from the loss. This measures suffix likelihood conditioned on the prefix. Compute the same value for the ground-truth 64-token reference as a calibration baseline.
- **MAUVE:** compare the 1,024 completion-0 generated 64-token continuations with their 1,024 aligned reference continuations. Evaluator model and revision are pinned and loaded without network fallback.
- **Grouped Self-BLEU:** use prompts 0–255 and all five continuations per prompt; compute BLEU-4 with equal n-gram weights and smoothing method 1, using the GPT-2 tokenization already recorded by the benchmark.
- **Entropy:** compute token entropy from continuation token IDs only, with the same grouping represented in metric metadata.
- **Prefix exact-match rate:** must equal 1.0; any mismatch invalidates the cell rather than being reported as a soft degradation.

Metric JSON includes schema/protocol version, cell identity, checkpoint/tokenizer/prompt manifest digests, sample artifact digest, evaluator revisions, sample counts, truncation lengths, and numerical results.

## 9. Timing

Conditional timing uses prompt 0, batch size 1, five warmups, and 32 measured repeats per supported cell. The prefix tensor is moved to the target device before the timer starts. Measured time includes all conditional projection/clamping work and the full native sampling canvas, but excludes checkpoint loading, tokenizer loading, manifest parsing, and host-to-device prompt transfer.

For OWT this times the full 1,024-position process with a 960-token generated suffix; the 64-token evaluation truncation is metadata, not the timed generation length. Timing artifacts bind the same request, checkpoint, tokenizer, prompt manifest, step count, device, and precision metadata as sample artifacts.

## 10. Aggregation

Add `evaluation/conditional_aggregate.py` and a conditional summary command. Aggregation validates before producing tables:

- matrix schema and the expected 137 supported step tasks, plus the explicit RDLM/OWT unsupported inventory record;
- exact 2,048-record completion schedule per cell;
- exact prefix preservation in every record;
- prompt, sample, metric, checkpoint, tokenizer, and timing digest linkage;
- matching dataset, model, step count, and protocol identity;
- presence and validity of all required metrics and timing artifacts.

Conditional aggregation never scans unconditional paths.

## 11. Commands and scripts

Add these entry points following existing script conventions:

- prompt build and verification commands;
- `scripts/run_conditional_one.sh`;
- `scripts/run_conditional_all.sh` and a four-GPU launcher;
- conditional evaluation and four-GPU evaluation launchers;
- conditional timing and four-GPU timing launchers;
- conditional aggregate/summary command.

The one-cell interface accepts at least model, dataset, step count, prompt manifest, checkpoint provenance, output root, and resume/overwrite controls. Resume is allowed only when the complete request identity and all bound digests match.

## 12. Failure handling

Before GPU launch, reject:

- prompt or source-manifest digest mismatch;
- wrong tokenizer/dataset pairing;
- incorrect prefix/reference lengths;
- out-of-range token IDs;
- duplicate/missing prompt IDs;
- incompatible batch sizes.

During generation, fail immediately on schedule exhaustion, record misalignment, non-finite latent state, or prefix mismatch. Write shards to temporary paths and atomically publish only validated artifacts. Do not silently trim, pad, regenerate under a different seed, or reuse an artifact with a partial identity match.

## 13. Testing and verification

Implementation follows test-driven development. Required coverage includes:

- deterministic prompt construction and tamper detection;
- conditional configuration/request/schema round trips and validation failures;
- conditional matrix parity and unsupported-cell accounting;
- fake-sampler tests for every runtime projection category, checking the prefix before/after every relevant update;
- adapter CLI/request serialization for all baseline families;
- metric fixtures for suffix-only loss masking, MAUVE inputs, grouped Self-BLEU, entropy, and prefix validity;
- aggregate rejection of missing, duplicated, mismatched, or provenance-invalid artifacts;
- shell-script contract tests and dry runs.

Before a full benchmark run, perform server-backed smoke tests with two prompts, two completions, and both one-step and maximum-step settings for each sampler family. Then run the complete unit/integration suite and the repository's existing unconditional tests to prove backward compatibility.

## 14. Alternatives considered

1. **Overwrite only the final prefix tokens.** Rejected because the denoiser would generate the suffix without observing a stable prefix during intermediate steps, so it would not implement conditional generation.
2. **Modify every upstream repository.** Rejected because it multiplies maintenance and provenance risk; scoped adapter-level projection provides one auditable protocol while retaining native APIs where available.
3. **Use a time-noised prefix for every model.** Not selected for the primary benchmark because implementations differ across model families and would introduce an additional modeling choice. It can be evaluated later as an explicitly versioned ablation.
4. **Retrain conditional checkpoints.** Out of scope: the requested experiment measures existing checkpoints only.

## 15. Acceptance criteria

The feature is complete when:

1. Existing unconditional commands and tests remain unchanged and pass.
2. Both prompt manifests deterministically verify and contain 1,024 unique examples.
3. The conditional matrix contains exactly 137 supported step tasks, and its separate unsupported inventory explicitly contains RDLM/OWT.
4. Every supported cell can generate exactly 2,048 validated records from its existing checkpoint without training.
5. Every generated record preserves all 64 prefix tokens exactly.
6. Conditional quality, diversity, and timing artifacts pass provenance and count validation.
7. Aggregation produces a separate conditional summary and refuses incomplete or mismatched runs.
8. Smoke tests cover every sampler family at minimum and maximum tested step counts before the full experiment is launched.

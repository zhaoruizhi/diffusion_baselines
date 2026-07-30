# Task 7 implementation report

## Status

Implemented FLM/FMLM, Duo/Duo+DCD, MDLM, and CANDI adapters against the
canonical runner protocol. Upstreams remain read-only. Commands are argv arrays
and dry-run only reads manifests/configs.

## Pinned upstream audit

| Upstream | SHA | Inspected entry/config/sampling/eval sources |
|---|---|---|
| FLM | `a1918d5164e5038e37d0b7a4fb2010ce75b863b3` | complete `main.py`; `configs/config.yaml`; FLM/FMLM algo and LM1B/OWT data configs; all four `gen_ppl_*_{flm,fmlm}.sh`; FLM/FMLM `generate_samples` implementations |
| Duo | `7c9b498f5b717de064d6fad7e2509c866e6cb620` | complete `main.py`; `configs/config.yaml`; `duo_base.yaml`, `duo.yaml`; LM1B/OWT generation and eval scripts; ancestral implementation and HF loader |
| MDLM | `c112c526d193436838c98d81455ee51f90309470` | complete `main.py`; `configs/config.yaml`; LM1B/OWT data configs; README sample commands; OWT eval and LM1B train scripts; DDPM implementation in `diffusion.py` |
| CANDI | `cd57ae9eec98d6ac71cd52bdc50eeec8dfd70f91` | complete `main.py`; `configs/config.yaml`; `candi.yaml`; OWT generation/sweep and Slurm scripts; cached hybrid implementation in `algo.py` |

Actual keys verified include `sampling.steps`, `sampling.num_sample_batches`,
`loader.eval_batch_size`, checkpoint/output keys where present, and each
algorithm-specific key. Every rendered data config group is checked against a
real pinned YAML file.

## Per-model argv mapping

| Model | Entrypoint | Dataset config / eval batch | Sampling mapping |
|---|---|---|---|
| FLM | `upstreams/flm/main.py` | `lm1b-wrap`/32; `openwebtext-split`/16 | `algo=flm`, Euler, exact `sampling.steps` |
| FMLM | `upstreams/flm/main.py` | `lm1b-wrap`/16; `openwebtext-split`/16 | `algo=fmlm`, gamma 0.8/1.0, exact steps |
| Duo | `upstreams/duo/main.py` | `lm1b-wrap`/64; `openwebtext-split`/8 | `algo=duo_base`, ancestral predictor/removal |
| Duo+DCD | `upstreams/duo/main.py` | same as Duo | same lineage/sampler; DCD checkpoint only |
| MDLM | `upstreams/mdlm/main.py` | `lm1b`/16; `openwebtext-split`/16 | `parameterization=subs`, `predictor=ddpm`, removal true |
| CANDI | `upstreams/candi/main.py` | `lm1b-wrap`/2; `openwebtext-split`/2 | cached hybrid, mix 0.5, step size 1.0, temp 1.0 |

All commands force one device so total output is deterministic:
`ceil(requested/eval_batch_size)` batches, followed by stable-prefix trimming.
Duo/MDLM do not expose a temperature Hydra key at these SHAs. Their unscaled
categorical probabilities are the effective 1.0 path; no override was invented.

## Checkpoints and conversion

Selection is derived only from `artifacts/checkpoints.yaml` coverage or the
registry's recipe output. Teacher family is checked before argv rendering.
Dry-run renders the canonical expected path without bytes; real rendering
requires all resource files, or one unambiguous non-empty recipe checkpoint.

Accepted formats:

- FLM/Duo/CANDI upstream JSON: exactly `generative_ppl`, `entropy`, and
  `generated_seqs`, cross-checked with the capture sidecar.
- Project compatibility capture: exactly
  `dlb-upstream-token-capture-v1` with sequential unique IDs, non-empty text,
  and non-empty token IDs. MDLM requires this because pinned `main.py` only
  prints its final batch.

Captured upstream token IDs are preferred. If a standard JSON predates the
wrapper, conversion may re-tokenize with the locally cached tokenizer at the
revision pinned in `artifacts/data.yaml`, recording `token_ids_source`.
Unexpected counts/formats, missing or extra rows, duplicate IDs, empty samples,
and out-of-vocabulary tokens are rejected. `write_samples_atomic` remains the
runner's publication boundary. Upstreams expose no per-sample latency, so
records use schema-permitted `0.0` only with explicit
`unavailable_excluded_sentinel`/`exclude_from_latency` conversion metadata;
Task 11 owns official latency.

## RED / GREEN and local checks

- RED: focused collection failed with `ModuleNotFoundError: dlb.adapters`.
- GREEN: `21 passed` in `tests/test_teacher_adapters.py`.
- Dry-run: 12/12 requested model/dataset cells rendered `supported`; no writes,
  unresolved templates, empty paths, braces, or `None` arguments.
- Compilation: `python -m compileall -q src adapters tests/test_teacher_adapters.py` passed.
- Full CPU suite: `198 passed in 11.67s` (the two loopback HTTP tests were run
  with socket permission; they use only a temporary `127.0.0.1` server).

## Concerns

- No model/checkpoint/data/GPU execution was performed, as required.
- The capture wrapper is intentionally narrow: it observes the real sampler's
  returned tensor and delegates every sampling operation to pinned code.
- Server validation still needs the real environment/checkpoint/data assets;
  Task 11 separately establishes latency measurements.

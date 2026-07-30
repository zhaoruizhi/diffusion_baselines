# RDLM adapter audit

The adapter targets the read-only official RDLM checkout at commit
`67443aa6a2d0fa981eb7c6105f9cbc563e59c5c1`. It does not modify upstream
files. Only RDLM/LM1B is supported; RDLM/OWT remains the registry's structured
unsupported cell and is never mapped to Text8 or another checkpoint.

The pinned official sampling command is `main.py` with `run_mode=sample`,
`server=sample`, and `exp=sample_lm1b`. `sampling.get_sampling_fn` passes
`config.sampling.steps` into `get_sde_sampler`, whose registered `grw`
predictor performs the published SDE updates. The adapter therefore renders
the actual Hydra overrides `sampling.predictor=grw` and
`sampling.steps=<requested>`, plus `seed=<requested>` and `ngpus=1`. Generation
uses bounded microbatches of at most eight rows rather than placing the default
1,024 samples in one GPU batch. It disables the separate entropy/NLL/PPL
evaluation passes; those are not part of canonical sample generation.

The official release supplies and the adapter binds all three LM1B artifacts:

- `LM1B/checkpoint.pth`
- `LM1B/config.yaml`
- `LM1B/sde.pkl`

Task 7's checkpoint provenance digest covers the verified lock inventory,
including each trio member's observed SHA-256 (and the other files in the
single official Drive resource). Conversion re-resolves that digest before
reading output, so changing either `config.yaml` or `sde.pkl` invalidates a
request just as changing `checkpoint.pth` does.

Two pinned runtime quirks require a narrow compatibility layer in the shared
capture process. `run_sample.py` hard-codes an eight-sample target and looks
for `sde.pkl` two directories above `model_path`, which does not match the
downloaded `LM1B/` trio layout. The wrapper keeps the official Hydra
entrypoint and official SDE sampler, but executes its one-GPU worker inline,
uses the saved YAML as the checkpoint training config, routes the SDE read to
the immutable saved `sde.pkl`, and replaces the hard-coded loop bound with
`ceil(requested/microbatch)` iterations. Only a possible final ceiling excess
is removed by stable prefix before capture publication. The pinned upstream
tree remains unchanged.

The wrapper captures the texts and post-shift upstream token IDs returned by
the real `find_bos_and_shift_fn`. Exact 128-token rows are preserved. If BOS
alignment makes an upstream row shorter, conversion honestly retokenizes the
captured text with the locally cached, revision-pinned BERT tokenizer using
explicit max-length padding/truncation metadata. Output count is always exact;
no ceiling rows are silently trimmed. As with LangFlow, `0.0` timing is only an
excluded unavailable sentinel, never a measured-latency claim.

# MDLM adapter audit

Pinned upstream: `upstreams/mdlm` at `c112c526d193436838c98d81455ee51f90309470`.

No upstream patch is required. `dlb.adapters.mdlm.MDLMAdapter` maps `mdlm` to
the pinned `main.py` with `parameterization=subs`, the exact ancestral
`sampling.predictor=ddpm` path, and `sampling.noise_removal=True`. Checkpoints
must declare `masked_mdlm`; LM1B uses the actual `data=lm1b` config group and
OWT uses `data=openwebtext-split`.

This SHA has neither a sampling-temperature key nor a complete sample file: it
prints only the final text batch. Unscaled DDPM probabilities are the effective
temperature-1.0 behavior. The project-owned `dlb.adapters.capture` wrapper
observes every token tensor returned by `restore_model_and_sample` and writes
`dlb-upstream-token-capture-v1`; it does not copy or replace the sampler.

Audit with:

```bash
git -C upstreams/mdlm rev-parse HEAD
git -C upstreams/mdlm status --short
python -m dlb.command --models mdlm --datasets lm1b,owt --dry-run
```

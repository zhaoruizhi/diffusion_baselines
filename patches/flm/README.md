# FLM adapter audit

Pinned upstream: `upstreams/flm` at `a1918d5164e5038e37d0b7a4fb2010ce75b863b3`.

No upstream patch is required. `dlb.adapters.flm.FLMAdapter` maps `flm` and
`fmlm` to the pinned `main.py` `sample_eval` path. FLM fixes the existing
`sampling.solver=euler` key; FMLM uses the released `algo=fmlm` path and the
paper scripts' gamma (`0.8` for LM1B, `1.0` for OWT). Both use the manifest's
`continuous_flm` checkpoint selection and the data manifest's sequence length.

The project-owned `dlb.adapters.capture` wrapper observes token tensors returned
by `restore_model_and_sample`; it does not alter sampling. The upstream
`generated_seqs` JSON remains required and is cross-checked against the capture.

Audit with:

```bash
git -C upstreams/flm rev-parse HEAD
git -C upstreams/flm status --short
python -m dlb.command --models flm,fmlm --datasets lm1b,owt --dry-run
```

# CANDI adapter audit

Pinned upstream: `upstreams/candi` at `cd57ae9eec98d6ac71cd52bdc50eeec8dfd70f91`.

No upstream patch is required. `dlb.adapters.candi.CANDIAdapter` maps `candi`
to the pinned `main.py` hybrid path using `algo=candi`, `algo.sampler=cached`,
`algo.mixed_coeff=0.5`, `algo.step_size=1.0`, `algo.temp=1.0`, and the existing
percentile schedule. Checkpoint selection must declare `hybrid_candi`.

The project-owned `dlb.adapters.capture` wrapper observes token tensors returned
by the hybrid sampler. The pinned `generated_seqs` JSON remains required and is
cross-checked before conversion.

Audit with:

```bash
git -C upstreams/candi rev-parse HEAD
git -C upstreams/candi status --short
python -m dlb.command --models candi --datasets lm1b,owt --dry-run
```

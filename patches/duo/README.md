# Duo adapter audit

Pinned upstream: `upstreams/duo` at `7c9b498f5b717de064d6fad7e2509c866e6cb620`.

No upstream patch is required. `dlb.adapters.duo.DuoAdapter` maps `duo` and
`duo_dcd` to the pinned `main.py` with `algo=duo_base`,
`sampling.predictor=ancestral`, and `sampling.noise_removal=ancestral`.
`duo_dcd` shares the Duo sampling lineage and differs only in canonical
checkpoint selection. Every selected checkpoint or recipe must declare
`uniform_duo`.

This SHA has no `sampling.temperature` Hydra key. Its categorical draw uses
unscaled probabilities, which is the effective temperature-1.0 path; the
adapter intentionally does not invent an override. `dlb.adapters.capture`
records the returned token tensors without changing that path.

Audit with:

```bash
git -C upstreams/duo rev-parse HEAD
git -C upstreams/duo status --short
python -m dlb.command --models duo,duo_dcd --datasets lm1b,owt --dry-run
```

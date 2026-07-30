# SDTT integration notes

No upstream file is edited. `adapters/sample_sdtt.py` loads the pinned source at
commit `1150985e90b8f2d5749e4469d5154eff9ec922c4`, the locally verified KLD
round-7 (`baselines_kld_step_70000`) student bytes, and the revision-pinned local
tokenizer. It calls the released `MultiRoundSDTT.sample` API with ancestral
sampling and writes exact token IDs through an atomic project-owned capture.
The runner authenticates both `model.safetensors` and the manifest-selected
`config.json`; the wrapper treats that config as authoritative and cross-checks
the tokenizer snapshot against both `artifacts/data.yaml` and the server-created
download manifest.

The LM1B cell is a Task 12 reference reproduction. Its manifest-selected sampling
artifacts are `student_checkpoints/70000.ckpt` and a fully resolved `config.yaml`.
Its Lightning config is checked against the manifest config before use. Absent,
ambiguous, or provenance-mismatched bytes are rejected.

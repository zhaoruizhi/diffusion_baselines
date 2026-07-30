# Di4C integration notes

No upstream file is edited. `adapters/sample_di4c.py` uses the language-model
implementation under the pinned Di4C repository's `sdtt/` subtree at commit
`ac61ff9fe8e85120f9e1d2a8c5a332f8b8353dd3`. It loads only local verified
checkpoint and tokenizer bytes, calls the inherited SDTT ancestral sampler, moves
each batch to CPU immediately, and atomically publishes exact token IDs.
The official OWT checkpoint uses the fully composed project inference config
whose content hash and source commit are pinned in `artifacts/checkpoints.yaml`.
For Lightning checkpoints, the wrapper verifies the canonical file digest before
the full load required for OmegaConf `DictConfig`, then requires the embedded
architecture to agree with the manifest-selected config.

The official Zenodo `sdtt7-di4c2.ckpt` is bound exclusively to the masked-MDLM
teacher family. Duo+Di4C is always `uniform_duo` and must come from Task 12's
separate recipe outputs. The manifest selects the 20k LM1B and 50k OWT
intermediate checkpoints, so the adapter never searches for or guesses a weight.
Each Task 12 recipe must also publish its fully composed `config.yaml`; the
manifest-selected config remains authoritative at sampling time.
Config checks select only constructor/sampler fields and resolve direct standard
references such as `${model.length}`; unrelated trainer/loader custom resolvers
are never evaluated by the wrapper.

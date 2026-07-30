# Di4C integration notes

No upstream file is edited. `adapters/sample_di4c.py` uses the language-model
implementation under the pinned Di4C repository's `sdtt/` subtree at commit
`ac61ff9fe8e85120f9e1d2a8c5a332f8b8353dd3`. It loads only local verified
checkpoint and tokenizer bytes, calls the inherited SDTT ancestral sampler, moves
each batch to CPU immediately, and atomically publishes exact token IDs.

The official Zenodo `sdtt7-di4c2.ckpt` is bound exclusively to the masked-MDLM
teacher family. Duo+Di4C is always `uniform_duo` and must come from Task 12's
separate recipe outputs. The manifest selects the 20k LM1B and 50k OWT
intermediate checkpoints, so the adapter never searches for or guesses a weight.

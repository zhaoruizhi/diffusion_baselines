# LangFlow adapter audit

The adapter targets the read-only LangFlow checkout at commit
`a712b08570c56c787d6ef24f8e906a2dcdf768f5`. It does not patch or copy model
code. The supported registry cells are LangFlow/LM1B and LangFlow/OWT; each
cell binds its own official Hugging Face checkpoint and tokenizer contract.

The pinned release's public `upstreams/langflow/inference.py` hard-codes its
bundled config directory, which is not dataset-safe for the LM1B and OWT
checkpoint pair. The adapter therefore enters through
`adapters/sample_langflow.py`: it imports the pinned upstream `LangFlow` model
code, loads `LangFlowConfig` from the selected checkpoint directory, and keeps
the upstream underscore-style argparse contract. The exact mapping is:

- `--checkpoint`: immutable canonical
  `checkpoints/official/langflow/<dataset>/model.safetensors`
- `--num_samples`: requested sample count, with upstream remainder batching
- `--num_steps`: requested many-step grid value
- `--seq_length`: canonical dataset length, 128 for LM1B and 1024 for OWT
- `--seed`: request seed
- `--output`: canonical `upstream_samples.txt` in the run directory
- `--tokenizer`: canonical dataset tokenizer from `artifacts/data.yaml`

The shared `dlb.adapters.capture` process observes the token tensor returned by
the real `LangFlow.generate_samples` method and the text returned by the real
tokenizer. It writes `upstream_token_ids.json` using
`dlb-upstream-token-capture-v1`; conversion does not parse the upstream text
file's human-oriented sample delimiters. Before upstream execution, the
wrapper cross-checks the dataset tokenizer and revision in `artifacts/data.yaml`
against `data/manifests/downloads.json`, requires that manifest's immutable
local snapshot directory, and replaces the upstream movable tokenizer request
with that path plus `local_files_only=True` under Hugging Face offline mode.
There is no network or moving-revision fallback. Conversion requires exactly the
requested number of fixed-length token rows and validates Task 7's runner-resolved
checkpoint digest, lock ID, selection, and `continuous_langflow` family before
reading the capture.

The upstream does not report per-sample latency. Canonical records therefore
use schema-permitted `0.0` only as an excluded sentinel; conversion metadata
states `unavailable_excluded_sentinel` and `exclude_from_latency: true`.

# LangFlow adapter audit

The adapter targets the read-only LangFlow checkout at commit
`a712b08570c56c787d6ef24f8e906a2dcdf768f5`. It does not patch or copy model
code. The only supported registry cell is LangFlow/OWT; LangFlow/LM1B remains a
structured registry rejection and is never replaced with the OWT checkpoint.

The pinned release's real entrypoint is `upstreams/langflow/inference.py`, not
the Hydra `main.py` shape used by the Task 7 teachers. Its argparse options use
underscores, so the adapter deliberately renders `--num_samples`,
`--batch_size`, `--num_steps`, and `--seq_length` rather than the hyphenated
flags in the older plan example. The remaining exact mapping is:

- `--checkpoint`: immutable canonical
  `checkpoints/official/langflow/owt/model.safetensors`
- `--num_samples`: requested sample count, with upstream remainder batching
- `--num_steps`: requested many-step grid value
- `--seq_length`: canonical OWT length 1024
- `--seed`: request seed
- `--output`: canonical `upstream_samples.txt` in the run directory

The shared `dlb.adapters.capture` process observes the token tensor returned by
the real `LangFlow.generate_samples` method and the text returned by the real
tokenizer. It writes `upstream_token_ids.json` using
`dlb-upstream-token-capture-v1`; conversion does not parse the upstream text
file's human-oriented sample delimiters. Conversion requires exactly the
requested number of 1024-token rows and validates Task 7's runner-resolved
checkpoint digest, lock ID, selection, and `continuous_langflow` family before
reading the capture.

The upstream does not report per-sample latency. Canonical records therefore
use schema-permitted `0.0` only as an excluded sentinel; conversion metadata
states `unavailable_excluded_sentinel` and `exclude_from_latency: true`.

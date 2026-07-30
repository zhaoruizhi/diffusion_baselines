import math
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

from evaluation.generative_perplexity import (
    aggregate_nll,
    compute_gen_ppl,
    GPT2Assets,
    load_offline_gpt2_large,
    resolve_gpt2_assets,
)


class TinyTokenizer:
    pad_token_id = 0

    def __init__(self, rows: dict[str, list[int]]) -> None:
        self.rows = rows

    def __call__(self, texts, **kwargs):
        encoded = [self.rows[text][: kwargs["max_length"]] for text in texts]
        width = max(map(len, encoded))
        return {
            "input_ids": [row + [0] * (width - len(row)) for row in encoded],
            "attention_mask": [[1] * len(row) + [0] * (width - len(row)) for row in encoded],
        }


class FakeLogitScorer:
    """Reduce fixed logits without importing or running a model framework."""

    def __init__(self, logits_by_target: dict[int, list[float]]) -> None:
        self.logits_by_target = logits_by_target
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def score_batch(self, input_ids, attention_mask):
        total = 0.0
        count = 0
        for row, mask in zip(input_ids, attention_mask, strict=True):
            for target, valid in zip(row[1:], mask[1:], strict=True):
                if valid:
                    logits = self.logits_by_target[target]
                    maximum = max(logits)
                    log_partition = maximum + math.log(
                        sum(math.exp(value - maximum) for value in logits)
                    )
                    total += log_partition - logits[target]
                    count += 1
        return total, count


def test_ppl_uses_token_weighted_nll() -> None:
    """Catch averaging per-sample perplexities instead of all valid tokens."""

    result = aggregate_nll([(math.log(2), 1), (3 * math.log(4), 3)])

    assert result.perplexity == pytest.approx(
        math.exp((math.log(2) + 3 * math.log(4)) / 4)
    )
    assert result.total_nll == pytest.approx(math.log(2) + 3 * math.log(4))
    assert result.valid_token_count == 4


def test_compute_gen_ppl_is_batch_invariant_and_scores_final_short_batch() -> None:
    """Catch batch averaging or floor division silently dropping the last sample."""

    texts = ["a", "b", "c"]
    tokenizer = TinyTokenizer({"a": [9, 1], "b": [9, 2, 2], "c": [9, 3]})
    logits = {1: [0.0] * 2, 2: [0.0] * 4, 3: [0.0] * 8}

    one = compute_gen_ppl(texts, FakeLogitScorer(logits), tokenizer, batch_size=1)
    two = compute_gen_ppl(texts, FakeLogitScorer(logits), tokenizer, batch_size=2)

    expected_nll = math.log(2) + 2 * math.log(4) + math.log(8)
    assert one.perplexity == pytest.approx(math.exp(expected_nll / 4))
    assert two.perplexity == pytest.approx(one.perplexity)
    assert two.valid_token_count == 4
    assert two.sample_count == 3


def test_compute_gen_ppl_rejects_corpus_without_next_token_targets() -> None:
    """Catch publishing an undefined perplexity for one-token/empty targets."""

    tokenizer = TinyTokenizer({"one": [9]})

    with pytest.raises(ValueError, match="valid next-token"):
        compute_gen_ppl(["one"], FakeLogitScorer({}), tokenizer)


def test_compute_gen_ppl_rejects_any_sample_without_a_next_token_target() -> None:
    """Catch one short row disappearing inside an otherwise valid corpus aggregate."""

    tokenizer = TinyTokenizer({"short": [9], "valid": [9, 1]})

    with pytest.raises(ValueError, match="sample 0.*no valid next-token"):
        compute_gen_ppl(
            ["short", "valid"], FakeLogitScorer({1: [0.0, 0.0]}), tokenizer
        )


def test_compute_gen_ppl_rejects_non_right_padded_attention_mask() -> None:
    """Catch an interior padding hole shifting a target into the score."""

    class HoleTokenizer(TinyTokenizer):
        def __call__(self, texts, **kwargs):
            return {"input_ids": [[9, 0, 1]], "attention_mask": [[1, 0, 1]]}

    with pytest.raises(ValueError, match="right-padded"):
        compute_gen_ppl(
            ["row"], FakeLogitScorer({1: [0.0, 0.0]}), HoleTokenizer({})
        )


def test_compute_gen_ppl_rejects_scorer_token_count_drift() -> None:
    """Catch padding or first-position leakage hidden behind a plausible total NLL."""

    class WrongCountScorer(FakeLogitScorer):
        def score_batch(self, input_ids, attention_mask):
            nll, count = super().score_batch(input_ids, attention_mask)
            return nll, count + 1

    tokenizer = TinyTokenizer({"row": [9, 1]})

    with pytest.raises(ValueError, match="attention mask.*expected 1.*reported 2"):
        compute_gen_ppl(["row"], WrongCountScorer({1: [0.0, 0.0]}), tokenizer)


def test_compute_gen_ppl_rejects_context_beyond_gpt2_limit() -> None:
    """Catch silently requesting positions beyond GPT-2 Large's fixed context."""

    with pytest.raises(ValueError, match="must not exceed 1024"):
        compute_gen_ppl(
            ["row"],
            FakeLogitScorer({1: [0.0, 0.0]}),
            TinyTokenizer({"row": [9, 1]}),
            max_length=1025,
        )


@pytest.mark.parametrize(
    "parts",
    [[], [(float("nan"), 1)], [(1.0, 0)], [(float("inf"), 2)]],
)
def test_aggregate_nll_rejects_empty_or_nonfinite_aggregates(parts) -> None:
    """Catch non-finite or tokenless aggregate artifacts."""

    with pytest.raises(ValueError):
        aggregate_nll(parts)


def write_asset_manifests(root: Path, *, recorded_revision: str | None = None) -> None:
    revisions = {"gpt2": "a" * 40, "gpt2-large": "b" * 40}
    (root / "artifacts").mkdir()
    (root / "data" / "manifests").mkdir(parents=True)
    (root / "artifacts" / "data.yaml").write_text(
        yaml.safe_dump({"models": revisions}), encoding="utf-8"
    )
    models = {}
    for name, revision in revisions.items():
        path = root / "data" / "raw" / name / "snapshots" / revision
        path.mkdir(parents=True)
        models[name] = {
            "repo_id": name,
            "revision": recorded_revision if name == "gpt2-large" and recorded_revision else revision,
            "snapshot_path": path.relative_to(root).as_posix(),
        }
    (root / "data" / "manifests" / "downloads.json").write_text(
        json.dumps({"schema_version": 1, "models": models}), encoding="utf-8"
    )


def test_resolver_binds_model_and_tokenizer_to_pinned_local_snapshots(tmp_path: Path) -> None:
    """Catch a moving Hub alias or tokenizer/model revision mix-up."""

    write_asset_manifests(tmp_path)

    assets = resolve_gpt2_assets(tmp_path)

    assert assets.model_id == "gpt2-large"
    assert assets.model_revision == "b" * 40
    assert assets.model_path.name == "b" * 40
    assert assets.tokenizer_id == "gpt2"
    assert assets.tokenizer_revision == "a" * 40
    assert assets.tokenizer_path.name == "a" * 40


def test_resolver_rejects_download_record_at_another_revision(tmp_path: Path) -> None:
    """Catch silently evaluating from a stale or repointed local snapshot."""

    write_asset_manifests(tmp_path, recorded_revision="c" * 40)

    with pytest.raises(ValueError, match="gpt2-large revision"):
        resolve_gpt2_assets(tmp_path)


def test_resolver_rejects_unknown_download_manifest_schema(tmp_path: Path) -> None:
    """Catch interpreting an incompatible lock format as current provenance."""

    write_asset_manifests(tmp_path)
    downloads_path = tmp_path / "data" / "manifests" / "downloads.json"
    downloads = json.loads(downloads_path.read_text(encoding="utf-8"))
    downloads["schema_version"] = 2
    downloads_path.write_text(json.dumps(downloads), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        resolve_gpt2_assets(tmp_path)


def test_resolver_rejects_symlinked_snapshot_ancestor(tmp_path: Path) -> None:
    """Catch a lock path being redirected after the download manifest was written."""

    write_asset_manifests(tmp_path)
    downloads_path = tmp_path / "data" / "manifests" / "downloads.json"
    downloads = json.loads(downloads_path.read_text(encoding="utf-8"))
    real_parent = tmp_path / "data" / "raw" / "gpt2-large"
    alias = tmp_path / "data" / "raw" / "gpt2-large-alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    revision = "b" * 40
    downloads["models"]["gpt2-large"]["snapshot_path"] = (
        alias / "snapshots" / revision
    ).relative_to(tmp_path).as_posix()
    downloads_path.write_text(json.dumps(downloads), encoding="utf-8")

    with pytest.raises(ValueError, match="symlink"):
        resolve_gpt2_assets(tmp_path)


def test_loader_forces_local_snapshots_and_offline_mode(
    tmp_path: Path, monkeypatch
) -> None:
    """Catch Transformers falling back to a network model or moving alias."""

    calls = []

    class Tokenizer:
        pad_token_id = None
        eos_token_id = 7
        eos_token = "<eos>"
        pad_token = None

    class Model:
        def to(self, device):
            calls.append(("device", device))
            return self

        def eval(self):
            calls.append(("eval",))
            return self

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("tokenizer", path, kwargs))
            return Tokenizer()

    class AutoModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("model", path, kwargs))
            return Model()

    model_path = tmp_path / "model"
    tokenizer_path = tmp_path / "tokenizer"
    model_path.mkdir()
    tokenizer_path.mkdir()
    assets = GPT2Assets(
        "gpt2-large", "b" * 40, model_path, "gpt2", "a" * 40, tokenizer_path
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForCausalLM=AutoModel, AutoTokenizer=AutoTokenizer),
    )
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    _, tokenizer = load_offline_gpt2_large(assets, device="cuda:1")

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert ("tokenizer", str(tokenizer_path), {"local_files_only": True}) in calls
    assert ("model", str(model_path), {"local_files_only": True}) in calls
    assert ("device", "cuda:1") in calls
    assert ("eval",) in calls
    assert tokenizer.pad_token == "<eos>"
    assert tokenizer.padding_side == "right"
    assert tokenizer.truncation_side == "right"

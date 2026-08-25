import math

import pytest

from dlb.schema import ConditionalSampleRecord
from evaluation.conditional_perplexity import (
    compute_conditional_gen_ppl,
    conditional_texts,
)


class SourceTokenizer:
    def decode(self, token_ids, **kwargs):
        del kwargs
        row = list(token_ids)
        if len(row) == 64 and set(row) == {1}:
            return "prompt"
        if len(row) == 64 and set(row) == {2}:
            return "generated"
        if len(row) == 64 and set(row) == {3}:
            return "reference"
        if len(row) == 128 and row[:64] == [1] * 64 and row[64:] == [2] * 64:
            return "prompt generated"
        if len(row) == 960 and set(row) == {2}:
            return "full generated suffix"
        raise ValueError(f"unexpected decoded row length {len(row)}")


class ScorerTokenizer:
    def __call__(self, texts, **kwargs):
        rows = {
            "prompt": [9, 8],
            "prompt generated": [9, 8, 2, 3],
        }
        encoded = [rows[text][: kwargs["max_length"]] for text in texts]
        width = max(map(len, encoded))
        return {
            "input_ids": [row + [0] * (width - len(row)) for row in encoded],
            "attention_mask": [[1] * len(row) + [0] * (width - len(row)) for row in encoded],
        }


class PromptExcludedScorer:
    def __init__(self) -> None:
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def score_prompt_excluded_batch(self, input_ids, attention_mask, target_mask):
        del attention_mask
        total = 0.0
        count = 0
        for row, targets in zip(input_ids, target_mask, strict=True):
            for token, target in zip(row[1:], targets[1:], strict=True):
                if target:
                    assert token in {2, 3}
                    total += math.log(4)
                    count += 1
        return total, count


def record(*, suffix_length: int = 64) -> ConditionalSampleRecord:
    prefix = [1] * 64
    continuation = [2] * suffix_length
    return ConditionalSampleRecord(
        sample_id=0,
        prompt_id=0,
        completion_id=0,
        source_index=10,
        prefix_token_ids=prefix,
        continuation_token_ids=continuation,
        reference_token_ids=[3] * 64,
        full_token_ids=prefix + continuation,
        prefix_text="prompt",
        continuation_text="generated",
        reference_text="reference",
        full_text="prompt generated",
        seed=42,
        generation_seconds=0.0,
        prefix_exact_match=True,
    )


def test_conditional_ppl_excludes_prompt_tokens_from_nll() -> None:
    """Catch prompt tokens being included in conditional generative perplexity."""

    result = compute_conditional_gen_ppl(
        [record()],
        PromptExcludedScorer(),
        ScorerTokenizer(),
        SourceTokenizer(),
        batch_size=1,
    )

    assert result.valid_token_count == 2
    assert result.perplexity == pytest.approx(4.0)


def test_conditional_texts_decode_only_evaluation_suffix_for_long_canvas() -> None:
    """Catch OWT's 960-token generated suffix leaking into main quality metrics."""

    decoded = conditional_texts([record(suffix_length=960)], SourceTokenizer())

    assert decoded[0].generated_suffix == "generated"
    assert decoded[0].reference_suffix == "reference"
    assert decoded[0].prefix_and_generated_suffix == "prompt generated"

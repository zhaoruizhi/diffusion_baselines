import math

import pytest

import evaluation.self_bleu as self_bleu_module
from evaluation.self_bleu import SelfBleuConfig, compute_self_bleu


def test_self_bleu_detects_mode_collapse() -> None:
    """Catch an inverted or non-reference-based diversity score."""

    collapsed = ["the same four words here"] * 8
    diverse = [f"unique sentence number {i} token{i} extra{i}" for i in range(8)]

    collapsed_result = compute_self_bleu(collapsed)
    diverse_result = compute_self_bleu(diverse)

    assert collapsed_result.score > 0.99
    assert diverse_result.score < collapsed_result.score
    assert 0.0 <= diverse_result.score <= 1.0


def test_self_bleu_has_explicit_deterministic_standard_conventions() -> None:
    """Catch hidden random reference sampling or unrecorded BLEU conventions."""

    result = compute_self_bleu(
        ["a b c d e", "a b c d e", "q r s t u"], SelfBleuConfig()
    )

    assert result.ngram_order == 4
    assert result.weights == pytest.approx((0.25, 0.25, 0.25, 0.25))
    assert result.reference_rule == "all_other_samples"
    assert result.smoothing == "chen_cherry_method1"
    assert result.tokenization == "whitespace"
    assert result.sample_count == 3
    assert result.flm_revision == "a1918d5164e5038e37d0b7a4fb2010ce75b863b3"
    assert result.flm_metric_path == "metrics.py"
    assert result.flm_self_bleu_present is False


@pytest.mark.parametrize("texts", [[], ["only one sample"], ["ok", "   "]])
def test_self_bleu_rejects_missing_reference_or_empty_tokens(texts) -> None:
    """Catch undefined BLEU rows being normalized into plausible scores."""

    with pytest.raises(ValueError):
        compute_self_bleu(texts)


def test_self_bleu_builds_each_row_order_counter_once(monkeypatch) -> None:
    """Catch O(N²·L) n-gram reconstruction on the 1,024-sample OWT matrix."""

    original = self_bleu_module._ngrams
    calls = []

    def counted(tokens, order):
        calls.append((tuple(tokens), order))
        return original(tokens, order)

    monkeypatch.setattr(self_bleu_module, "_ngrams", counted)
    texts = [f"row {index} has five unique tokens{index}" for index in range(6)]

    compute_self_bleu(texts)

    assert len(calls) == len(texts) * 4
    assert len(set(calls)) == len(calls)


def test_method1_keeps_zero_when_candidate_has_no_unigram_reference_match() -> None:
    """Catch smoothing the BLEU unigram gate contrary to standard sentence BLEU."""

    result = compute_self_bleu(["a b c d", "w x y z"])

    assert result.score == 0.0


def test_leave_one_out_clipping_uses_second_max_for_unique_top_owner() -> None:
    """Catch a candidate clipping its own repeated n-grams as a reference."""

    result = compute_self_bleu(
        ["a a a a a", "a a", "b b b b b"],
        SelfBleuConfig(ngram_order=1, weights=(1.0,)),
    )

    assert result.score == pytest.approx((2 / 5 + math.exp(-1.5)) / 3)

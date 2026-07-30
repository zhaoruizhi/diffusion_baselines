"""Deterministic leave-one-out Self-BLEU.

The pinned FLM commit ``a1918d5164e5038e37d0b7a4fb2010ce75b863b3``
contains generative perplexity and per-sample entropy in ``metrics.py`` but no
Self-BLEU implementation anywhere in that repository.  Therefore this module
binds and reports an explicit standard convention instead of claiming a port:
whitespace tokens, BLEU-4 with equal weights, every other sample as references,
Chen & Cherry method-1 epsilon smoothing, and a mean sentence score in [0, 1].
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from bisect import bisect_left
import math
from typing import Sequence


@dataclass(frozen=True)
class SelfBleuConfig:
    ngram_order: int = 4
    weights: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    smoothing_epsilon: float = 0.1

    def __post_init__(self) -> None:
        if self.ngram_order < 1 or len(self.weights) != self.ngram_order:
            raise ValueError("weights must match a positive n-gram order")
        if any(weight <= 0 or not math.isfinite(weight) for weight in self.weights):
            raise ValueError("Self-BLEU weights must be finite and positive")
        if not math.isclose(math.fsum(self.weights), 1.0):
            raise ValueError("Self-BLEU weights must sum to one")
        if self.smoothing_epsilon <= 0 or not math.isfinite(self.smoothing_epsilon):
            raise ValueError("smoothing epsilon must be finite and positive")


@dataclass(frozen=True)
class SelfBleuResult:
    score: float
    sample_count: int
    ngram_order: int
    weights: tuple[float, ...]
    smoothing: str = "chen_cherry_method1"
    smoothing_epsilon: float = 0.1
    reference_rule: str = "all_other_samples"
    tokenization: str = "whitespace"
    normalization: str = "mean_sentence_bleu_0_1"
    flm_revision: str = "a1918d5164e5038e37d0b7a4fb2010ce75b863b3"
    flm_metric_path: str = "metrics.py"
    flm_self_bleu_present: bool = False


def _ngrams(tokens: Sequence[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def _global_maxima(
    all_counts: Sequence[Counter[tuple[str, ...]]],
) -> dict[tuple[str, ...], tuple[int, int, int]]:
    """Map n-gram to (top count, unique top owner or -1, second count)."""

    maxima: dict[tuple[str, ...], tuple[int, int, int]] = {}
    for sample_index, counts in enumerate(all_counts):
        for ngram, count in counts.items():
            top, owner, second = maxima.get(ngram, (0, -1, 0))
            if count > top:
                maxima[ngram] = (count, sample_index, top)
            elif count == top:
                maxima[ngram] = (top, -1, top)
            elif count > second:
                maxima[ngram] = (top, owner, count)
    return maxima


def _reference_lengths(lengths: Sequence[int]) -> list[int]:
    frequencies = Counter(lengths)
    unique = sorted(frequencies)
    result = []
    for candidate in lengths:
        if frequencies[candidate] > 1:
            result.append(candidate)
            continue
        position = bisect_left(unique, candidate)
        choices = []
        if position:
            choices.append(unique[position - 1])
        if position + 1 < len(unique):
            choices.append(unique[position + 1])
        result.append(min(choices, key=lambda length: (abs(length - candidate), length)))
    return result


def compute_self_bleu(
    texts: Sequence[str], config: SelfBleuConfig | None = None
) -> SelfBleuResult:
    """Average deterministic sentence BLEU against every other generated sample."""

    if len(texts) < 2:
        raise ValueError("Self-BLEU requires at least two samples")
    rows: list[list[str]] = []
    for index, text in enumerate(texts):
        if type(text) is not str or not text.strip():
            raise ValueError(f"Self-BLEU sample {index} has no whitespace tokens")
        rows.append(text.split())
    resolved = config or SelfBleuConfig()
    weighted_log_precision = [0.0] * len(rows)
    zero_unigram = [False] * len(rows)
    for order, weight in enumerate(resolved.weights, start=1):
        counts = [_ngrams(row, order) for row in rows]
        maxima = _global_maxima(counts)
        for candidate_index, candidate_counts in enumerate(counts):
            denominator = sum(candidate_counts.values())
            numerator = 0
            for ngram, count in candidate_counts.items():
                top, unique_owner, second = maxima[ngram]
                reference_count = second if unique_owner == candidate_index else top
                numerator += min(count, reference_count)
            if order == 1 and numerator == 0:
                zero_unigram[candidate_index] = True
                continue
            precision = (
                resolved.smoothing_epsilon / max(denominator, 1)
                if numerator == 0
                else numerator / denominator
            )
            weighted_log_precision[candidate_index] += weight * math.log(precision)
    reference_lengths = _reference_lengths([len(row) for row in rows])
    scores = []
    for index, row in enumerate(rows):
        if zero_unigram[index]:
            scores.append(0.0)
            continue
        candidate_length = len(row)
        reference_length = reference_lengths[index]
        brevity_penalty = (
            1.0
            if candidate_length > reference_length
            else math.exp(1.0 - reference_length / candidate_length)
        )
        scores.append(brevity_penalty * math.exp(weighted_log_precision[index]))
    score = math.fsum(scores) / len(scores)
    if not math.isfinite(score) or score < 0 or score > 1 + 1e-12:
        raise ValueError("Self-BLEU score is outside [0, 1]")
    return SelfBleuResult(
        score=min(score, 1.0),
        sample_count=len(rows),
        ngram_order=resolved.ngram_order,
        weights=resolved.weights,
        smoothing_epsilon=resolved.smoothing_epsilon,
    )

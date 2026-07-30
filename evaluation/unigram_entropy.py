"""Mean per-sample empirical unigram entropy in natural-log units."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class EntropyResult:
    mean_entropy: float
    sample_count: int
    token_count: int
    excluded_token_ids: tuple[int, ...]
    aggregation: str = "arithmetic_mean_of_per_sample_entropy"
    logarithm: str = "natural"
    unit: str = "nats"


def unigram_entropy(token_ids: Sequence[int]) -> float:
    """Return the natural-log empirical entropy of one non-empty token row."""

    if not token_ids:
        raise ValueError("token row has no tokens")
    if not all(type(token_id) is int and token_id >= 0 for token_id in token_ids):
        raise ValueError("token IDs must be non-negative integers")
    total = len(token_ids)
    return -sum(
        (count / total) * math.log(count / total) for count in Counter(token_ids).values()
    )


def _token_ids(record: object, index: int) -> Sequence[int]:
    if isinstance(record, Mapping):
        value = record.get("token_ids")
    else:
        value = getattr(record, "token_ids", None)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"record {index} has invalid token_ids")
    return value


def mean_unigram_entropy(
    records: Iterable[object], special_ids: Iterable[int] = ()
) -> EntropyResult:
    """Remove exactly the caller-declared padding IDs, then average row entropy."""

    excluded = frozenset(special_ids)
    if not all(type(token_id) is int and token_id >= 0 for token_id in excluded):
        raise ValueError("special token IDs must be non-negative integers")
    entropies: list[float] = []
    token_count = 0
    for index, record in enumerate(records):
        row = [token_id for token_id in _token_ids(record, index) if token_id not in excluded]
        if not row:
            raise ValueError(f"record {index} has no tokens after exclusions")
        entropies.append(unigram_entropy(row))
        token_count += len(row)
    if not entropies:
        raise ValueError("no sample records were provided")
    result = math.fsum(entropies) / len(entropies)
    if not math.isfinite(result):
        raise ValueError("mean entropy is not finite")
    return EntropyResult(
        mean_entropy=result,
        sample_count=len(entropies),
        token_count=token_count,
        excluded_token_ids=tuple(sorted(excluded)),
    )

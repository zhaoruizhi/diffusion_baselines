import json
from pathlib import Path


def _write_samples(root: Path, step: int, texts: list[str]) -> Path:
    sample_dir = root / "results" / "samples" / "lm1b" / "rdlm" / f"steps_{step}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    with (sample_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for index, text in enumerate(texts):
            handle.write(
                json.dumps(
                    {
                        "sample_id": index,
                        "text": text,
                        "token_ids": [0, 101, index + 1, 102],
                        "seed": 42,
                        "generation_seconds": 0.0,
                    }
                )
                + "\n"
            )
    return sample_dir


def _write_metrics(
    root: Path, step: int, *, perplexity: float, entropy: float, self_bleu: float
) -> None:
    metrics_path = (
        root / "results" / "metrics" / "lm1b" / "rdlm" / f"steps_{step}" / "metrics.json"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sample_count": 4,
                "partial": False,
                "metrics": {
                    "generative_perplexity": {"perplexity": perplexity},
                    "unigram_entropy": {"mean_entropy": entropy},
                    "self_bleu": {"score": self_bleu},
                },
            }
        ),
        encoding="utf-8",
    )


def _write_conversion(sample_dir: Path, source: str, reason: str | None = None) -> None:
    transformation = {"operation": "validated_exact_length"}
    if reason is not None:
        transformation["reason"] = reason
    (sample_dir / "conversion_metadata.json").write_text(
        json.dumps(
            {
                "format": "dlb-upstream-token-capture-v1",
                "token_ids_source": source,
                "token_ids_transformation": transformation,
            }
        ),
        encoding="utf-8",
    )


def test_rdlm_diagnostics_flags_collapsed_low_ppl_step(tmp_path: Path) -> None:
    """Catch reading a low generative-PPL row as a quality win despite collapse signals."""

    from dlb.rdlm_diagnostics import diagnose_rdlm

    collapsed_dir = _write_samples(tmp_path, 1, ["the same short sample"] * 4)
    _write_metrics(tmp_path, 1, perplexity=22.7, entropy=3.2, self_bleu=0.99)
    _write_conversion(
        collapsed_dir, "retokenized", "upstream_ids_not_canonical_length"
    )
    normal_dir = _write_samples(
        tmp_path,
        32,
        [
            "a longer generated sentence about policy and finance with enough separate clauses to resemble a full sample",
            "another generated sentence with separate wording and enough content to avoid the short text warning path",
            "markets discussed a different event in parliament while officials compared several economic indicators",
            "officials released new numbers after the meeting and said the report would be reviewed next month",
        ],
    )
    _write_metrics(tmp_path, 32, perplexity=375.94, entropy=4.33, self_bleu=0.12)
    _write_conversion(normal_dir, "upstream")

    report = diagnose_rdlm(tmp_path, steps=(1, 32), expected_sample_count=4)

    assert report["schema"] == "dlb-rdlm-diagnostics-v1"
    assert report["summary"]["suspect_steps"] == [1]
    assert report["summary"]["verdict"] == "low_ppl_points_show_collapse_signals"
    collapsed = report["rows"][0]
    assert collapsed["steps"] == 1
    assert collapsed["unique_text_ratio"] == 0.25
    assert collapsed["token_ids_source"] == "retokenized"
    assert collapsed["token_ids_reason"] == "upstream_ids_not_canonical_length"
    assert {
        "low_ppl_with_low_entropy",
        "high_self_bleu",
        "high_duplicate_texts",
        "retokenized_from_text",
    } <= set(collapsed["warnings"])


def test_rdlm_diagnostics_reports_missing_step_artifacts(tmp_path: Path) -> None:
    """Catch silently omitting RDLM steps whose server artifacts were never produced."""

    from dlb.rdlm_diagnostics import diagnose_rdlm

    report = diagnose_rdlm(tmp_path, steps=(1024,), expected_sample_count=4)

    assert report["summary"]["missing_steps"] == [1024]
    assert report["rows"][0]["warnings"] == ["missing_samples", "missing_metrics"]

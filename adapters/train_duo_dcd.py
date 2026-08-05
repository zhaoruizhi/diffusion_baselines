"""Run Duo+DCD with a narrow compatibility patch for the pinned Duo source."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


def _patch_distillation_nll() -> None:
    import algo

    original = algo.Distillation.nll
    if getattr(original, "_dlb_accepts_labels", False):
        return

    def nll(
        self: Any,
        x0: Any,
        labels_or_output_tokens: Any = None,
        output_tokens: Any = None,
        current_accumulation_step: Any = None,
        train_mode: Any = None,
    ) -> Any:
        if output_tokens is None:
            output_tokens = labels_or_output_tokens
        return original(
            self,
            x0,
            output_tokens,
            current_accumulation_step,
            train_mode,
        )

    nll._dlb_accepts_labels = True  # type: ignore[attr-defined]
    algo.Distillation.nll = nll


def main() -> int:
    sys.path.insert(0, str(Path.cwd()))
    _patch_distillation_nll()
    from main import main as upstream_main

    return int(upstream_main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())

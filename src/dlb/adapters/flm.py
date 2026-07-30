"""Adapter for pinned FLM and FMLM checkpoints."""

from dlb.adapters.base import BaseTeacherAdapter, CheckpointSelection
from dlb.runner import RunRequest


class FLMAdapter(BaseTeacherAdapter):
    identity = "dlb.adapters.flm:v1"
    upstream = "flm"
    supported_models = frozenset({"flm", "fmlm"})
    teacher_families = {"flm": "continuous_flm", "fmlm": "continuous_flm"}
    batch_sizes = {
        ("flm", "lm1b"): 32,
        ("flm", "owt"): 16,
        ("fmlm", "lm1b"): 16,
        ("fmlm", "owt"): 16,
    }

    def _sampling_overrides(
        self, request: RunRequest, checkpoint: CheckpointSelection
    ) -> list[str]:
        del checkpoint
        overrides = [
            f"algo={request.model_id}",
            "algo.backbone=hf_dit",
            "eval.disable_ema=False",
            "sampling.solver=euler",
        ]
        if request.model_id == "flm":
            overrides.extend(["algo.double_temb=False", "sampling.gamma=0.0"])
        else:
            gamma = "0.8" if request.dataset_id == "lm1b" else "1.0"
            overrides.extend(
                [
                    "algo.double_temb=True",
                    "algo.learnable_loss_weighting=False",
                    f"sampling.gamma={gamma}",
                ]
            )
        return overrides


adapter = FLMAdapter()

"""Adapter for the pinned masked discrete language model."""

from dlb.adapters.base import BaseTeacherAdapter, CheckpointSelection
from dlb.runner import RunRequest


class MDLMAdapter(BaseTeacherAdapter):
    identity = "dlb.adapters.mdlm:v1"
    upstream = "mdlm"
    supported_models = frozenset({"mdlm"})
    teacher_families = {"mdlm": "masked_mdlm"}
    batch_sizes = {("mdlm", "lm1b"): 16, ("mdlm", "owt"): 16}

    def _data_config(self, request: RunRequest) -> str:
        return "lm1b" if request.dataset_id == "lm1b" else "openwebtext-split"

    def _sampling_overrides(
        self, request: RunRequest, checkpoint: CheckpointSelection
    ) -> list[str]:
        del checkpoint
        backbone = "hf_dit" if request.dataset_id == "owt" else "dit"
        return [
            f"backbone={backbone}",
            "parameterization=subs",
            "sampling.predictor=ddpm",
            "sampling.noise_removal=True",
            "eval.disable_ema=False",
        ]


adapter = MDLMAdapter()

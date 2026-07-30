"""Adapter for pinned Duo and discrete-consistency-distilled Duo."""

from dlb.adapters.base import BaseTeacherAdapter, CheckpointSelection
from dlb.runner import RunRequest


class DuoAdapter(BaseTeacherAdapter):
    identity = "dlb.adapters.duo:v1"
    upstream = "duo"
    supported_models = frozenset({"duo", "duo_dcd"})
    teacher_families = {"duo": "uniform_duo", "duo_dcd": "uniform_duo"}
    batch_sizes = {
        ("duo", "lm1b"): 64,
        ("duo", "owt"): 8,
        ("duo_dcd", "lm1b"): 64,
        ("duo_dcd", "owt"): 8,
    }

    def _sampling_overrides(
        self, request: RunRequest, checkpoint: CheckpointSelection
    ) -> list[str]:
        del checkpoint
        backbone = "hf_dit" if request.dataset_id == "owt" else "dit"
        return [
            "algo=duo_base",
            f"algo.backbone={backbone}",
            "sampling.predictor=ancestral",
            "sampling.noise_removal=ancestral",
            "eval.disable_ema=False",
        ]


adapter = DuoAdapter()

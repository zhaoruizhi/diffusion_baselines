"""Adapter for the pinned CANDI hybrid discrete/continuous sampler."""

from dlb.adapters.base import BaseTeacherAdapter, CheckpointSelection
from dlb.runner import RunRequest


class CANDIAdapter(BaseTeacherAdapter):
    identity = "dlb.adapters.candi:v1"
    upstream = "candi"
    supported_models = frozenset({"candi"})
    teacher_families = {"candi": "hybrid_candi"}
    batch_sizes = {("candi", "lm1b"): 2, ("candi", "owt"): 2}

    def _loader_batch_size(self, request: RunRequest) -> int:
        del request
        return 16

    def _sampling_overrides(
        self, request: RunRequest, checkpoint: CheckpointSelection
    ) -> list[str]:
        del request, checkpoint
        return [
            "algo=candi",
            "algo.sampler=cached",
            "algo.mixed_coeff=0.5",
            "algo.step_size=1.0",
            "algo.temp=1.0",
            "algo.use_percentile_scheduling=True",
            "sampling.noise_removal=ancestral",
            "trainer.accumulate_grad_batches=1",
            "eval.disable_ema=False",
        ]


adapter = CANDIAdapter()

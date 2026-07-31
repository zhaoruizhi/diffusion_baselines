#!/usr/bin/env bash
# Train/distill one paper-locked recipe. Site resources are supplied to sbatch.
#SBATCH --job-name=dlb-train
#SBATCH --output=results/logs/slurm-train-%A.out
#SBATCH --error=results/logs/slurm-train-%A.err
set -euo pipefail

ROOT="${DLB_ROOT:-${SLURM_SUBMIT_DIR}}"
RECIPE="${DLB_RECIPE:?set DLB_RECIPE (for example flm or mdlm_sdtt)}"
DATASET="${DLB_DATASET:?set DLB_DATASET (lm1b or owt)}"
MODEL="${DLB_MODEL:-}"
registry_model="$RECIPE"
if [[ "$RECIPE" == "di4c" ]]; then
  registry_model="${MODEL:?DLB_MODEL is required for di4c}"
fi
environment="$(
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "${DLB_PYTHON:-python}" - "$ROOT" "$registry_model" <<'PY'
from pathlib import Path
import sys
from dlb.registry import load_registry
root, model = Path(sys.argv[1]), sys.argv[2]
registry = load_registry(root / "configs/experiments.yaml")
print(registry.models[model].environment)
PY
)"
mkdir -p "$ROOT/results/logs"
args=(--root "$ROOT" --recipe "$RECIPE" --dataset "$DATASET")
[[ -n "$MODEL" ]] && args+=(--model "$MODEL")
exec "${DLB_CONDA:-conda}" run -n "$environment" python -m dlb.recipes "${args[@]}"

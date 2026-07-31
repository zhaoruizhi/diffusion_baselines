#!/usr/bin/env bash
# Site-specific resources are supplied to sbatch by the operator.
#SBATCH --job-name=dlb-evaluate
#SBATCH --output=results/logs/slurm-evaluate-%A.out
#SBATCH --error=results/logs/slurm-evaluate-%A.err
set -euo pipefail

ROOT="${DLB_ROOT:-${SLURM_SUBMIT_DIR}}"
mkdir -p "$ROOT/results/logs"
exec "${DLB_CONDA:-conda}" run -n "${DLB_EVAL_ENV:-dlb-eval}" \
  bash "$ROOT/scripts/evaluate_all.sh" --root "$ROOT"

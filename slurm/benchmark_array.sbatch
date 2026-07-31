#!/usr/bin/env bash
# Submit with an explicit site-specific array, e.g. --array=0-129.
#SBATCH --job-name=dlb-benchmark
#SBATCH --output=results/logs/slurm-benchmark-%A_%a.out
#SBATCH --error=results/logs/slurm-benchmark-%A_%a.err
set -euo pipefail

ROOT="${DLB_ROOT:-${SLURM_SUBMIT_DIR}}"
MATRIX="${DLB_MATRIX:-$ROOT/results/matrix/generation.tsv}"
if [[ ! -f "$MATRIX" ]]; then
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "${DLB_PYTHON:-python}" -m dlb.matrix \
    --root "$ROOT" --output "$MATRIX" --sample-count 1024 --seed 42
fi
line_number=$((SLURM_ARRAY_TASK_ID + 3))
IFS=$'\t' read -r task_id category model dataset steps sample_count seed environment adapter source provenance sample_dir metrics_path timing_path \
  < <(sed -n "${line_number}p" "$MATRIX")
[[ -n "${task_id:-}" && "$task_id" != "task_id" ]] || { echo "invalid matrix row $line_number" >&2; exit 2; }
mkdir -p "$ROOT/results/logs"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${DLB_CONDA:-conda}" run -n "$environment" python -m dlb.benchmarking \
  --root "$ROOT" --models "$model" --datasets "$dataset" --steps "$steps" \
  --seed "$seed" --precision author

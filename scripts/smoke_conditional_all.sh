#!/usr/bin/env bash
# Conditional smoke pass. It still uses the full 2,048-record C64 schedule,
# because partial conditional artifacts are not comparable to production runs.
set -uo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DLB_ROOT="${DLB_ROOT:-$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)}"
PYTHON_BIN="${DLB_PYTHON:-python}"
MATRIX="${DLB_ROOT}/results/conditional/matrix/smoke.tsv"
if [[ "${1:-}" == "--help" ]]; then
  printf 'Usage: %s\n' "$0"
  exit 0
fi
mkdir -p "$DLB_ROOT/results/conditional/matrix" "$DLB_ROOT/results/logs"
PYTHONPATH="$DLB_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m dlb.conditional_matrix \
  --root "$DLB_ROOT" --output "$MATRIX" || exit $?

status=0
while IFS=$'\t' read -r task_id category model dataset steps sample_count seed environment adapter source provenance protocol conditioning_manifest conditioning_manifest_sha256 sample_dir metrics_path timing_path; do
  [[ "$task_id" == "task_id" || -z "$task_id" ]] && continue
  echo "CONDITIONAL SMOKE $task_id"
  "$SCRIPT_DIR/run_conditional_one.sh" --model "$model" --dataset "$dataset" --steps "$steps" \
    --seed "$seed" --results-root "$DLB_ROOT/results/conditional/smoke"
  if (($? != 0)); then status=1; fi
done < <(tail -n +3 "$MATRIX")
exit "$status"

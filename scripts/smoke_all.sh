#!/usr/bin/env bash
# Server-side one-sample smoke pass. Its sample_count=1 identity cannot satisfy
# the production 1,024-sample aggregation contract.
set -uo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DLB_ROOT="${DLB_ROOT:-$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)}"
PYTHON_BIN="${DLB_PYTHON:-python}"
MATRIX="${DLB_ROOT}/results/matrix/smoke.tsv"
mkdir -p "$DLB_ROOT/results/matrix" "$DLB_ROOT/results/logs"
PYTHONPATH="$DLB_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m dlb.matrix \
  --root "$DLB_ROOT" --output "$MATRIX" --sample-count 1 --seed 42 || exit $?

status=0
while IFS=$'\t' read -r task_id category model dataset steps sample_count seed environment adapter source provenance sample_dir metrics_path timing_path; do
  [[ "$task_id" == "task_id" || -z "$task_id" ]] && continue
  echo "SMOKE $task_id"
  "$SCRIPT_DIR/run_one.sh" --model "$model" --dataset "$dataset" --steps "$steps" \
    --num-samples 1 --seed "$seed" --results-root "$DLB_ROOT/results/smoke"
  if (($? != 0)); then status=1; fi
done < <(tail -n +3 "$MATRIX")
exit "$status"

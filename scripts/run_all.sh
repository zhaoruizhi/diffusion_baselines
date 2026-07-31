#!/usr/bin/env bash
# Server-side serial/resumable generation over the canonical matrix.
set -uo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DLB_ROOT="${DLB_ROOT:-$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)}"
PYTHON_BIN="${DLB_PYTHON:-python}"
MATRIX="${DLB_MATRIX:-}"
SAMPLE_COUNT=1024
SEED=42

usage() {
  printf 'Usage: %s [--root PATH] [--matrix PATH] [--num-samples N] [--seed N]\n' "$0"
}

while (($#)); do
  case "$1" in
    --root) DLB_ROOT="$2"; shift 2 ;;
    --matrix) MATRIX="$2"; shift 2 ;;
    --num-samples) SAMPLE_COUNT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

DLB_ROOT="$(CDPATH= cd -- "$DLB_ROOT" && pwd -P)"
if [[ -z "$MATRIX" ]]; then MATRIX="$DLB_ROOT/results/matrix/generation.tsv"; fi
export DLB_ROOT
mkdir -p "$DLB_ROOT/results/matrix" "$DLB_ROOT/results/logs"
PYTHONPATH="$DLB_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m dlb.matrix \
  --root "$DLB_ROOT" --output "$MATRIX" --sample-count "$SAMPLE_COUNT" --seed "$SEED" \
  || exit $?

failures="$DLB_ROOT/results/logs/generation_failures.tsv"
printf 'task_id\tmodel\tdataset\tsteps\texit_code\n' >"$failures"
status=0
while IFS=$'\t' read -r task_id category model dataset steps sample_count seed environment adapter source provenance sample_dir metrics_path timing_path; do
  [[ "$task_id" == "task_id" || -z "$task_id" ]] && continue
  echo "GENERATE $task_id"
  "$SCRIPT_DIR/run_one.sh" \
    --model "$model" --dataset "$dataset" --steps "$steps" \
    --num-samples "$sample_count" --seed "$seed"
  exit_code=$?
  if ((exit_code != 0)); then
    printf '%s\t%s\t%s\t%s\t%s\n' "$task_id" "$model" "$dataset" "$steps" "$exit_code" >>"$failures"
    status=1
  fi
done < <(tail -n +3 "$MATRIX")
if [[ ! -s "$failures" ]]; then
  rm -f "$failures"
fi
exit "$status"

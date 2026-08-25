#!/usr/bin/env bash
# Server-side quality evaluation over completed C64 conditional samples.
set -uo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DLB_ROOT="${DLB_ROOT:-$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)}"
PYTHON_BIN="${DLB_EVAL_PYTHON:-${DLB_PYTHON:-python}}"
MATRIX="${DLB_CONDITIONAL_MATRIX:-}"
METRICS="conditional_gen_ppl,mauve_suffix,entropy,self_bleu,prefix_exact_match"

usage() {
  printf 'Usage: %s [--root PATH] [--matrix PATH] [--metrics LIST]\n' "$0"
}

while (($#)); do
  case "$1" in
    --root) DLB_ROOT="$2"; shift 2 ;;
    --matrix) MATRIX="$2"; shift 2 ;;
    --metrics) METRICS="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

DLB_ROOT="$(CDPATH= cd -- "$DLB_ROOT" && pwd -P)"
if [[ -z "$MATRIX" ]]; then MATRIX="$DLB_ROOT/results/conditional/matrix/generation.tsv"; fi
export DLB_ROOT
mkdir -p "$DLB_ROOT/results/conditional/matrix" "$DLB_ROOT/results/logs"
PYTHONPATH="$DLB_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "${DLB_PYTHON:-python}" -m dlb.conditional_matrix \
  --root "$DLB_ROOT" --output "$MATRIX" || exit $?

failures="$DLB_ROOT/results/logs/conditional_evaluation_failures.tsv"
printf 'task_id\tmodel\tdataset\tsteps\texit_code\n' >"$failures"
status=0
while IFS=$'\t' read -r task_id category model dataset steps sample_count seed environment adapter source provenance protocol conditioning_manifest conditioning_manifest_sha256 sample_dir metrics_path timing_path; do
  [[ "$task_id" == "task_id" || -z "$task_id" ]] && continue
  mkdir -p "$(dirname -- "$metrics_path")"
  echo "CONDITIONAL EVALUATE $task_id"
  PYTHONPATH="$DLB_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m evaluation.conditional_evaluate \
    --root "$DLB_ROOT" --samples "$sample_dir/samples.jsonl" \
    --metrics "$METRICS" --dataset "$dataset" --output "$metrics_path"
  exit_code=$?
  if ((exit_code != 0)); then
    printf '%s\t%s\t%s\t%s\t%s\n' "$task_id" "$model" "$dataset" "$steps" "$exit_code" >>"$failures"
    status=1
  fi
done < <(tail -n +3 "$MATRIX")
if [[ ! -s "$failures" ]]; then rm -f "$failures"; fi
exit "$status"

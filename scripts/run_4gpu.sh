#!/usr/bin/env bash
# Four-GPU local generation over the canonical matrix.
set -uo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DLB_ROOT="${DLB_ROOT:-$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)}"
PYTHON_BIN="${DLB_PYTHON:-python}"
MATRIX="${DLB_MATRIX:-}"
CUSTOM_MATRIX=0
if [[ -n "$MATRIX" ]]; then CUSTOM_MATRIX=1; fi
SAMPLE_COUNT=1024
SEED=42
GPUS="${DLB_GPUS:-0,1,2,3}"
MAX_JOBS="${DLB_MAX_JOBS:-4}"
DRY_RUN=()

usage() {
  printf 'Usage: %s [--root PATH] [--matrix PATH] [--num-samples N] [--seed N] [--gpus LIST] [--max-jobs N] [--dry-run]\n' "$0"
}

while (($#)); do
  case "$1" in
    --root) DLB_ROOT="$2"; shift 2 ;;
    --matrix) MATRIX="$2"; CUSTOM_MATRIX=1; shift 2 ;;
    --num-samples) SAMPLE_COUNT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --max-jobs) MAX_JOBS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=(--dry-run); shift ;;
    --help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

DLB_ROOT="$(CDPATH= cd -- "$DLB_ROOT" && pwd -P)"
if [[ -z "$MATRIX" ]]; then MATRIX="$DLB_ROOT/results/matrix/generation.tsv"; fi
export DLB_ROOT
mkdir -p "$DLB_ROOT/results/matrix" "$DLB_ROOT/results/logs"
if (( CUSTOM_MATRIX )); then
  if [[ ! -f "$MATRIX" ]]; then
    echo "custom matrix does not exist: $MATRIX" >&2
    exit 2
  fi
else
  PYTHONPATH="$DLB_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m dlb.matrix \
    --root "$DLB_ROOT" --output "$MATRIX" --sample-count "$SAMPLE_COUNT" --seed "$SEED" \
    || exit $?
fi
exec env PYTHONPATH="$DLB_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m dlb.gpu_matrix \
  --root "$DLB_ROOT" --stage generate --matrix "$MATRIX" \
  --gpus "$GPUS" --max-jobs "$MAX_JOBS" "${DRY_RUN[@]}"

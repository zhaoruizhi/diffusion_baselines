#!/usr/bin/env bash
# Four-GPU local timing pass over the canonical generation matrix.
set -uo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DLB_ROOT="${DLB_ROOT:-$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)}"
PYTHON_BIN="${DLB_PYTHON:-python}"
MATRIX="${DLB_MATRIX:-}"
SEED=42
GPUS="${DLB_GPUS:-0,1,2,3}"
MAX_JOBS="${DLB_MAX_JOBS:-4}"
DRY_RUN=()

usage() {
  printf 'Usage: %s [--root PATH] [--matrix PATH] [--seed N] [--gpus LIST] [--max-jobs N] [--dry-run]\n' "$0"
}

while (($#)); do
  case "$1" in
    --root) DLB_ROOT="$2"; shift 2 ;;
    --matrix) MATRIX="$2"; shift 2 ;;
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
PYTHONPATH="$DLB_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m dlb.matrix \
  --root "$DLB_ROOT" --output "$MATRIX" --sample-count 1024 --seed "$SEED" || exit $?
exec env PYTHONPATH="$DLB_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m dlb.gpu_matrix \
  --root "$DLB_ROOT" --stage benchmark --matrix "$MATRIX" \
  --gpus "$GPUS" --max-jobs "$MAX_JOBS" --precision author "${DRY_RUN[@]}"

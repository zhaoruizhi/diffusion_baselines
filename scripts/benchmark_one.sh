#!/usr/bin/env bash
# Run one server-side, registry-bound primary-latency benchmark.
set -u

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DLB_ROOT="${DLB_ROOT:-$(CDPATH= cd -- "$script_dir/.." && pwd -P)}"

model=""
dataset=""
steps=""
seed=""
precision=""
dry_run="false"

usage() {
  echo "usage: $0 --model MODEL --dataset {lm1b,owt} --steps N --seed N --precision author [--dry-run]" >&2
}

while (( $# )); do
  case "$1" in
    --model|--dataset|--steps|--seed|--precision)
      if (( $# < 2 )); then usage; exit 2; fi
      case "$1" in
        --model) model="$2" ;;
        --dataset) dataset="$2" ;;
        --steps) steps="$2" ;;
        --seed) seed="$2" ;;
        --precision) precision="$2" ;;
      esac
      shift 2 ;;
    --dry-run) dry_run="true"; shift ;;
    --help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$model" || -z "$dataset" || -z "$steps" || -z "$seed" || -z "$precision" ]]; then
  usage; exit 2
fi
if [[ ! "$steps" =~ ^[1-9][0-9]*$ || ! "$seed" =~ ^-?[0-9]+$ ]]; then
  echo "steps must be positive and seed must be an integer" >&2; exit 2
fi
if [[ "$precision" != "author" ]]; then
  echo "precision must be author (the pinned upstream inference policy)" >&2; exit 2
fi

DLB_PYTHON="${DLB_PYTHON:-python}"
module_args=(--root "$DLB_ROOT" --models "$model" --datasets "$dataset" --steps "$steps" --seed "$seed" --precision "$precision")
if [[ "$dry_run" == "true" ]]; then
  exec env PYTHONPATH="$DLB_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$DLB_PYTHON" -m dlb.benchmarking "${module_args[@]}" --dry-run
fi

if environment="$(PYTHONPATH="$DLB_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$DLB_PYTHON" -m dlb.runner \
  --root "$DLB_ROOT" --model "$model" --dataset "$dataset" --steps "$steps" \
  --num-samples 1 --seed "$seed" --validate-only)"; then
  :
else
  exit $?
fi
if [[ ! "$environment" =~ ^[a-z][a-z0-9-]*$ ]]; then
  echo "registry validation returned an unsafe environment name" >&2; exit 2
fi

exec "${DLB_CONDA:-conda}" run -n "$environment" env \
  PYTHONPATH="$DLB_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  python -m dlb.benchmarking "${module_args[@]}"

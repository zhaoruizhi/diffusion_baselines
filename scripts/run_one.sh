#!/usr/bin/env bash
# Run one registry-backed request without interpreting user input as shell code.
set -u

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DLB_ROOT="${DLB_ROOT:-$(CDPATH= cd -- "$script_dir/.." && pwd -P)}"

model=""
dataset=""
steps=""
num_samples=""
seed=""
device=""
results_root=""

usage() {
  echo "usage: $0 --model MODEL --dataset {lm1b,owt} --steps N --num-samples N --seed N [--device DEVICE] [--results-root PATH]" >&2
}

while (( $# )); do
  case "$1" in
    --model|--dataset|--steps|--num-samples|--seed|--device|--results-root)
      if (( $# < 2 )); then usage; exit 2; fi
      case "$1" in
        --model) model="$2" ;;
        --dataset) dataset="$2" ;;
        --steps) steps="$2" ;;
        --num-samples) num_samples="$2" ;;
        --seed) seed="$2" ;;
        --device) device="$2" ;;
        --results-root) results_root="$2" ;;
      esac
      shift 2 ;;
    --help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$model" || -z "$dataset" || -z "$steps" || -z "$num_samples" || -z "$seed" ]]; then
  usage; exit 2
fi
if [[ ! "$steps" =~ ^[0-9]+$ || ! "$num_samples" =~ ^[1-9][0-9]*$ || ! "$seed" =~ ^-?[0-9]+$ ]]; then
  echo "steps, num-samples, and seed must be integers; num-samples must be positive" >&2
  exit 2
fi

DLB_PYTHON="${DLB_PYTHON:-python}"
if environment="$(PYTHONPATH="$DLB_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$DLB_PYTHON" -m dlb.runner \
  --root "$DLB_ROOT" --model "$model" --dataset "$dataset" --steps "$steps" \
  --num-samples "$num_samples" --seed "$seed" --validate-only)"; then
  :
else
  exit $?
fi
if [[ ! "$environment" =~ ^[a-z][a-z0-9-]*$ ]]; then
  echo "registry validation returned an unsafe environment name" >&2; exit 2
fi

runner_args=(--root "$DLB_ROOT" --model "$model" --dataset "$dataset" --steps "$steps" --num-samples "$num_samples" --seed "$seed")
if [[ -n "$device" ]]; then runner_args+=(--device "$device"); fi
if [[ -n "$results_root" ]]; then runner_args+=(--results-root "$results_root"); fi
exec "${DLB_CONDA:-conda}" run -n "$environment" env \
  PYTHONPATH="$DLB_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  python -m dlb.runner "${runner_args[@]}"

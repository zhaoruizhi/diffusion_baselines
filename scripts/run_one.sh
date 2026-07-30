#!/usr/bin/env bash
# Run one registry-backed request without interpreting user input as shell code.
set -u

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DLB_ROOT="$(CDPATH= cd -- "$script_dir/.." && pwd -P)"

model=""
dataset=""
steps=""
num_samples=""
seed=""
device=""

usage() {
  echo "usage: $0 --model MODEL --dataset {lm1b,owt} --steps N --num-samples N --seed N [--device DEVICE]" >&2
}

while (( $# )); do
  case "$1" in
    --model|--dataset|--steps|--num-samples|--seed|--device)
      if (( $# < 2 )); then usage; exit 2; fi
      case "$1" in
        --model) model="$2" ;;
        --dataset) dataset="$2" ;;
        --steps) steps="$2" ;;
        --num-samples) num_samples="$2" ;;
        --seed) seed="$2" ;;
        --device) device="$2" ;;
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

# The canonical registry format is deliberately simple; extract only this model's
# category, environment, and selected dataset status without evaluating YAML.
registry="$DLB_ROOT/configs/experiments.yaml"
if [[ ! -f "$registry" ]]; then echo "missing registry: $registry" >&2; exit 2; fi
registry_row="$({
  awk -v wanted_model="$model" -v wanted_dataset="$dataset" '
    $1 == wanted_model ":" { active=1; wanted=0; next }
    active && /^  [a-zA-Z0-9_]+:/ { exit }
    active && $1 == "category:" { category=$2 }
    active && $1 == "environment:" { environment=$2 }
    active && $1 == wanted_dataset ":" {
      wanted=1
      if ($0 ~ /status: supported/) status="supported"
      else if ($0 ~ /status: unsupported/) status="unsupported"
      next
    }
    active && wanted && /^      [a-zA-Z0-9_]+:/ { wanted=0 }
    active && wanted && $1 == "status:" { status=$2 }
    END { if (active && category != "" && environment != "" && status != "") print category, environment, status }
  ' "$registry"
})"
read -r category environment status <<<"$registry_row"
if [[ -z "${category:-}" || -z "${environment:-}" ]]; then
  echo "unknown model/dataset cell: $model/$dataset" >&2; exit 2
fi
if [[ "$status" != "supported" ]]; then
  echo "unsupported model/dataset cell: $model/$dataset" >&2; exit 2
fi
case "$category:$steps" in
  many:1|many:2|many:4|many:8|many:16|many:32|many:1024|few:1|few:2|few:4|few:8|few:16|few:32) ;;
  *) echo "invalid step count $steps for $category category" >&2; exit 2 ;;
esac

runner_args=(--root "$DLB_ROOT" --model "$model" --dataset "$dataset" --steps "$steps" --num-samples "$num_samples" --seed "$seed")
if [[ -n "$device" ]]; then runner_args+=(--device "$device"); fi
exec "${DLB_CONDA:-conda}" run -n "$environment" python -m dlb.runner "${runner_args[@]}"

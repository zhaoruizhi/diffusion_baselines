#!/usr/bin/env bash
# One serial timing group per physical GPU; every invocation uses fresh outputs.
set -euo pipefail
gpu="${1:?usage: run_elf_timing_group.sh GPU GROUP [--plan]}"
group="${2:?group required}"
plan="${3:-}"
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'GPU must be a physical numeric index' >&2; exit 2; }
[[ "$group" == owt || "$group" == seq2seq ]] || { echo 'group must be owt or seq2seq' >&2; exit 2; }
[[ -z "$plan" || "$plan" == --plan ]] || exit 2
root="${DLB_ROOT:-$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
python_bin="${ELF_PYTHON:-python}"

matrix() {
  local task steps
  if [[ "$group" == owt ]]; then
    for task in owt owt-prefix; do
      for steps in 1 2 4 8 16 32 1024; do
        printf '%s validation %s\n' "$task" "$steps"
      done
    done
  else
    for task in wmt14 xsum; do
      for steps in 1 2 4 8 16 32 64; do
        printf '%s validation %s\n' "$task" "$steps"
      done
      for steps in 32 64; do
        printf '%s test %s\n' "$task" "$steps"
      done
    done
  fi
}
if [[ "$plan" == --plan ]]; then
  echo "GPU=$gpu group=$group; batch=1 seed=42 compile=0 warmups=5 repeats=32 prompt=0"
  echo 'OWT split is unconditional; owt-prefix split is c64-text-t5 (suite resolves these).'
  matrix
  exit 0
fi
[[ "$(uname -s)" == Linux ]] || { echo 'Timing must run on the Linux GPU server' >&2; exit 2; }
uuid="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
[[ "$uuid" == GPU-* ]] || { echo 'Cannot resolve GPU UUID' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$uuid" DLB_ROOT="$root" ELF_PYTHON="$python_bin"
export ELF_SEED=42 ELF_COMPILE=0 ELF_TIMING_PROMPT=0 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export ELF_RESULTS="$root/results/elf-timing-gpu${gpu}-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir -p "$ELF_RESULTS/logs" "$ELF_RESULTS/environment"
echo "TIMING OUTPUT: $ELF_RESULTS"
printf 'physical_gpu=%s\nuuid=%s\ngroup=%s\n' "$gpu" "$uuid" "$group" > "$ELF_RESULTS/environment/binding.txt"
git -C "$root" rev-parse HEAD > "$ELF_RESULTS/environment/project-commit.txt"
nvidia-smi -i "$uuid" > "$ELF_RESULTS/environment/gpu-start.txt"

while read -r task split steps; do
  # Check between cells; this cannot reserve the device against other users.
  pids="$(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader,nounits)"
  if [[ -n "${pids//[[:space:]]/}" ]]; then
    echo "GPU $gpu has compute processes ($pids); stop before $task/$split/$steps. Do not kill others." >&2
    exit 2
  fi
  tag="${task}_${split}_s${steps}"
  nvidia-smi -i "$uuid" > "$ELF_RESULTS/environment/${tag}-before.txt"
  echo "TIMING GPU $gpu: $task/$split/$steps"
  env ELF_STEPS="$steps" ELF_SPLIT="$split" \
    bash "$root/scripts/run_elf_suite.sh" timing "$task" 2>&1 | tee "$ELF_RESULTS/logs/$tag.log"
  nvidia-smi -i "$uuid" > "$ELF_RESULTS/environment/${tag}-after.txt"
done < <(matrix)

"$python_bin" "$root/scripts/summarize_elf.py" \
  --results "$ELF_RESULTS" --output "$ELF_RESULTS/summary.csv"
echo "FINISHED group=$group GPU=$gpu; $ELF_RESULTS/summary.csv"

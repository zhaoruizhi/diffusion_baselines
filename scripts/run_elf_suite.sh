#!/usr/bin/env bash
# Independent ELF cells; run one process per GPU. No downloads/training here.
set -euo pipefail
root="${DLB_ROOT:-$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
python_bin="${ELF_PYTHON:-python}"
stage="${1:?usage: run_elf_suite.sh STAGE TASK}"
task="${2:?task required}"
case "$stage" in generate|timing|evaluate|smoke) ;; *) exit 2 ;; esac
case "$task" in owt|owt-prefix|wmt14|xsum) ;; *) exit 2 ;; esac
seed="${ELF_SEED:-42}"
split="${ELF_SPLIT:-validation}"
compiled="${ELF_COMPILE:-0}"
steps_default="1 2 4 8 16 32 1024"
if [[ "$task" == wmt14 || "$task" == xsum ]]; then steps_default="1 2 4 8 16 32 64"; fi
read -r -a steps_list <<< "${ELF_STEPS:-$steps_default}"
input_args=()
count="${ELF_COUNT:-1024}"
if [[ "$task" == wmt14 || "$task" == xsum ]]; then
  input_args=(--input "$root/data/elf/$task-$split.jsonl")
  count="${ELF_COUNT:-0}"
elif [[ "$task" == owt-prefix ]]; then
  input_args=(--input "$root/data/elf/owt-prefix.jsonl")
  split="c64-text-t5"
else
  split="unconditional"
fi
compile_args=()
if [[ "$compiled" == 1 ]]; then compile_args=(--compile); fi
output_base="${ELF_RESULTS:-$root/results/elf}"
for steps in "${steps_list[@]}"; do
  stem="$task/$split/steps_$steps/seed_$seed/compile_$compiled"
  case "$stage" in
    generate|smoke)
      kind=quality
      diversity_args=()
      if [[ "$task" == owt-prefix ]]; then diversity_args=(--diversity); fi
      if [[ "$stage" == smoke ]]; then kind=smoke; count=2; fi
      "$python_bin" "$root/scripts/run_elf.py" generate --root "$root" --task "$task" --steps "$steps" \
        --seed "$seed" --num-samples "$count" --batch-size "${ELF_BATCH_SIZE:-8}" \
        --output "$output_base/$kind/$stem" ${input_args[@]+"${input_args[@]}"} ${compile_args[@]+"${compile_args[@]}"} ${diversity_args[@]+"${diversity_args[@]}"}
      ;;
    timing)
      "$python_bin" "$root/scripts/run_elf.py" timing --root "$root" --task "$task" --steps "$steps" \
        --seed "$seed" --batch-size 1 --timing-prompt-index "${ELF_TIMING_PROMPT:-0}" \
        --output "$output_base/timing/$stem/prompt_${ELF_TIMING_PROMPT:-0}" ${input_args[@]+"${input_args[@]}"} ${compile_args[@]+"${compile_args[@]}"}
      ;;
    evaluate)
      "$python_bin" "$root/scripts/evaluate_elf.py" --root "$root" \
        --run "$output_base/quality/$stem" --batch-size "${ELF_EVAL_BATCH_SIZE:-8}"
      ;;
  esac
done

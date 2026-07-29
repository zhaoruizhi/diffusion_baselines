#!/usr/bin/env bash
# Create/update only the isolated baseline environments. This script never
# activates an environment or changes shell startup files.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${DLB_CONDA:-}" ]]; then
  CONDA_BIN="${DLB_CONDA}"
elif command -v mamba >/dev/null 2>&1; then
  CONDA_BIN="$(command -v mamba)"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
else
  echo "Neither mamba nor conda is available; set DLB_CONDA to an explicit executable." >&2
  exit 127
fi

ALL_ENVIRONMENTS=(
  dlb-flm dlb-langflow dlb-duo dlb-mdlm dlb-candi dlb-rdlm dlb-sdtt dlb-di4c dlb-eval
)
if [[ -n "${DLB_ENV_NAMES:-}" ]]; then
  IFS=',' read -r -a ENVIRONMENTS <<<"${DLB_ENV_NAMES}"
else
  ENVIRONMENTS=("${ALL_ENVIRONMENTS[@]}")
fi

environment_file() {
  local environment="$1"
  local filename="${environment#dlb-}"
  local path="${REPO_ROOT}/envs/${filename}.yml"
  [[ " ${ALL_ENVIRONMENTS[*]} " == *" ${environment} "* && -f "${path}" ]] || return 1
  printf '%s\n' "${path}"
}

environment_exists() {
  local environment="$1"
  local listing
  listing="$("${CONDA_BIN}" env list --json)" || return 1
  python3 -c '
import json
import pathlib
import sys
target = sys.argv[1]
paths = json.load(sys.stdin).get("envs", [])
sys.exit(0 if any(pathlib.Path(path).name == target for path in paths) else 1)
' "${environment}" <<<"${listing}"
}

flash_attention_version() {
  case "$1" in
    dlb-flm) printf '%s\n' '2.8.3' ;;
    dlb-duo|dlb-sdtt|dlb-di4c) printf '%s\n' '2.7.4.post1' ;;
    dlb-mdlm) printf '%s\n' '2.5.6' ;;
    dlb-candi) printf '%s\n' '2.6.1' ;;
    *) return 1 ;;
  esac
}

failures=()
for environment in "${ENVIRONMENTS[@]}"; do
  if ! yaml_path="$(environment_file "${environment}")"; then
    echo "Unknown environment: ${environment}" >&2
    failures+=("${environment}")
    continue
  fi

  if environment_exists "${environment}"; then
    action=(env update --file "${yaml_path}" --prune=false)
  else
    action=(env create --file "${yaml_path}")
  fi

  if ! "${CONDA_BIN}" "${action[@]}"; then
    echo "FAILED ${environment}: conda environment ${action[1]} failed." >&2
    failures+=("${environment}")
    continue
  fi

  if flash_version="$(flash_attention_version "${environment}")"; then
    if ! "${CONDA_BIN}" run -n "${environment}" python -m pip install \
      "flash-attn==${flash_version}" --no-build-isolation; then
      echo "FAILED ${environment}: FlashAttention ${flash_version} installation failed." >&2
      failures+=("${environment}")
      continue
    fi
  fi
done

if ((${#failures[@]})); then
  echo "Environment creation/update failed: ${failures[*]}" >&2
  exit 1
fi


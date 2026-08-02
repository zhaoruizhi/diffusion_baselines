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
  local info
  listing="$("${CONDA_BIN}" env list --json)" || return 2
  info="$("${CONDA_BIN}" info --json)" || return 2
  python3 -c '
import json
import os
import sys
target = sys.argv[1]
try:
    environment_document = json.loads(sys.argv[2])
    info_document = json.loads(sys.argv[3])
except (json.JSONDecodeError, TypeError):
    sys.exit(2)
if not isinstance(environment_document, dict) or not isinstance(info_document, dict):
    sys.exit(2)
environments = environment_document.get("envs")
directories = info_document.get("envs_dirs")
if not isinstance(environments, list) or not isinstance(directories, list):
    sys.exit(2)
if not all(isinstance(path, str) for path in environments + directories):
    sys.exit(2)
known = {os.path.normpath(path) for path in environments}
expected = {os.path.normpath(os.path.join(directory, target)) for directory in directories}
if known & expected:
    sys.exit(0)
if any(os.path.basename(path) == target for path in known):
    sys.exit(2)
sys.exit(1)
' "${environment}" "${listing}" "${info}"
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

post_torch_pip_requirements() {
  case "$1" in
    dlb-mdlm)
      printf '%s\n' \
        'git+https://github.com/Dao-AILab/causal-conv1d.git@v1.1.3.post1' \
        'git+https://github.com/state-spaces/mamba.git@v1.1.4'
      ;;
  esac
}

check_pytorch_import() {
  local environment="$1"
  "${CONDA_BIN}" run -n "${environment}" python -c \
    'import torch; print(torch.__version__, torch.version.cuda)'
}

install_post_torch_pip_requirements() {
  local environment="$1"
  local requirement
  local requirements=()

  while IFS= read -r requirement; do
    [[ -n "${requirement}" ]] && requirements+=("${requirement}")
  done < <(post_torch_pip_requirements "${environment}")

  if ((${#requirements[@]} == 0)); then
    return 0
  fi

  "${CONDA_BIN}" run -n "${environment}" python -m pip install \
    --no-build-isolation "${requirements[@]}"
}

failures=()
for environment in "${ENVIRONMENTS[@]}"; do
  if ! yaml_path="$(environment_file "${environment}")"; then
    echo "Unknown environment: ${environment}" >&2
    failures+=("${environment}")
    continue
  fi

  if environment_exists "${environment}"; then
    action=(env update --file "${yaml_path}")
  else
    discovery_status=$?
    if ((discovery_status == 1)); then
      action=(env create --file "${yaml_path}")
    else
      echo "FAILED ${environment}: environment discovery was unreliable." >&2
      failures+=("${environment}")
      continue
    fi
  fi

  if ! "${CONDA_BIN}" "${action[@]}"; then
    echo "FAILED ${environment}: conda environment ${action[1]} failed." >&2
    failures+=("${environment}")
    continue
  fi

  if ! check_pytorch_import "${environment}"; then
    echo "FAILED ${environment}: PyTorch import failed before post-install steps." >&2
    failures+=("${environment}")
    continue
  fi

  if ! install_post_torch_pip_requirements "${environment}"; then
    echo "FAILED ${environment}: post-torch pip requirements installation failed." >&2
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

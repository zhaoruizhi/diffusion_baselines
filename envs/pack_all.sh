#!/usr/bin/env bash
# Non-destructive environment archives for transfer to compatible Linux hosts.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARTIFACT_DIR="${DLB_ARTIFACT_DIR:-${REPO_ROOT}/artifacts/conda-packs}"

if [[ -n "${DLB_CONDA_PACK:-}" ]]; then
  CONDA_PACK_BIN="${DLB_CONDA_PACK}"
elif command -v conda-pack >/dev/null 2>&1; then
  CONDA_PACK_BIN="$(command -v conda-pack)"
else
  echo "conda-pack is required; install it in the controlling environment or set DLB_CONDA_PACK." >&2
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

mkdir -p "${ARTIFACT_DIR}"
failures=()
for environment in "${ENVIRONMENTS[@]}"; do
  if [[ " ${ALL_ENVIRONMENTS[*]} " != *" ${environment} "* ]]; then
    echo "Unknown environment: ${environment}" >&2
    failures+=("${environment}")
    continue
  fi

  destination="${ARTIFACT_DIR}/${environment}.tar.gz"
  if [[ -e "${destination}" && ! -f "${destination}" ]]; then
    echo "FAILED ${environment}: archive destination is not a regular file." >&2
    failures+=("${environment}")
    continue
  fi

  temporary_directory="$(mktemp -d "${ARTIFACT_DIR}/.${environment}.XXXXXX")" || {
    echo "FAILED ${environment}: could not create an atomic staging directory." >&2
    failures+=("${environment}")
    continue
  }
  temporary_archive="${temporary_directory}/${environment}.tar.gz"

  if "${CONDA_PACK_BIN}" -n "${environment}" -o "${temporary_archive}"; then
    if ! mv -f "${temporary_archive}" "${destination}"; then
      echo "FAILED ${environment}: could not atomically publish the archive." >&2
      failures+=("${environment}")
    elif ! rmdir "${temporary_directory}"; then
      echo "FAILED ${environment}: archive published but staging cleanup failed." >&2
      failures+=("${environment}")
    fi
  else
    echo "FAILED ${environment}: conda-pack failed; staging directory retained at ${temporary_directory}." >&2
    failures+=("${environment}")
  fi
done

if ((${#failures[@]})); then
  echo "Environment packing failed: ${failures[*]}" >&2
  exit 1
fi

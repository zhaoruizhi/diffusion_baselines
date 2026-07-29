#!/usr/bin/env bash
# Server-side runtime checks. One JSON object is written to stdout per
# environment; stderr carries the human-readable aggregate summary.
set -uo pipefail

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

method_imports() {
  case "$1" in
    dlb-flm) printf '%s\n' 'einops lightning hydra rich flash_attn' ;;
    dlb-langflow) printf '%s\n' 'einops safetensors transformers' ;;
    dlb-duo) printf '%s\n' 'lightning hydra rich triton torchvision flash_attn' ;;
    dlb-mdlm) printf '%s\n' 'lightning mamba_ssm hydra rich flash_attn' ;;
    dlb-candi) printf '%s\n' 'lightning evaluate hydra rich flash_attn' ;;
    dlb-rdlm) printf '%s\n' 'accelerate hydra datasets' ;;
    dlb-sdtt|dlb-di4c) printf '%s\n' 'lightning torchdata einops flash_attn' ;;
    dlb-eval) printf '%s\n' 'datasets evaluate transformers' ;;
    *) return 1 ;;
  esac
}

failures=()
for environment in "${ENVIRONMENTS[@]}"; do
  if ! imports="$(method_imports "${environment}")"; then
    printf '{"environment":"%s","error":"unknown environment"}\n' "${environment}"
    failures+=("${environment}")
    continue
  fi

  record=""
  if record="$("${CONDA_BIN}" run -n "${environment}" python - "${environment}" ${imports} <<'PY'
import importlib
import json
import platform
import sys

environment = sys.argv[1]
modules = sys.argv[2:]
record = {
    "environment": environment,
    "python": platform.python_version(),
    "torch": None,
    "torch_cuda": None,
    "cuda_available": False,
    "imports": {},
}
failed = False
try:
    import torch

    record["torch"] = torch.__version__
    record["torch_cuda"] = torch.version.cuda
    record["cuda_available"] = torch.cuda.is_available()
    if not record["cuda_available"]:
        record["cuda_error"] = "CUDA is unavailable"
        failed = True
except Exception as error:
    record["torch_error"] = str(error)
    failed = True

for module in modules:
    try:
        importlib.import_module(module)
        record["imports"][module] = True
    except Exception as error:
        record["imports"][module] = False
        record.setdefault("import_errors", {})[module] = str(error)
        failed = True

print(json.dumps(record, sort_keys=True))
sys.exit(1 if failed else 0)
PY
)"; then
    printf '%s\n' "${record}"
  else
    if [[ -n "${record}" ]]; then
      printf '%s\n' "${record}"
    else
      printf '{"environment":"%s","python":null,"torch":null,"torch_cuda":null,"cuda_available":false,"imports":{},"error":"verification failed"}\n' "${environment}"
    fi
    failures+=("${environment}")
  fi
done

if ((${#failures[@]})); then
  echo "Environment verification failed: ${failures[*]}" >&2
  exit 1
fi

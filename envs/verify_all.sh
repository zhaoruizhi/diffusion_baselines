#!/usr/bin/env bash
# Server-side runtime checks. Stdout contains exactly one validated JSON object
# per requested environment; manager output is never forwarded directly.
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
    dlb-flm) printf '%s\n' 'datasets einops flash_attn fsspec hydra lightning omegaconf rich scipy timm tokenizers torchmetrics tqdm transformers triton wandb' ;;
    dlb-langflow) printf '%s\n' 'einops huggingface_hub safetensors transformers' ;;
    dlb-duo) printf '%s\n' 'datasets einops flash_attn fsspec h5py hydra lightning omegaconf rich timm tokenizers torchmetrics torchvision tqdm transformers triton wandb' ;;
    dlb-mdlm) printf '%s\n' 'causal_conv1d datasets einops flash_attn fsspec hydra lightning mamba_ssm omegaconf rich timm transformers wandb' ;;
    dlb-candi) printf '%s\n' 'datasets einops evaluate flash_attn fsspec hydra lightning omegaconf rich scipy tokenizers torchmetrics tqdm transformers' ;;
    dlb-rdlm) printf '%s\n' 'accelerate datasets einops fsspec hydra numpy omegaconf scipy tokenizers tqdm transformers wandb' ;;
    dlb-sdtt|dlb-di4c) printf '%s\n' 'datasets einops flash_attn fsspec huggingface_hub hydra lightning loguru mauve omegaconf pandas tensorboard timm tokenizers torchdata tqdm transformers wandb' ;;
    dlb-eval) printf '%s\n' 'accelerate datasets evaluate fsspec mauve sacrebleu scipy tokenizers transformers' ;;
    *) return 1 ;;
  esac
}

emit_failure() {
  python3 - "$1" "$2" <<'PY'
import json
import sys

print(json.dumps({
    "environment": sys.argv[1],
    "python": None,
    "torch": None,
    "torch_cuda": None,
    "cuda_available": False,
    "imports": {},
    "error": sys.argv[2],
}, sort_keys=True))
PY
}

validate_probe() {
  python3 - "$1" "$2" "$3" <<'PY'
import json
import sys

marker = "DLB_ENV_PROBE_V1:"
environment, modules, raw = sys.argv[1:]
expected_modules = modules.split()
lines = raw.splitlines()
if len(lines) != 1 or not lines[0].startswith(marker):
    raise SystemExit(2)
try:
    record = json.loads(lines[0][len(marker):])
except json.JSONDecodeError:
    raise SystemExit(2)
if not isinstance(record, dict) or record.get("environment") != environment:
    raise SystemExit(2)
required = {"environment", "python", "torch", "torch_cuda", "cuda_available", "imports"}
if not required <= record.keys():
    raise SystemExit(2)
if not isinstance(record["python"], str):
    raise SystemExit(2)
if not (isinstance(record["torch"], str) or record["torch"] is None):
    raise SystemExit(2)
if not (isinstance(record["torch_cuda"], str) or record["torch_cuda"] is None):
    raise SystemExit(2)
if not isinstance(record["cuda_available"], bool) or not isinstance(record["imports"], dict):
    raise SystemExit(2)
if set(record["imports"]) != set(expected_modules):
    raise SystemExit(2)
if not all(isinstance(value, bool) for value in record["imports"].values()):
    raise SystemExit(2)
print(json.dumps(record, sort_keys=True))
healthy = record["cuda_available"] and all(record["imports"].values()) and record["torch"] is not None
raise SystemExit(0 if healthy else 1)
PY
}

failures=()
for environment in "${ENVIRONMENTS[@]}"; do
  if ! imports="$(method_imports "${environment}")"; then
    emit_failure "${environment}" "unknown environment"
    failures+=("${environment}")
    continue
  fi

  probe_output=""
  manager_status=0
  probe_output="$("${CONDA_BIN}" run -n "${environment}" python - "${environment}" ${imports} <<'PY' 2>&1
import importlib
import json
import platform
import sys

marker = "DLB_ENV_PROBE_V1:"
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

print(marker + json.dumps(record, sort_keys=True))
raise SystemExit(1 if failed else 0)
PY
)" || manager_status=$?

  probe_status=0
  record="$(validate_probe "${environment}" "${imports}" "${probe_output}")" || probe_status=$?
  if ((probe_status <= 1)); then
    printf '%s\n' "${record}"
    if ((manager_status != 0 || probe_status != 0)); then
      failures+=("${environment}")
    fi
  else
    emit_failure "${environment}" "verification probe failed"
    failures+=("${environment}")
  fi
done

if ((${#failures[@]})); then
  echo "Environment verification failed: ${failures[*]}" >&2
  exit 1
fi

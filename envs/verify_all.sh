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
    dlb-flm) printf '%s\n' 'datasets einops entmax flash_attn fsspec huggingface_hub hydra lightning numpy omegaconf requests rich scipy timm tokenizers torchmetrics tqdm transformers triton wandb' ;;
    dlb-langflow) printf '%s\n' 'einops huggingface_hub safetensors transformers' ;;
    dlb-duo) printf '%s\n' 'datasets einops flash_attn fsspec h5py huggingface_hub hydra lightning numpy omegaconf requests rich scipy timm tokenizers torchmetrics torchvision tqdm transformers triton wandb' ;;
    dlb-mdlm) printf '%s\n' 'causal_conv1d datasets einops flash_attn fsspec huggingface_hub hydra lightning mamba_ssm numpy omegaconf requests rich timm tokenizers torchmetrics transformers wandb' ;;
    dlb-candi) printf '%s\n' 'datasets einops evaluate flash_attn fsspec huggingface_hub hydra lightning numpy omegaconf requests rich scipy tokenizers torchmetrics tqdm transformers' ;;
    dlb-rdlm) printf '%s\n' 'accelerate datasets einops fsspec huggingface_hub hydra numpy omegaconf requests scipy tokenizers tqdm transformers wandb' ;;
    dlb-sdtt|dlb-di4c) printf '%s\n' 'datasets einops flash_attn fsspec huggingface_hub hydra lightning loguru mauve numpy omegaconf pandas requests safetensors tensorboard timm tokenizers torchdata tqdm transformers wandb' ;;
    dlb-eval) printf '%s\n' 'accelerate datasets evaluate fsspec mauve sacrebleu scipy tokenizers transformers' ;;
    *) return 1 ;;
  esac
}

emit_failure() {
  python3 - "$1" "$2" "${3:-}" <<'PY'
import json
import sys

record = {
    "environment": sys.argv[1],
    "python": None,
    "torch": None,
    "torch_cuda": None,
    "cuda_available": False,
    "imports": {},
    "error": sys.argv[2],
}
if len(sys.argv) > 3 and sys.argv[3]:
    record["diagnostic"] = sys.argv[3][-8000:]
print(json.dumps(record, sort_keys=True, allow_nan=False))
PY
}

probe_diagnostic() {
  local reason="$1"
  local raw_output="$2"
  local stderr_path="$3"
  local stderr_output=""

  [[ "${DLB_VERIFY_DEBUG:-}" == "1" ]] || return 0

  if [[ -s "${stderr_path}" ]]; then
    stderr_output="$(<"${stderr_path}")"
  fi

  printf '%s\n' "${reason}"
  if [[ -n "${stderr_output}" ]]; then
    printf 'stderr:\n%s\n' "${stderr_output}"
  fi
  if [[ -n "${raw_output}" ]]; then
    printf 'stdout:\n%s\n' "${raw_output}"
  fi
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


def reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


try:
    record = json.loads(
        lines[0][len(marker):],
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )
except (json.JSONDecodeError, ValueError):
    raise SystemExit(2)
if not isinstance(record, dict) or record.get("environment") != environment:
    raise SystemExit(2)
required = {"environment", "python", "torch", "torch_cuda", "cuda_available", "imports"}
optional = {"torch_error", "cuda_error", "import_errors"}
if not required <= record.keys() or not set(record) <= required | optional:
    raise SystemExit(2)
if type(record["python"]) is not str:
    raise SystemExit(2)
if not (type(record["torch"]) is str or record["torch"] is None):
    raise SystemExit(2)
if not (type(record["torch_cuda"]) is str or record["torch_cuda"] is None):
    raise SystemExit(2)
if type(record["cuda_available"]) is not bool or type(record["imports"]) is not dict:
    raise SystemExit(2)
if set(record["imports"]) != set(expected_modules):
    raise SystemExit(2)
if not all(type(value) is bool for value in record["imports"].values()):
    raise SystemExit(2)
if "torch_error" in record and (
    type(record["torch_error"]) is not str or record["torch"] is not None
):
    raise SystemExit(2)
if "cuda_error" in record and (
    type(record["cuda_error"]) is not str or record["cuda_available"]
):
    raise SystemExit(2)
if "import_errors" in record:
    errors = record["import_errors"]
    failed_modules = {name for name, available in record["imports"].items() if not available}
    if (
        type(errors) is not dict
        or set(errors) != failed_modules
        or not all(type(value) is str for value in errors.values())
    ):
        raise SystemExit(2)
elif not all(record["imports"].values()):
    raise SystemExit(2)
print(json.dumps(record, sort_keys=True, allow_nan=False))
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

  if ! probe_stderr="$(mktemp "${TMPDIR:-/tmp}/dlb-env-probe.XXXXXX")"; then
    emit_failure "${environment}" "verification probe failed"
    failures+=("${environment}")
    continue
  fi
  if ! probe_script="$(mktemp "${TMPDIR:-/tmp}/dlb-env-probe.XXXXXX")"; then
    emit_failure "${environment}" "verification probe failed"
    rm -f -- "${probe_stderr}"
    failures+=("${environment}")
    continue
  fi

  if ! cat >"${probe_script}" <<'PY'
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
try:
    import torch

    record["torch"] = torch.__version__
    record["torch_cuda"] = torch.version.cuda
    record["cuda_available"] = torch.cuda.is_available()
    if not record["cuda_available"]:
        record["cuda_error"] = "CUDA is unavailable"
except Exception as error:
    record["torch_error"] = str(error)

for module in modules:
    try:
        importlib.import_module(module)
        record["imports"][module] = True
    except Exception as error:
        record["imports"][module] = False
        record.setdefault("import_errors", {})[module] = str(error)

print(marker + json.dumps(record, sort_keys=True, allow_nan=False))
raise SystemExit(0)
PY
  then
    emit_failure "${environment}" "verification probe failed"
    rm -f -- "${probe_stderr}" "${probe_script}"
    failures+=("${environment}")
    continue
  fi

  probe_output=""
  manager_status=0
  probe_output="$("${CONDA_BIN}" run -n "${environment}" python \
    "${probe_script}" "${environment}" ${imports} 2>"${probe_stderr}")" || manager_status=$?
  diagnostic=""
  if ((manager_status != 0)); then
    diagnostic="$(
      probe_diagnostic "conda run exited with status ${manager_status}" \
        "${probe_output}" "${probe_stderr}"
    )"
  fi

  if ((manager_status != 0)); then
    emit_failure "${environment}" "verification probe failed" "${diagnostic}"
    rm -f -- "${probe_stderr}" "${probe_script}"
    failures+=("${environment}")
    continue
  fi

  probe_status=0
  record="$(validate_probe "${environment}" "${imports}" "${probe_output}")" || probe_status=$?
  if ((probe_status <= 1)); then
    printf '%s\n' "${record}"
    if ((probe_status != 0)); then
      failures+=("${environment}")
    fi
  else
    diagnostic="$(
      probe_diagnostic "probe output was not valid DLB_ENV_PROBE_V1 JSON" \
        "${probe_output}" "${probe_stderr}"
    )"
    emit_failure "${environment}" "verification probe failed" "${diagnostic}"
    failures+=("${environment}")
  fi
  rm -f -- "${probe_stderr}" "${probe_script}"
done

if ((${#failures[@]})); then
  echo "Environment verification failed: ${failures[*]}" >&2
  exit 1
fi

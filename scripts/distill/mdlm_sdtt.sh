#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DLB_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
export PYTHONPATH="$DLB_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${DLB_PYTHON:-python}"
exec "$PYTHON_BIN" -m dlb.recipes --root "$DLB_ROOT" --recipe mdlm_sdtt "$@"

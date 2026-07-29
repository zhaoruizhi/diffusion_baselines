#!/usr/bin/env bash
set -euo pipefail

DLB_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export DLB_ROOT
DLB_PYTHON=${PYTHON_BIN:-"$DLB_ROOT/.venv/bin/python"}

if [[ ! -x "$DLB_PYTHON" ]]; then
    echo "ERROR Python interpreter not found: $DLB_PYTHON" >&2
    exit 1
fi

source_rows=$(PYTHONPATH="$DLB_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$DLB_PYTHON" - "$DLB_ROOT" <<'PY'
from pathlib import Path
import sys

import yaml

from dlb.registry import load_registry

root = Path(sys.argv[1])
with (root / "artifacts/sources.yaml").open(encoding="utf-8") as manifest_file:
    manifest = yaml.safe_load(manifest_file)
required_sources = {entry.source for entry in load_registry(root / "configs/experiments.yaml").models.values()}
if not isinstance(manifest, dict) or set(manifest) != required_sources:
    raise SystemExit("source manifest IDs must match registry sources")
for name in sorted(required_sources):
    source = manifest[name]
    if not isinstance(source, dict) or set(source) != {"url", "commit"}:
        raise SystemExit(f"invalid manifest entry for {name}")
    print(f"{name}\t{source['url']}\t{source['commit']}")
PY
)

while IFS=$'\t' read -r name url commit; do
    source_dir="$DLB_ROOT/upstreams/$name"
    if [[ ! -e "$source_dir" ]]; then
        git clone --filter=blob:none "$url" "$source_dir"
    elif [[ ! -d "$source_dir/.git" ]]; then
        echo "ERROR $name is not a Git repository: $source_dir" >&2
        exit 1
    fi

    if [[ "$(git -C "$source_dir" remote get-url origin)" != "$url" ]]; then
        echo "ERROR $name origin does not match manifest" >&2
        exit 1
    fi
    if [[ -n "$(git -C "$source_dir" status --porcelain)" ]]; then
        echo "ERROR $name has uncommitted changes" >&2
        exit 1
    fi

    git -C "$source_dir" fetch --depth=1 origin "$commit"
    git -C "$source_dir" checkout --detach "$commit"
    test "$(git -C "$source_dir" rev-parse HEAD)" = "$commit"
    test -z "$(git -C "$source_dir" status --porcelain)"
done <<< "$source_rows"

PYTHONPATH="$DLB_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$DLB_PYTHON" "$DLB_ROOT/scripts/verify_sources.py" --root "$DLB_ROOT"

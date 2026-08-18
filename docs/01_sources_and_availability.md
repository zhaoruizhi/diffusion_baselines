# 01. 源码与可用性

## 服务器端下载

源码仓库由 `artifacts/sources.yaml` 固定 URL 和 40 位 commit。服务器执行：

```bash
export DLB_ROOT="$PWD"
conda activate dlb-bootstrap
export DLB_PYTHON="$CONDA_PREFIX/bin/python"
bash scripts/fetch_sources.sh
"$DLB_PYTHON" scripts/verify_sources.py --root "$DLB_ROOT"
```

`fetch_sources.sh` 优先使用 `DLB_PYTHON`，否则使用 `PYTHON_BIN`、当前 Conda 环境的 `python`、PATH 中的 `python` 或 `python3`。脚本会拒绝 origin 不一致、脏工作树和未处于锁定 commit 的 checkout。`upstreams/` 是只读依赖；必要兼容修复只能放入 `patches/<source>/`，不能直接改上游目录。

## 公开可用性

注册表 `configs/experiments.yaml` 是唯一的 coverage 来源：

- `supported` cell 才会进入 generation matrix。
- `langflow/lm1b` 和 `langflow/owt` 使用官方 Hugging Face checkpoint，并进入 many-step generation matrix。
- `rdlm/owt` unsupported：官方 RDLM release 没有 OWT model。

unsupported 不是失败，也不能使用另一个数据集或另一个 teacher 替代。矩阵生成后再次确认：

```bash
python -m dlb.matrix --root "$DLB_ROOT" \
  --output "$DLB_ROOT/results/matrix/generation.tsv"
cat "$DLB_ROOT/results/matrix/unsupported.tsv"
```

## 不在本地做的事

本地 worktree 不运行 `fetch_sources.sh`、`fetch_data.py`、`fetch_checkpoints.py` 的真实下载，不创建 GPU 环境，不训练，不生成 1,024 样本。提交 GitHub 后，在服务器 clone 同一 commit 执行上述命令。

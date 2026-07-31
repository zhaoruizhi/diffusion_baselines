# 03. Conda 环境

## 环境列表

| 环境 | 方法 |
|---|---|
| `dlb-flm` | FLM、FMLM |
| `dlb-langflow` | LangFlow |
| `dlb-duo` | Duo、Duo+DCD |
| `dlb-mdlm` | MDLM |
| `dlb-candi` | CANDI |
| `dlb-rdlm` | RDLM |
| `dlb-sdtt` | MDLM+SDTT |
| `dlb-di4c` | Duo/MDLM+Di4C |
| `dlb-eval` | GPT-2 Large、统一评测 |

YAML 文件在 `envs/`。版本差异和上游要求以 YAML 与 `patches/` 为准，不要把当前机器已有的任意环境当作实验环境。

## 创建和验证

```bash
export DLB_CONDA="${DLB_CONDA:-conda}"
bash envs/create_all.sh
bash envs/verify_all.sh
```

只创建部分环境时：

```bash
DLB_ENV_NAMES=dlb-flm,dlb-eval bash envs/create_all.sh
DLB_ENV_NAMES=dlb-flm,dlb-eval bash envs/verify_all.sh
```

验证脚本会检查 Python、PyTorch、CUDA 和方法依赖导入。出现 CUDA/import failure 时，先修复环境，不要在 `run_all.sh` 中降低精度或换模型。

## 可搬运 archive

服务器资源允许时，使用：

```bash
bash envs/pack_all.sh
```

输出在 `artifacts/conda-packs/`，不会提交 Git。archive 在另一台服务器解包后仍需运行 `envs/verify_all.sh`。

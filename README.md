# Diffusion Language Baselines

这是一个面向 Linux GPU 服务器的可复现实验框架，用于比较连续、离散、混合扩散语言模型及其少步蒸馏版本。当前 Mac/worktree 只保存代码、锁文件、脚本和文档；数据、源码仓库快照、checkpoint、Conda 环境、训练和推理都必须在服务器执行。

## 实验范围

- 多步生成：FLM、LangFlow、Duo、MDLM、CANDI、RDLM。
- 少步/单步蒸馏：FMLM、Duo+DCD、Duo+Di4C、MDLM+SDTT、MDLM+Di4C。
- LM1B 使用 `bert-base-uncased`、长度 128；OWT 使用 `gpt2`、长度 1024。
- 多步步数：`1 2 4 8 16 32 1024`；少步步数：`1 2 4 8 16 32`。
- 每个受支持配置生成 1,024 个样本，seed 固定为 42。
- 质量指标：GPT-2 Large Generative PPL、平均 unigram entropy、Self-BLEU；主延迟指标：独立 CUDA timing 的 seconds/sample。

130 个受支持 generation cell 会写入 `results/matrix/generation.tsv`。LangFlow/LM1B 和 RDLM/OWT 不会伪造任务，原因写入 `results/matrix/unsupported.tsv`。

## 服务器最短路径

在 Linux GPU 服务器上执行：

```bash
git clone <你的 GitHub 仓库地址> diffusion_baseline
cd diffusion_baseline
export DLB_ROOT="$PWD"
export DLB_PYTHON="${DLB_PYTHON:-python}"
```

先准备一个能导入 `PyYAML`、`pydantic` 的 bootstrap Python；它只用于解析 manifest 和启动下载脚本：

```bash
python -m pip install -e ".[dev,data,checkpoints]"
```

然后严格按以下顺序执行：

```bash
bash scripts/fetch_sources.sh
python scripts/verify_sources.py --root "$DLB_ROOT"
python scripts/fetch_data.py --root "$DLB_ROOT"
python scripts/preprocess_data.py --root "$DLB_ROOT" --dataset all
python scripts/verify_data.py --root "$DLB_ROOT" --dataset all
python scripts/fetch_checkpoints.py --root "$DLB_ROOT" --all-public
python scripts/verify_checkpoints.py --root "$DLB_ROOT"
bash envs/create_all.sh
bash envs/verify_all.sh
bash scripts/smoke_all.sh
bash scripts/run_all.sh
bash scripts/evaluate_all.sh
bash scripts/benchmark_all.sh
python scripts/aggregate_results.py --root "$DLB_ROOT"
```

`smoke_all.sh` 使用独立的 `results/smoke/`，不会被正式汇总接受。若只需先验证一个方法，可运行：

```bash
bash scripts/run_one.sh --model flm --dataset lm1b --steps 1 \
  --num-samples 1 --seed 42
```

严格汇总失败时，不要删除失败目录；先查看 `results/summary/failures.csv`，修复后重跑对应阶段。需要先查看当前可用结果时：

```bash
python scripts/aggregate_results.py --root "$DLB_ROOT" --partial
```

`--partial` 只用于诊断，不是论文主结果。

## 文档索引

- [源码与可用性](docs/01_sources_and_availability.md)
- [数据下载与预处理](docs/02_datasets.md)
- [Conda 环境](docs/03_environments.md)
- [Checkpoint 下载与校验](docs/04_checkpoints.md)
- [训练与蒸馏](docs/05_training.md)
- [生成、评测与 timing](docs/06_generation_and_evaluation.md)
- [Slurm](docs/07_slurm.md)
- [故障排查](docs/08_troubleshooting.md)
- [实验矩阵](docs/EXPERIMENT_MATRIX.md)

所有可复现输入均在 `artifacts/` 和 `configs/` 中锁定。不要把 `data/`、`checkpoints/`、`upstreams/`、`results/` 或 Conda archive 提交到 Git。

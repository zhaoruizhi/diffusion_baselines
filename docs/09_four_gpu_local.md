# 09. 4-GPU 本地并行运行

本路径适用于单机服务器直接用 Bash/Conda 跑实验，不依赖 Slurm。默认只使用
GPU `0,1,2,3`，并且最多同时运行 4 个 matrix task。

## 默认资源策略

- 生成、smoke 和 benchmark：每个 task 绑定 1 张 GPU。
- 评测：每个 task 绑定 1 张 GPU，GPT-2 Large PPL 在该 GPU 上计算。
- 绑定方式：对子进程设置 `CUDA_VISIBLE_DEVICES=<gpu_id>`。子进程内看到的设备通常是 `cuda:0`。
- 默认并发：`DLB_MAX_JOBS=4`。

可以覆盖 GPU 列表或并发数：

```bash
DLB_GPUS=0,1,2,3 DLB_MAX_JOBS=4 bash scripts/run_4gpu.sh
DLB_GPUS=0,1 DLB_MAX_JOBS=2 bash scripts/run_4gpu.sh
```

## 推荐顺序

先完成 README 中的源码、数据、checkpoint 和 Conda 环境准备。需要
`reference_reproduction` 的 checkpoint 时，先按 `docs/05_training.md` 训练或蒸馏
对应产物。

然后执行：

```bash
export DLB_GPUS=0,1,2,3
export DLB_MAX_JOBS=4

bash scripts/smoke_4gpu.sh
bash scripts/run_4gpu.sh
bash scripts/evaluate_4gpu.sh
bash scripts/benchmark_4gpu.sh
"$DLB_PYTHON" scripts/aggregate_results.py --root "$DLB_ROOT"
```

## 输出

- smoke 样本：`results/smoke/samples/...`
- 正式样本：`results/samples/<dataset>/<model>/steps_<steps>/samples.jsonl`
- 指标：`results/metrics/<dataset>/<model>/steps_<steps>/metrics.json`
- timing：`results/timing/<dataset>/<model>/steps_<steps>/timing.json`
- 汇总表：`results/summary/results_wide.csv` 和 `results/summary/results_long.csv`

每个 task 的 stdout/stderr 会写入：

```text
results/logs/4gpu-<stage>-<task_id>.out
results/logs/4gpu-<stage>-<task_id>.err
```

失败清单按阶段写入：

```text
results/logs/smoke_4gpu_failures.tsv
results/logs/generation_4gpu_failures.tsv
results/logs/evaluation_4gpu_failures.tsv
results/logs/benchmark_4gpu_failures.tsv
```

如果失败清单不存在，表示该阶段没有失败 task。

## Dry Run

在真正占用 GPU 前可以检查调度计划：

```bash
bash scripts/run_4gpu.sh --dry-run
bash scripts/evaluate_4gpu.sh --dry-run
bash scripts/benchmark_4gpu.sh --dry-run
```

dry-run 会输出每个 task 绑定的 GPU 和实际 argv，不启动模型。

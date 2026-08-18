# 07. Slurm

Slurm 模板不硬编码 partition、GPU 型号、walltime 或内存。请根据服务器站点在提交命令上补充这些资源参数；模板通过注册表 TSV 的 `environment` 字段选择 Conda 环境。

## 生成 array

先在登录节点创建矩阵，再提交 137 个 array task：

```bash
python -m dlb.matrix --root "$DLB_ROOT" \
  --output "$DLB_ROOT/results/matrix/generation.tsv"
sbatch --array=0-136%8 slurm/generate_array.sbatch
```

## 质量评测和 timing

```bash
sbatch slurm/evaluate.sbatch
sbatch --array=0-136%8 slurm/benchmark_array.sbatch
```

`evaluate.sbatch` 使用 `DLB_EVAL_ENV`（默认 `dlb-eval`）；array 模板读取每一行的模型、数据集、steps 和 registry environment，不根据 task id 猜环境。

## 训练/蒸馏

通过环境变量指定 recipe：

```bash
DLB_RECIPE=candi DLB_DATASET=owt \
  sbatch --gres=gpu:1 slurm/train.sbatch
DLB_RECIPE=di4c DLB_MODEL=duo_di4c DLB_DATASET=lm1b \
  sbatch --gres=gpu:1 slurm/train.sbatch
```

如站点需要 partition、constraint、时间或内存，请加到 `sbatch` 命令，而不是修改模板中的固定值。大训练任务的 devices/nodes 通过 `DLB_TRAIN_DEVICES`、`DLB_TRAIN_NODES` 或 recipe 命令参数设置，必须保持锁定 global batch。

## 日志与重试

Slurm 输出在 `results/logs/slurm-*.out`/`.err`；单个 array task 失败可只重提对应 index。不要删除成功样本或 timing 后盲目重跑，先核对 `run_metadata.json`、`checkpoint_lock.json` 和失败日志。

# 05. 训练与蒸馏

训练只在服务器运行，且必须先完成源码、数据、环境和 teacher checkpoint 校验。所有 recipe 会验证 source commit、processed dataset manifest、全局 batch、学习率、teacher family 和输出 checkpoint。

## Teacher / 多步模型

直接使用已下载官方 checkpoint 的方法无需重新训练。需要 reference reproduction 或 project-owned checkpoint 时，用对应 wrapper：

```bash
bash scripts/train/flm.sh --dataset lm1b --source upstreams/flm \
  --output checkpoints/self_trained/lm1b/flm
bash scripts/train/duo.sh --dataset lm1b --source upstreams/duo \
  --output checkpoints/self_trained/lm1b/duo
bash scripts/train/mdlm.sh --dataset lm1b --source upstreams/mdlm \
  --output checkpoints/self_trained/lm1b/mdlm
bash scripts/train/candi.sh --dataset owt --source upstreams/candi \
  --output checkpoints/reference_reproduction/candi/owt
bash scripts/train/rdlm.sh --dataset lm1b --source upstreams/rdlm \
  --output checkpoints/self_trained/lm1b/rdlm
```

这些命令默认使用锁定 recipe。FLM 论文配置为 1,000,000 steps、global batch 512、LR `3e-4`、warmup 2,500；官方发布脚本的 1,500,000 steps 只作为 drift evidence，不覆盖论文主配置。CANDI/OWT 是项目 reference recipe，生成前必须确认 `model.ckpt` 和 `config.yaml` 已被 checkpoint lock 记录。

## 少步蒸馏

```bash
bash scripts/distill/fmlm.sh --dataset lm1b --teacher-checkpoint \
  checkpoints/official/flm/lm1b/model.safetensors \
  --output checkpoints/self_trained/lm1b/fmlm
bash scripts/distill/duo_dcd.sh --dataset lm1b --source upstreams/duo \
  --teacher checkpoints/reference_reproduction/flm_baselines/lm1b/lm1b_Duo.ckpt \
  --output checkpoints/reference_reproduction/duo_dcd/lm1b
bash scripts/distill/mdlm_sdtt.sh --dataset lm1b --source upstreams/sdtt \
  --teacher checkpoints/reference_reproduction/flm_baselines/lm1b/lm1b_MDLM_.ckpt \
  --output checkpoints/reference_reproduction/mdlm_sdtt/lm1b
bash scripts/distill/di4c.sh --model duo_di4c --dataset lm1b \
  --teacher-family uniform_duo \
  --teacher-checkpoint checkpoints/reference_reproduction/flm_baselines/lm1b/lm1b_Duo.ckpt \
  --output checkpoints/reference_reproduction/duo_di4c/lm1b \
  --student-init teacher \
  --upstream-override is_di4c=true
```

DCD/SDTT 使用 8 rounds × 10,000 steps、global batch 128、LR `6e-5`、warmup 2,500。Di4C 的 LM1B/OWT 采样 checkpoint 分别是 20,000/50,000；不能把 Zenodo masked SDTT checkpoint 当作 Duo/Di4C teacher。LM1B Di4C 默认用 `--student-init teacher`，避免公开 OWT/GPT2-vocab student 初始化与 LM1B/BERT vocab 不兼容；`--student-init scratch` 仅用于实验性随机 student 初始化。

## dry-run 与恢复

先在服务器检查命令而不启动训练：

```bash
python -m dlb.recipes --root "$DLB_ROOT" --recipe candi \
  --dataset owt --dry-run
```

默认 `--resume`。输出目录中会写 `recipe.json`、`launch_argv.json`、`provenance.json`、`config.yaml`、日志和 `completed.json`。若 checkpoint/config digest 改变，recipe 会拒绝复用旧输出；不要删除 provenance 来强行继续。

# 实验矩阵

## 任务展开

| 类别 | 逻辑方法 | 步数 | 数据集 cell | 任务数 |
|---|---|---|---:|---:|
| Many-step | FLM | 1,2,4,8,16,32,1024 | LM1B、OWT | 14 |
| Many-step | LangFlow | 1,2,4,8,16,32,1024 | LM1B、OWT | 14 |
| Many-step | Duo | 1,2,4,8,16,32,1024 | LM1B、OWT | 14 |
| Many-step | MDLM | 1,2,4,8,16,32,1024 | LM1B、OWT | 14 |
| Many-step | CANDI | 1,2,4,8,16,32,1024 | LM1B、OWT | 14 |
| Many-step | RDLM | 1000,1024 | LM1B | 2 |
| Few-step | FMLM | 1,2,4,8,16,32 | LM1B、OWT | 12 |
| Few-step | Duo+DCD | 1,2,4,8,16,32 | LM1B、OWT | 12 |
| Few-step | Duo+Di4C | 1,2,4,8,16,32 | LM1B、OWT | 12 |
| Few-step | MDLM+SDTT | 1,2,4,8,16,32 | LM1B、OWT | 12 |
| Few-step | MDLM+Di4C | 1,2,4,8,16,32 | LM1B、OWT | 12 |
| **合计** |  |  |  | **132** |

RDLM/OWT 是注册表声明的 unsupported cell，只保留一条 reason，不展开为 steps。
RDLM/LM1B 也不展开 `1,2,4,8,16,32`：RDLM 原始 baseline 是官方 SDE sampler，默认约 1000 个数值积分步；少步网格不是论文方法，也不是训练过的蒸馏版本。服务器诊断显示这些低步 RDLM 行存在 retokenization、非规范上游 ID、低 entropy 与 PPL 非单调信号，因此主表只保留官方默认 `1000` 和与本项目旧网格对齐的 `1024`。

## 结果契约

每行任务对应：

```text
results/samples/<dataset>/<model>/steps_<N>/samples.jsonl
results/samples/<dataset>/<model>/steps_<N>/run_metadata.json
results/metrics/<dataset>/<model>/steps_<N>/metrics.json
results/timing/<dataset>/<model>/steps_<N>/timing.json
```

正式任务必须 `sample_count=1024`、`seed=42`，并通过 source/config/checkpoint/adapter/environment provenance 审计。质量指标使用 GPT-2 Large Generative PPL、平均 per-sample unigram entropy、Self-BLEU；延迟使用独立 timing 的 `seconds_per_sample`。

## 可执行矩阵文件

```bash
python -m dlb.matrix --root "$DLB_ROOT" \
  --output "$DLB_ROOT/results/matrix/generation.tsv"
```

TSV 第一行是 `# schema=dlb-generation-matrix-v1`，第二行是固定字段名。解析使用 CSV/TSV reader，不使用 `eval`。不要手动重排或编辑任务行；如果 registry 改动，重新生成并审阅 unsupported inventory。

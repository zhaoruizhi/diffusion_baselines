# 实验矩阵

## 任务展开

| 类别 | 逻辑方法 | 步数 | 数据集 cell | 任务数 |
|---|---|---|---:|---:|
| Many-step | FLM | 1,2,4,8,16,32,1024 | LM1B、OWT | 14 |
| Many-step | LangFlow | 1,2,4,8,16,32,1024 | OWT | 7 |
| Many-step | Duo | 1,2,4,8,16,32,1024 | LM1B、OWT | 14 |
| Many-step | MDLM | 1,2,4,8,16,32,1024 | LM1B、OWT | 14 |
| Many-step | CANDI | 1,2,4,8,16,32,1024 | LM1B、OWT | 14 |
| Many-step | RDLM | 1,2,4,8,16,32,1024 | LM1B | 7 |
| Few-step | FMLM | 1,2,4,8,16,32 | LM1B、OWT | 12 |
| Few-step | Duo+DCD | 1,2,4,8,16,32 | LM1B、OWT | 12 |
| Few-step | Duo+Di4C | 1,2,4,8,16,32 | LM1B、OWT | 12 |
| Few-step | MDLM+SDTT | 1,2,4,8,16,32 | LM1B、OWT | 12 |
| Few-step | MDLM+Di4C | 1,2,4,8,16,32 | LM1B、OWT | 12 |
| **合计** |  |  |  | **130** |

LangFlow/LM1B 和 RDLM/OWT 是注册表声明的 unsupported cell，各自只保留一条 reason，不展开为 steps。

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

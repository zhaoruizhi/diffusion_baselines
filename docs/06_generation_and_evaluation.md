# 06. 生成、评测与 timing

## 生成矩阵

先生成并检查 TSV：

```bash
python -m dlb.matrix --root "$DLB_ROOT" \
  --output "$DLB_ROOT/results/matrix/generation.tsv" \
  --sample-count 1024 --seed 42
python -m dlb.matrix --root "$DLB_ROOT" \
  --output "$DLB_ROOT/results/matrix/generation.tsv" --validate
```

完整生成：

```bash
bash scripts/run_all.sh --root "$DLB_ROOT"
```

它会逐行继续执行，即使一个 cell 失败也会记录 `results/logs/generation_failures.tsv`。每个成功目录包含 `samples.jsonl`、`run_metadata.json`、`request.json` 和 stdout/stderr；runner 只有在样本 schema 和数量均通过后才写成功标记。

单个 cell：

```bash
bash scripts/run_one.sh --model mdlm --dataset owt --steps 32 \
  --num-samples 1024 --seed 42
```

smoke：

```bash
bash scripts/smoke_all.sh
```

smoke 输出在 `results/smoke/`，sample_count=1 的 identity 不会被正式聚合接受。

## 三项质量指标

```bash
bash scripts/evaluate_all.sh --root "$DLB_ROOT"
```

评测环境必须能读取固定 revision 的 `gpt2-large`。正式评测不允许 `--allow-partial` 或自定义 special-id；LM1B 只排除约定的 padding，保留 BOS/EOS 语义。输出为每个 cell 的 `results/metrics/.../metrics.json`，包含样本 SHA、sample_count、PPL、unigram entropy 和 Self-BLEU。

## 独立 latency

主 latency 协议固定为 batch size 1、5 warm-up、32 repetitions、CUDA synchronize；加载、首次编译、decode、指标计算和落盘不计入。单个 cell：

```bash
bash scripts/benchmark_one.sh --model flm --dataset lm1b --steps 32 \
  --seed 42 --precision author
```

批量执行：

```bash
bash scripts/benchmark_all.sh --root "$DLB_ROOT"
```

timing 输出在 `results/timing/.../timing.json`，`seconds_per_sample` 是主汇总字段；不要使用样本里的 `generation_seconds` 替代它。

## 严格汇总

```bash
python scripts/aggregate_results.py --root "$DLB_ROOT"
```

严格模式要求全部 132 个 supported task 都有 1,024 样本、三项质量指标、独立 timing 和匹配 provenance，输出：

```text
results/summary/results_long.csv
results/summary/results_wide.csv
results/summary/failures.csv
results/summary/unsupported.csv
results/summary/README.md
```

未完成时用 `--partial` 诊断；partial 报告必须标记 `complete=false`，不能作为最终论文表。

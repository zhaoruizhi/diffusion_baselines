# 08. 故障排查

## 常见问题

| 现象 | 处理 |
|---|---|
| `source is not pinned` | 在 `upstreams/<name>` 重新执行 `scripts/fetch_sources.sh`，确认无本地修改。 |
| OWT 磁盘预检失败 | 清理服务器无关缓存或换大盘；只有确认风险后才使用 `--allow-low-disk`。 |
| `checkpoint lock missing` | 重新执行 `fetch_checkpoints.py --all-public` 和 verifier，不手动编辑 lock。 |
| `processed dataset manifest` 不匹配 | 重新运行对应 preprocess/verify，检查 revision、tokenizer 和 sequence length。 |
| Conda CUDA/import 失败 | 在对应 `dlb-*` 环境运行 `envs/verify_all.sh`，修复环境后再生成。 |
| 生成 OOM | 只允许降低质量生成 batch size（如果上游 adapter 提供该选项）并记录；不要改 steps、序列长度、采样器或 precision policy。 |
| `expected 1024 records` | 该目录是 smoke/partial 或生成失败；先用 `run_one.sh` 补齐正式 1,024 样本。 |
| strict aggregation incomplete | 查看 `results/summary/failures.csv` 或阶段 failure TSV，只重跑缺失 cell。 |
| timing protocol invalid | 必须通过 `benchmark_one.sh`/`benchmark_array.sbatch` 生成，不能手写 timing.json。 |

## 恢复顺序

1. 保留失败目录、`failure.json`、stdout/stderr 和 manifest。
2. 运行 `python scripts/verify_sources.py`、`verify_data.py`、`verify_checkpoints.py`、`envs/verify_all.sh`，找出输入问题。
3. 只重跑对应阶段：生成失败重跑 `run_one.sh`，评测失败重跑 `evaluation.evaluate`/`evaluate_all.sh`，timing 失败重跑 `benchmark_one.sh`。
4. 用 `python scripts/aggregate_results.py --partial` 查看剩余缺口，全部补齐后再运行严格模式。

## 不要这样修复

不要把 unsupported 改成 supported、用另一个 dataset/checkpoint 代替、删除 provenance、把 smoke 样本复制到 1,024、把 validation PPL 当作 generative PPL，或将未经锁定的上游 `main` 当作实验版本。

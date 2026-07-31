# 02. 数据下载与预处理

## 固定数据契约

| 数据集 | Hub revision | tokenizer | 长度 | split |
|---|---|---|---:|---|
| LM1B | `billion-word-benchmark/lm1b@35161838ea9e05371a25a8db037f94fcae4c2064` | `bert-base-uncased` | 128 | `train` 与 `test` |
| OWT | `Skylion007/openwebtext@79d93d786212f7344586290adb811d4ae6a1762c` | `gpt2` | 1024 | 最后 100,000 篇为 validation |

LM1B 的 parquet revision、tokenizer revision、词表大小和文件校验均在 `artifacts/data.yaml`。OWT 约 8,013,769 篇文档，开始前需要至少 55 GiB 可用空间；只有明确使用 `--allow-low-disk` 才会绕过预检，并在 manifest 中留下证据。

## 下载、预处理和校验

以下命令只在服务器执行：

```bash
python scripts/fetch_data.py --root "$DLB_ROOT"
python scripts/preprocess_data.py --root "$DLB_ROOT" --dataset all
python scripts/verify_data.py --root "$DLB_ROOT" --dataset all
```

下载使用固定 Hub revision 和服务器本地 `data/raw/huggingface/` cache，可中断后重跑。预处理结果位于：

```text
data/processed/lm1b-bert-128/
data/processed/owt-gpt2-1024/
data/manifests/downloads.json
data/manifests/lm1b.json
data/manifests/owt.json
```

预处理会添加对应 tokenizer 的 BOS/EOS 语义并保证每个 packed sequence 长度严格一致。不要手动改 split、长度、tokenizer 或 OWT validation 边界；否则训练 recipe 和指标不再可比。

## 失败恢复

若 `verify_data.py` 报错，保留 `data/manifests/` 和日志，先检查磁盘、Hub cache 和 revision，再重新执行对应阶段。不要把不完整的 `data/processed/*` 改名为完成目录。

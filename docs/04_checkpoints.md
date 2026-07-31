# 04. Checkpoint 下载与校验

## 资源原则

checkpoint 清单是 `artifacts/checkpoints.yaml`。每个资源记录 backend、revision/record、目标目录、teacher family、所需文件和许可证链接。下载脚本不会根据 moving `main` 选择模型。

服务器执行：

```bash
python scripts/fetch_checkpoints.py --root "$DLB_ROOT" --all-public
python scripts/verify_checkpoints.py --root "$DLB_ROOT"
```

`--all-public` 是有意的确认开关；不带它只会退出并提示，不会下载。下载完成后生成 `artifacts/checkpoint_lock.json`，其中包含实际文件 SHA256。只要 lock 不完整或 manifest SHA 不匹配，runner 会拒绝运行。

## 目录约定

```text
checkpoints/official/
checkpoints/reference_reproduction/
checkpoints/self_trained/
artifacts/checkpoint_lock.json
```

官方 Hugging Face 资源包括 FLM/FMLM、LangFlow OWT、Duo OWT、Duo+DCD OWT、MDLM OWT、MDLM+SDTT OWT。Google Drive/Zenodo 资源用于 RDLM LM1B、LM1B reproduction、raw Duo OWT 和 MDLM Di4C。Duo/uniform teacher 不能使用 masked MDLM/SDTT 文件替代，反之亦然。

## 下载失败处理

脚本会保留失败状态和校验信息。不要手动编辑 `checkpoint_lock.json` 或用空文件占位。网络中断后直接重跑下载脚本；若资源确实不可达，在结果报告中保留失败原因，并把相应 cell 视为未完成，而不是 unsupported。

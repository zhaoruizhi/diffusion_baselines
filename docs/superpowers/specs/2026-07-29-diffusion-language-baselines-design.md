# Diffusion Language Baselines 实验工程设计

**日期：** 2026-07-29
**基准论文：** *Flow Map Language Models: One-step Language Modeling via Continuous Denoising*，arXiv:2602.16813v3
**工程根目录：** `/Users/zhaoruizhi/Documents/diffusion_baseline`

## 1. 目标与完成标准

本工程用于在 LM1B 与 OpenWebText（OWT）上复现并统一比较连续、离散、混合扩散语言模型及其少步蒸馏版本。最终交付必须让使用者把整个目录复制到 Linux GPU 服务器后，通过文档中的命令完成环境安装、数据与 checkpoint 校验、样本生成、四项指标计算和结果汇总。

完成标准如下：

1. 本地保存并锁定 8 个上游源码仓库：FLM、LangFlow、Duo、MDLM、CANDI、RDLM、SDTT、Di4C。
2. 本地保存 LM1B 与 OWT 原始数据快照及经过固定 tokenizer、长度和划分处理的数据缓存；大文件同时提供可恢复的一键下载脚本和校验清单。
3. 每个不兼容代码库拥有独立 Conda 环境定义；共享同一代码库的方法复用环境，但在清单中逐一映射。
4. 覆盖指定的 6 个 many-step 模型和 5 个 few-step/distilled 模型。
5. many-step 使用步数 `1,2,4,8,16,32,1024`；few-step 使用 `1,2,4,8,16,32`。
6. 每个可运行的“数据集 × 模型 × 步数”配置生成恰好 1,024 个有效样本。
7. 每个配置报告 GPT-2 Large Generative Perplexity、平均单样本 unigram entropy、Self-BLEU 和 generation seconds/sample。
8. 所有产物记录源码 commit、checkpoint 来源及 SHA256、数据集 revision、命令、随机种子、GPU、CUDA、PyTorch、精度和 batch size。
9. 官方资源缺失的组合明确输出 `unsupported`，不会用另一数据集或另一训练配置冒充。
10. 提供单机 Bash 和 Slurm 两条执行路径，以及可恢复、可重复运行的阶段化命令。

## 2. 实验范围

### 2.1 Many-step

| 逻辑模型 | 方法 | 首选源码 |
|---|---|---|
| FLM | 连续 one-hot flow teacher | `david3684/flm` |
| LangFlow | embedding-space continuous diffusion/flow | `nealchen2003/LangFlow` |
| Duo | uniform-state discrete diffusion | `s-sahoo/duo` |
| MDLM | masked discrete diffusion | `kuleshov-group/mdlm` |
| CANDI | hybrid discrete-continuous diffusion | `patrickpynadath1/candi-diffusion` |
| RDLM | Riemannian continuous diffusion | `harryjo97/RDLM` |

### 2.2 Few-step / distilled

| 逻辑模型 | Teacher / distillation | 首选源码 |
|---|---|---|
| FMLM | FLM flow-map student | `david3684/flm` |
| Duo + DCD | Duo + Discrete Consistency Distillation | `s-sahoo/duo` |
| Duo + Di4C | Duo + dimensional-correlation distillation | FLM 复现资产与 `sony/di4c` 方法代码 |
| MDLM + SDTT | MDLM + Self-Distillation Through Time | `jdeschena/sdtt` |
| MDLM + Di4C | MDLM + dimensional-correlation distillation | `sony/di4c` |

这些是 11 个逻辑实验模型而不是 11 个独立仓库。FLM/FMLM 和 Duo/DCD 分别共享源码与环境。

## 3. 资源覆盖与证据等级

结果按以下来源等级标注：

- `official`：模型作者发布的源码与 checkpoint。
- `reference_reproduction`：FLM 论文作者按统一设置训练并发布的复现 checkpoint。
- `self_trained`：由本工程命令在用户服务器上训练所得。
- `unsupported`：公开资源和可执行训练代码不足，无法忠实复现。

计划覆盖矩阵：

| 模型 | LM1B | OWT |
|---|---|---|
| FLM | official | official |
| FMLM | official | official |
| LangFlow | `unsupported`，除非作者发布 LM1B 权重或训练代码 | official checkpoint/inference |
| Duo | reference_reproduction 或 self_trained | official |
| Duo + DCD | reference_reproduction 或 self_trained | official |
| MDLM | reference_reproduction 或 self_trained | official |
| MDLM + SDTT | reference_reproduction 或 self_trained | official |
| CANDI | reference_reproduction 或 self_trained | author-shared/FLM reference reproduction；若无公开 checkpoint 则 self_trained |
| RDLM | official | `unsupported`，原论文也未报告该单元格 |
| Duo + Di4C | reference_reproduction 或 self_trained | FLM 复现资产；若无公开 checkpoint 则 self_trained |
| MDLM + Di4C | reference_reproduction 或 self_trained | official/FLM 采用的中间 checkpoint |

下载阶段将通过公开 API 枚举实际文件并生成最终 `artifacts/checkpoint_manifest.json`。若上游在 2026-07-29 之后改变资源，锁文件仍保留本次可复现版本。

## 4. 数据集与划分

### 4.1 LM1B

- 来源：Hugging Face `lm1b` 数据集的固定 revision。
- 训练使用官方 `train` split；验证/参考熵使用官方 `test` split，与 FLM/MDLM 代码行为一致。
- tokenizer：`bert-base-uncased` 的固定 revision，词表大小 30,522。
- 将文本连续 tokenization 并 packing 到长度 128；每个 block 包含模型代码所需的 BOS/CLS 和 EOS/SEP 语义。
- 不把测试数据用于训练、蒸馏或 checkpoint 选择。

### 4.2 OpenWebText

- 来源：Hugging Face `openwebtext` 的固定 revision。
- 按 FLM、Duo、MDLM 代码口径，在数据集原始确定性顺序上使用 `train[:-100000]` 作为训练集、`train[-100000:]` 作为验证集。
- tokenizer：`gpt2` 的固定 revision，词表大小 50,257；模型实现需要 padding token 时仅在适配层添加，不改变生成词表语义。
- 连续 tokenization 并 packing 到长度 1,024。
- OWT 完整数据约占 39.8 GB；下载脚本在开始前检查磁盘空间，支持 Hugging Face 缓存恢复和离线复用。

### 4.3 本地布局

```text
data/
  raw/huggingface/          # 完整 Hub snapshot/cache
  processed/lm1b-bert-128/
  processed/owt-gpt2-1024/
  manifests/               # revision、文件大小、SHA256、split 统计
```

预处理完成后，验证器检查 split 名称、样本数、token ID 范围、固定长度、tokenizer revision 和随机抽样解码。

## 5. 工程架构

```text
diffusion_baseline/
  README.md
  upstreams/               # 8 个保留 .git 的浅克隆/完整工作树
  artifacts/               # source/checkpoint/data 锁文件
  checkpoints/
    official/
    reference_reproduction/
    self_trained/
  data/
  envs/                    # Conda YAML 和创建/验证脚本
  adapters/                # 每个逻辑方法的薄适配器
  evaluation/              # 与模型无关的统一指标实现
  configs/                 # 统一实验注册表和模型矩阵
  scripts/                 # 下载、预处理、生成、评测、汇总
  slurm/                   # job array 与资源模板
  tests/                   # 纯 CPU 单测和静态 smoke test
  results/
    samples/
    metrics/
    timing/
    logs/
    summary/
  docs/
```

上游仓库视为只读。所有路径覆盖、输出转换、计时和兼容修补放在本工程适配层；确需修补上游时保存为 `patches/<repo>/*.patch`，并在锁文件记录应用顺序和补丁 SHA256。

## 6. Conda 环境设计

创建以下环境：

| 环境 | 覆盖方法 | 版本原则 |
|---|---|---|
| `dlb-flm` | FLM、FMLM、FLM 复现适配 | 上游要求，CUDA 12.4 系列，FlashAttention 2.8.3 |
| `dlb-langflow` | LangFlow | Python 3.12，上游 CUDA PyTorch wheel |
| `dlb-duo` | Duo、Duo+DCD | Python 3.12，CUDA 12.4，FlashAttention 2.7.4.post1 |
| `dlb-mdlm` | MDLM | 上游 `requirements.yaml` 的锁定副本 |
| `dlb-candi` | CANDI | 从其 requirements 锁定并显式固定 PyTorch/CUDA |
| `dlb-rdlm` | RDLM | Python 3.9、PyTorch 2.3.1 |
| `dlb-sdtt` | MDLM+SDTT | Python 3.10；替换失效 nightly CPU 安装为与服务器 CUDA 匹配的稳定组合并记录差异 |
| `dlb-di4c` | MDLM/Duo + Di4C | 基于 `sony/di4c/sdtt` 依赖锁定 |
| `dlb-eval` | 统一评测、汇总 | Transformers、PyTorch、NumPy、SciPy、SacreBLEU/NLTK、pandas |

每个 YAML 旁提供：

- `create_<name>.sh`：创建环境并安装 FlashAttention 等需要分步构建的包。
- `verify_<name>.sh`：输出包版本、CUDA 可用性并执行 import smoke test。
- `pack_<name>.sh`：可选用 `conda-pack` 生成服务器可搬运归档。

环境脚本只在服务器执行；本地不创建 Conda 环境。

## 7. 统一运行接口

逻辑命令采用统一参数：

```bash
bash scripts/run_one.sh \
  --model flm \
  --dataset lm1b \
  --steps 32 \
  --num-samples 1024 \
  --seed 42 \
  --device cuda:0
```

`run_one.sh` 根据 `configs/experiments.yaml` 选择环境、上游入口、checkpoint 和适配器。适配器必须输出：

```text
results/samples/<dataset>/<model>/steps_<N>/samples.jsonl
results/samples/<dataset>/<model>/steps_<N>/run_metadata.json
```

每条 JSONL 至少包含：

```json
{
  "sample_id": 0,
  "text": "decoded text",
  "token_ids": [101, 2023, 102],
  "seed": 42,
  "generation_seconds": 0.0123
}
```

适配器将上游参数映射到统一语义：采样步数、样本数、序列长度、temperature 1.0、ancestral sampler（离散基线）、Euler solver（FLM）、FMLM gamma-sampling 和 RDLM 官方 SDE sampler。模型特有的必要选项记录到 metadata。

## 8. 实验矩阵与随机性

- many-step：FLM、LangFlow、Duo、MDLM、CANDI、RDLM × `1,2,4,8,16,32,1024`。
- few-step：FMLM、Duo+DCD、Duo+Di4C、MDLM+SDTT、MDLM+Di4C × `1,2,4,8,16,32`。
- 两个数据集分别运行；资源矩阵中的 `unsupported` 单元格跳过并生成机器可读原因。
- 每个配置生成 1,024 个样本；写入采用原子临时文件，完成且通过 schema/数量校验后再改名。
- 全局 seed 为 42，并记录 Python、NumPy、PyTorch CPU/CUDA seed。不同模型状态空间不强求共享同一初始噪声，但同一模型不同步数在上游支持时共享确定性初始随机流。
- 默认生成精度为作者 checkpoint 推荐精度；任何 fp32/fp16/bf16 差异写入 metadata。

## 9. 评估指标

### 9.1 Generative Perplexity

- 使用 Hugging Face `gpt2-large` 的固定 revision，`eval()` 且禁用 dropout。
- 对每条解码文本重新使用 GPT-2 tokenizer 编码。
- 采用标准 causal next-token negative log-likelihood，忽略第一个预测位置和 padding；聚合全部有效预测 token 的 NLL 后计算 `exp(total_nll / total_tokens)`。
- 评测 batch size 可按显存调整，但必须记录；相同文本在不同 batch size 下通过回归测试保持数值一致。

### 9.2 Unigram Entropy

- 按 FLM 论文定义计算每个样本内部的 subword 经验分布熵，再对 1,024 个样本取算术平均。
- 使用对应数据集 tokenizer 的 token IDs；排除 padding，保留属于生成序列定义的 BOS/EOS 规则。
- 使用自然对数，单位为 nats。

### 9.3 Self-BLEU

- 与 FLM 仓库实现逐行核对并保留其 n-gram 阶数、权重、平滑和归一化约定。
- 主结果报告 `[0,1]` 范围分数，越低表示样本间 n-gram 多样性越高；完全相同的合成样本回归用例必须接近 1.0。
- 计算使用固定 seed 的确定性 reference 抽样或原实现的全体 reference 规则，并写入 metadata。

### 9.4 Generation Time / Sample

- 主指标为 batch size 1 的 latency，避免不同模型显存占用导致不公平的批量吞吐比较。
- checkpoint 加载、数据加载、首次编译、tokenizer decode 和落盘不计时。
- 先执行 5 个不计时 warm-up 样本；计时前后调用 `torch.cuda.synchronize()`。
- 对 32 个样本计时并报告总墙钟时间除以 32；同时保存逐样本时间、中位数和标准差。
- 质量生成的 1,024 个样本可使用模型适合的 batch size，不作为主时间指标。
- 同一张 GPU、同一功耗/频率策略完成横向比较；metadata 记录 GPU 型号、驱动、CUDA 和环境。

## 10. 执行阶段

1. `bootstrap`：检测 Linux、Conda/Mamba、GPU、磁盘和网络。
2. `fetch_sources`：克隆并锁定 8 个源码仓库。
3. `fetch_data`：下载 LM1B、OWT 和 tokenizer snapshots，生成校验清单。
4. `fetch_checkpoints`：下载官方与 reference reproduction 权重，生成 SHA256。
5. `create_envs`：在服务器建立并验证独立环境。
6. `preprocess`：生成两套固定数据缓存并验证 split。
7. `smoke`：每个受支持方法以 1 个样本、最小步数执行。
8. `generate`：运行完整 1,024 样本实验矩阵。
9. `evaluate`：统一计算三项质量/多样性指标。
10. `benchmark`：按统一 latency 协议运行计时。
11. `aggregate`：生成长表 CSV、宽表 CSV、Markdown 汇总和失败矩阵。

每阶段写 `.done` 标记和输入摘要；输入 commit、checkpoint、配置或脚本摘要变化时自动判定旧标记失效。

## 11. 错误处理与可恢复性

- 下载使用临时后缀和断点续传；校验失败的文件移入 `quarantine/`，不直接覆盖。
- 启动任务前检查 checkpoint、数据、GPU、可用磁盘和环境导入。
- 运行失败写结构化 `failure.json`，包含命令、退出码、日志路径和最近 stderr，不影响其他矩阵单元继续执行。
- OOM 时不自动改变精度或序列长度；允许仅减小质量生成 batch size，并把变化记录到 metadata。
- 样本不足 1,024、存在空文本、JSON schema 不合法或重复 `sample_id` 时，评测器拒绝出正式结果。
- 不支持的组合由注册表显式声明，汇总表显示原因而不是空白值。

## 12. 验证策略

### 静态验证

- 所有 Bash 脚本执行 `bash -n`，可用时再执行 ShellCheck。
- 所有 YAML/JSON 可解析，模型名、数据集名和步数网格通过 schema 校验。
- 锁文件中的每个仓库、数据 revision、checkpoint URL 和 SHA256 均可追溯。

### CPU 单元测试

- 指标：已知文本的 PPL 聚合、entropy、Self-BLEU 极端情况。
- 样本 schema：数量、ID、文本和 token ID 验证。
- 注册表：11 个逻辑模型全部存在，步数类别正确，unsupported 单元格有原因。
- 命令渲染：每个受支持单元格能生成无未解析变量的 shell 命令。

### 服务器 smoke test

- 每个环境完成 CUDA/import 检查。
- 每个受支持方法至少生成 1 个正确长度样本。
- GPT-2 Large 对少量样本可完成 PPL 计算。
- 计时日志包含 CUDA synchronize 后的正数时间。

### 完整结果验收

- 每个受支持配置恰好有 1,024 个样本和四项指标。
- 汇总行数与注册表计算出的受支持矩阵单元数一致。
- FLM/FMLM 等官方 checkpoint 的趋势与论文表格同方向；不以完全相同浮点值作为跨硬件验收条件。
- 所有结果可从 metadata 中的精确命令重新执行。

## 13. 文档交付

根 `README.md` 提供从空服务器开始的最短路径；另提供：

- `docs/01_sources_and_availability.md`
- `docs/02_datasets.md`
- `docs/03_environments.md`
- `docs/04_checkpoints.md`
- `docs/05_training.md`
- `docs/06_generation_and_evaluation.md`
- `docs/07_slurm.md`
- `docs/08_troubleshooting.md`
- `docs/EXPERIMENT_MATRIX.md`

命令同时给出逐方法运行和全矩阵运行方式，所有路径以工程根目录环境变量 `DLB_ROOT` 为基准，不写死当前 Mac 路径。

## 14. 明确不做的事项

- 不在当前 Mac 上安装 GPU Conda 环境或运行训练。
- 不声称公开资源缺失的实验已经可复现。
- 不把 validation perplexity 混入 Generative Perplexity。
- 不自动更改序列长度、temperature、采样器或 checkpoint 以掩盖失败。
- 不将大体量 checkpoint、Hugging Face cache 或实验输出提交到父 Git 仓库；这些资产由 manifest 和校验和追踪。

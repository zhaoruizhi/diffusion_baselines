# Diffusion Language Baselines 实验工程设计

**日期：** 2026-07-29
**基准论文：** *Flow Map Language Models: One-step Language Modeling via Continuous Denoising*（arXiv:2602.16813v3）

## 目标

在 LM1B 和 OpenWebText（OWT）上统一运行 6 个 many-step 模型和 5 个 few-step/distilled 模型。工程复制到 Linux GPU 服务器后，应能通过明确的 Bash/Slurm 命令完成环境安装、数据与 checkpoint 校验、训练或蒸馏（需要时）、1,024 样本生成、四项指标计算和结果汇总。

完成标准：

1. 当前目录包含 8 个锁定 commit 的上游源码工作树和可重建脚本。
2. 当前目录包含 LM1B、OWT 完整下载与预处理缓存，或包含明确可恢复的下载失败证据；所有可获得 checkpoint 同样本地化并校验。
3. 每个不兼容代码库拥有独立 Conda YAML、创建、验证和可选打包脚本。
4. many-step 步数为 `1,2,4,8,16,32,1024`，few-step 步数为 `1,2,4,8,16,32`。
5. 每个受支持配置生成恰好 1,024 个样本，并报告 GPT-2 Large Gen. PPL、平均单样本 unigram entropy、Self-BLEU 和 seconds/sample。
6. 每条结果能追溯到源码 SHA、checkpoint SHA256、数据 revision、命令、随机种子、GPU/CUDA/PyTorch、精度和 batch size。

## 模型范围

### Many-step

| 模型 | 类型 | 上游源码 |
|---|---|---|
| FLM | one-hot continuous flow teacher | `david3684/flm` |
| LangFlow | embedding-space continuous diffusion | `nealchen2003/LangFlow` |
| Duo | uniform-state discrete diffusion | `s-sahoo/duo` |
| MDLM | masked discrete diffusion | `kuleshov-group/mdlm` |
| CANDI | discrete-continuous hybrid diffusion | `patrickpynadath1/candi-diffusion` |
| RDLM | Riemannian continuous diffusion | `harryjo97/RDLM` |

### Few-step / distilled

| 模型 | 组合 | 上游源码 |
|---|---|---|
| FMLM | FLM flow-map student | `david3684/flm` |
| Duo + DCD | uniform teacher + consistency distillation | `s-sahoo/duo` |
| Duo + Di4C | uniform teacher + dimensional-correlation distillation | FLM 复现资产与 `sony/di4c` |
| MDLM + SDTT | masked teacher + self-distillation through time | `jdeschena/sdtt` |
| MDLM + Di4C | masked teacher + dimensional-correlation distillation | `sony/di4c` |

这些是 11 个逻辑模型，不是 11 个独立仓库。FLM/FMLM 和 Duo/DCD 分别共享源码与环境。

## 资源来源等级与支持矩阵

- `official`：模型作者发布的源码和 checkpoint。
- `reference_reproduction`：FLM 论文作者按统一设置训练的复现资产。
- `self_trained`：由本工程脚本在服务器训练或蒸馏所得。
- `unsupported`：公开资源或可执行训练代码不足，无法忠实复现。

| 模型 | LM1B | OWT |
|---|---|---|
| FLM | official | official |
| FMLM | official | official |
| LangFlow | unsupported：未完整公开 LM1B 权重/训练代码 | official checkpoint/inference |
| Duo | reference reproduction 或 self-trained | official |
| Duo + DCD | reference reproduction 或 self-trained | official |
| MDLM | reference reproduction 或 self-trained | official |
| MDLM + SDTT | reference reproduction 或 self-trained | official |
| CANDI | reference reproduction 或 self-trained | author-shared/reference reproduction 或 self-trained |
| RDLM | official | unsupported：原论文亦未报告 OWT |
| Duo + Di4C | reference reproduction 或 self-trained | FLM 复现资产或 self-trained |
| MDLM + Di4C | reference reproduction 或 self-trained | official/FLM 使用的中间 checkpoint |

不会以另一数据集、teacher family 或训练配置冒充缺失组合。所有可下载资产在 manifest 中记录 URL、文件、许可、来源等级和 SHA256。

## 数据集

### LM1B

- 固定 Hugging Face `lm1b` revision。
- `train` 用于训练，`test` 用于验证/参考熵。
- `bert-base-uncased` 固定 revision，词表 30,522。
- 连续 tokenization 并 packing 到长度 128，保留上游要求的 BOS/CLS 与 EOS/SEP 语义。

### OWT

- 固定 Hugging Face `openwebtext` revision。
- 原始确定性顺序的 `train[:-100000]` 为训练集，`train[-100000:]` 为 validation。
- `gpt2` 固定 revision，词表 50,257。
- 连续 tokenization 并 packing 到长度 1,024。
- 完整数据约占 39.8 GB；下载前检查至少 55 GiB 空间，支持缓存恢复和离线复用。

目录布局：

```text
data/
  raw/huggingface/
  processed/lm1b-bert-128/
  processed/owt-gpt2-1024/
  manifests/
```

预处理验证 split、样本数、token 范围、固定长度、tokenizer revision 和抽样解码。

## 工程架构

```text
upstreams/          # 8 个只读源码工作树
artifacts/          # source/data/checkpoint/environment 锁文件
checkpoints/        # official/reference_reproduction/self_trained
data/               # raw/processed/manifests
envs/               # Conda YAML 与 create/verify/pack
adapters/           # 上游命令与统一输出适配
evaluation/         # PPL、entropy、Self-BLEU、timing
configs/            # 实验注册表和支持矩阵
scripts/            # 下载、预处理、运行、评测、汇总
slurm/              # job arrays
src/dlb/            # schema、注册表、runner、聚合
tests/              # CPU 单测和静态 smoke tests
results/            # samples/metrics/timing/logs/summary
docs/               # 中文服务器手册
```

上游仓库默认只读。路径覆盖、输出转换、计时和兼容修复放在适配层；确需改上游时保存为 `patches/<repo>/*.patch` 并记录补丁 SHA256。

## Conda 环境

| 环境 | 方法 | 关键约束 |
|---|---|---|
| `dlb-flm` | FLM/FMLM | 上游 PyTorch/CUDA，FlashAttention 2.8.3 |
| `dlb-langflow` | LangFlow | Python 3.12，CUDA PyTorch wheel |
| `dlb-duo` | Duo/DCD | Python 3.12、CUDA 12.4、FlashAttention 2.7.4.post1 |
| `dlb-mdlm` | MDLM | 锁定上游 requirements YAML |
| `dlb-candi` | CANDI | 锁定 requirements 并显式固定 PyTorch/CUDA |
| `dlb-rdlm` | RDLM | Python 3.9、PyTorch 2.3.1 |
| `dlb-sdtt` | MDLM+SDTT | Python 3.10，使用稳定 CUDA 兼容依赖 |
| `dlb-di4c` | Di4C 组合 | 基于 `sony/di4c/sdtt` 锁定 |
| `dlb-eval` | 统一评测 | Transformers/PyTorch/NumPy/SciPy/BLEU/pandas |

环境只在服务器创建；当前 Mac 仅生成并验证配置。创建脚本须支持已有环境，验证脚本输出机器可读版本，打包脚本使用 `conda-pack`。

## 统一运行与样本格式

```bash
bash scripts/run_one.sh \
  --model flm --dataset lm1b --steps 32 \
  --num-samples 1024 --seed 42 --device cuda:0
```

输出：

```text
results/samples/<dataset>/<model>/steps_<N>/samples.jsonl
results/samples/<dataset>/<model>/steps_<N>/run_metadata.json
```

每条 JSONL 至少包含 `sample_id`、`text`、`token_ids`、`seed`、`generation_seconds`。写入先进入 `.partial`，数量/schema/唯一 ID 验证通过后原子改名。

采样规则：离散模型使用 ancestral/temperature 1.0，FLM 使用 Euler，FMLM 使用 gamma-sampling，RDLM 使用官方 SDE sampler。全局 seed 为 42；不同模型不强求共享状态空间不同的初始噪声。

## 指标

### Generative Perplexity

固定 `gpt2-large` revision，`eval()` 且无 dropout。每条文本重新用 GPT-2 tokenizer 编码，计算 causal next-token NLL；跨 batch 聚合全部有效 token 的 `total_nll / total_tokens`，最后只 exponentiate 一次。

### Unigram Entropy

以数据集 tokenizer token IDs 计算每个样本的经验 unigram 分布熵，使用自然对数（nats），排除 padding，再对 1,024 个样本取算术平均。BOS/EOS 处理与 FLM 实现逐行对齐并写入 metadata。

### Self-BLEU

移植并回归验证 FLM 的 n-gram 阶数、权重、平滑和归一化。主结果位于 `[0,1]`；完全相同样本应接近 1，越低表示越多样。

### Generation Time / Sample

主指标为 batch size 1 latency。排除 checkpoint 加载、首次编译、decode 和落盘；先 5 次 warm-up，再计时 32 个样本；每次前后 CUDA synchronize。报告均值、median、标准差和逐样本时间，同时记录 GPU/驱动/CUDA/精度。

## 执行阶段与恢复

1. `bootstrap`：检查 Linux、Conda、GPU、磁盘和网络。
2. `fetch_sources`：克隆 8 个仓库并锁定 SHA。
3. `fetch_data`：下载数据与 tokenizer snapshots。
4. `fetch_checkpoints`：下载公开资产并写 SHA256。
5. `create_envs`：服务器创建并验证环境。
6. `preprocess`：生成两套固定缓存。
7. `smoke`：每个受支持方法生成 1 个样本。
8. `generate`：完整 1,024 样本矩阵。
9. `evaluate`：计算三项质量/多样性指标。
10. `benchmark`：统一 latency protocol。
11. `aggregate`：生成长表、宽表、Markdown 和失败矩阵。

下载使用断点续传与临时后缀；校验不符的文件移入 `quarantine/`。运行失败生成 `failure.json` 并允许其他矩阵项继续。OOM 只允许减小质量生成 batch size，不自动改变精度、长度或采样器。样本不足、空文本、schema 错误或重复 ID 时拒绝产生正式指标。

## 验证与文档

- Bash/Slurm 执行 `bash -n`，YAML/JSON/schema 全部解析。
- CPU 单测覆盖注册表、11 个模型、支持矩阵、步数、样本 schema、PPL 聚合、entropy、Self-BLEU、计时与命令渲染。
- 服务器 smoke test 覆盖每个环境的 CUDA/import 和每个受支持方法的最小生成。
- 完整验收要求每个受支持配置有 1,024 样本和四项指标；汇总行数必须等于注册表的支持矩阵。
- 文档包含来源、数据、环境、checkpoint、训练/蒸馏、生成/评测、Slurm 和故障排查，并提供逐方法及全矩阵命令。

## 不在当前 Mac 执行

- 不安装 GPU Conda 环境或运行训练。
- 不声称未在服务器执行的 GPU smoke test 已完成。
- 不把 validation perplexity 当作 Generative Perplexity。
- 不提交大体量源码 clone、HF cache、checkpoint 或结果到父 Git；使用 manifest 和校验和追踪。

# ELF 加入 baseline：服务器实验方案与操作步骤

**2026-09-13 最新进度：12 个正式质量配置已完成。后续请执行 [剩余质量实验步骤](14_elf_quality_remaining.md)，不要再重复核心 8/16/32/64 步。timing 继续暂缓。**

| 实验 | 已确认完成 | 尚需补齐 |
|---|---|---|
| OWT unconditional | 8/16/32/64 步各 1024 条，全部 valid=true | 1/2/4/128/256/512/1024 步 |
| WMT14 validation | 8/16/32/64 步各 3000 条；BLEU 24.1705/25.3318/26.1424/26.6741 | 1/2/4 步；冻结的 32/64 步 test |
| XSum validation | 8/16/32/64 步各 11332 条；32 步 ROUGE 35.8402/12.1714/27.6562，64 步 36.0270/12.3522/27.8165 | 1/2/4 步；冻结的 32/64 步 test |
| OWT 前缀续写 | 尚未收到结果 | 先准备 prompts、跑核心，再补扩展 |
| 所有正式 timing | 用户要求暂缓 | GPU 独占时另行安排 |

OWT 的四个正式 PPL 为 70.9471/32.1761/24.3517/19.1247，entropy 为 5.2528/5.1609/5.1651/5.0774。32 步另有 1000 条 sanity，不算额外正式配置。条件目录的 summary.csv 为 8 行，与两项任务各四步一致，不包含另一根目录的 OWT。

OWT 正式质量最初写在 `results/elf-h200-gpu2`；EOS 修复后的条件实验写在 `results/elf-h200-gpu2-eos-v2`。只检查后者会误以为没有跑过 OWT。下面只读扫描所有 ELF 根目录，不重新生成或评测；`valid` 是已有指标文件记录的状态，此清单不替代 SHA256 完整性校验。

```bash
cd ~/diffusion_baseline
python - <<'PY'
import json
from pathlib import Path
for steps in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
    paths = sorted(Path('results').glob(
        f'elf*/quality/owt/unconditional/steps_{steps}/seed_42/compile_0'))
    if not paths:
        print(f'{steps:>4} steps: 未找到质量目录')
    for path in paths:
        mp, gp = path / 'metrics.json', path / 'generation.json'
        if mp.exists():
            r = json.loads(mp.read_text())
            m = r['metrics']
            print(f'{steps:>4} steps: N={r["sample_count"]}, valid={r.get("valid")}, '
                  f'PPL={m.get("generative_ppl", {}).get("perplexity")}, '
                  f'entropy={m.get("entropy_gpt2_nats")}, path={path}')
        elif gp.exists():
            r = json.loads(gp.read_text())
            print(f'{steps:>4} steps: 已生成 {r["sample_count"]} 条，缺指标；path={path}')
        else:
            print(f'{steps:>4} steps: 目录存在但生成未完成；path={path}')
PY
```

正式 OWT 格子要求每步 1024 条，seed=42。最新清单已确认核心四步完成，此命令留作后续检查；只补缺失步骤：有完整生成文件时只 evaluate；没有目录时 generate 后 evaluate；未完成目录保留并用新重试根目录处理。已确认成功的 OWT 不受条件 EOS 修复影响，无需重跑。质量可与同伴的小进程共享 GPU，但不要与自己的 WMT14/XSum 生成同时占用同一张卡。OWT 与旧 baseline 的 tokenizer、EOS 和实际输出长度差异见下文“可比性”，不能仅把两种 native token 的 entropy 放进同一列。

**同步故障补充：若看到 `Cannot fast-forward to multiple branches`，代码尚未更新，不能继续运行实验。请先按 [文档 13 第 1 步](13_elf_eos_fix.md) 显式 fetch/merge 并验证 EOS 提交。该节也解释 GPU 2 上小进程对质量运行和正式 timing 的不同影响。**

**最新修复：服务器诊断确认原始 WMT14/XSum 条件缺少作者预处理中的 EOS，造成 XSum 空条件中断，并解释了 WMT14 降分的输入差异。请优先执行 [EOS 修复后的恢复步骤](13_elf_eos_fix.md)：更新代码、CPU 对照检查、重跑条件 64 步全集，再测其余质量和 timing。** OWT 与已有 official-validation 结果保留；旧原始条件生成结果不能只重算指标。使用新结果目录，无需重新下载或导出数据。

**你已在物理 GPU 2 完成三项 smoke 和 1000 样本 sanity，日志中的主指标合理，无需重跑。接下来请按 [DS210029 单张 H200 逐步执行清单](11_elf_h200_next_steps.md) 的 B 恢复环境，再从 E 开始。** D1 已记录此次指标判读与长度警告解释；E/G 分别提供核心质量与独立 timing 命令，F 是另外准备的 OWT 前缀续写。下文保留完整协议和此前故障说明，避免重复跑下载或混用结果目录。

核查日期：2026-09-10。先用官方公开权重建立 ELF 的质量和耗时基线；你的方法尚未训练 WMT14/XSum，本轮不安排训练。所有下载、环境安装和模型运行都在 Linux GPU 服务器完成。本地新增的是锁文件、脚本、CPU 合约检查和本文档。

**交付状态（2026-09-13 更新）：** 下述独立 ELF 入口已做本地 CPU 合约检查与语法检查；服务器结果确认 CUDA、三项 smoke/sanity、OWT 核心四步，以及 EOS 修复后两项完整 validation 各四步，共 12 个正式质量配置；不能声称完整实验矩阵已完成或已经取得加速。旧的 `run_one.sh --model elf`、`run_all.sh`、conditional registry 和严格聚合器**尚不支持 ELF**；请使用本文新增的入口。结果根目录由 `ELF_RESULTS` 指定，OWT 原根目录与条件 EOS 修复根目录不同，不改动已有矩阵。

**1. 本次需要跑什么**

| 实验 | 权重 | 输入与输出长度 | 首轮步数 | 质量指标 | timing |
|---|---|---|---|---|---|
| OWT unconditional | ELF-B-owt-torch | 原生 1024 T5 token canvas | 1,2,4,8,16,32,64,128,256,512,1024 | GPT-2 Large Gen PPL、GPT-2 unigram entropy；保存实际输出长度 | 同步 sampler latency |
| OWT 零样本前缀续写 | 同一 OWT 权重 | 现有 64 GPT-2 token 前缀解码成文本，再用 T5 编码；总 canvas 1024 | 同上，先跑 8,16,32,64 | suffix CPPL、entropy、grouped Self-BLEU；可选 MAUVE | 缓存条件、包含 T5 条件编码两种口径 |
| WMT14 De→En 翻译 | ELF-B-de-en-torch | source 上限 64、target 64、总长 128 T5 token | 1,2,4,8,16,32,64 | corpus BLEU；完整英文 reference | 两种口径 |
| XSum 摘要 | ELF-B-xsum-torch | source 上限 1024、target 64、总长 1088 T5 token | 1,2,4,8,16,32,64 | ROUGE-1/2/L F1，均值与样本间 SEM | 两种口径 |

截图里的 De-En 是 WMT14 德译英，XSum 是摘要，不是两个翻译数据集。ELF-B 主网络 105M；条件生成还需约 35M 的 T5-small encoder。优先做 B，M/L 不属于本轮必需项。

OWT 主结果沿用本仓库的 1024 个样本、seed=42；另用 1000 个样本做论文 sanity check。前缀实验使用原有 1024 prompts，前 256 prompts 各额外采样 4 次，共 2048 条。WMT14/XSum 正式评测使用完整 split，不能把默认 1000 条当作完整验证集/测试集。

**2. 已核实的公开资源与采样配置**

源代码使用官方 [PyTorch 分支](https://github.com/lillian039/ELF/tree/pytorch_elf)，不用面向 TPU 的 JAX main，也不用后来的 distillation 分支。

| 资源 | 固定版本/文件 |
|---|---|
| 源代码提交 | `b29d8833609e9ab7f67cd9da39435ac5cea04837` |
| [OWT checkpoint](https://huggingface.co/embedded-language-flows/ELF-B-owt-torch) | `checkpoint_95085`；revision `146f84133c1389bfd4ef47f14ec7a955da22faa7` |
| [德译英 checkpoint](https://huggingface.co/embedded-language-flows/ELF-B-de-en-torch) | `checkpoint_880600`；revision `93ce98315a2dec985cdefb8d8e62ab618e896ec4` |
| [XSum checkpoint](https://huggingface.co/embedded-language-flows/ELF-B-xsum-torch) | `checkpoint_39800`；revision `aca4abd1c7ff5f841f0e2a3ec8a18b0a927927f1` |
| T5 | `google-t5/t5-small`，版本见 `artifacts/elf_lock.json` |
| 评测模型 | GPT-2 Large 和 GPT-2 tokenizer，版本与现有 `artifacts/data.yaml` 一致 |
| WMT14/XSum | 作者发布的 validation Arrow + 原始公开 validation/test Parquet，均锁 revision |

下载脚本只下载上述模型及 evaluation splits。**OWT 无条件采样不需要重新下载 OWT 训练语料；本轮也不需要下载 WMT14/XSum 训练集。** 前缀实验复用服务器已准备好的 OWT held-out prompts。

论文 Appendix D.2：OWT 用 logit-normal 时间网格、SC-CFG=3；8/16 步 SDE gamma=2，32 步 gamma=1.5。发布代码的 64 步 SDE gamma=1。本文把 1/2/4 步设为 gamma=2，128/256/512/1024 步设为 gamma=1；这些是本项目扩展配置，不应写成作者报告的对应数值。条件任务默认 64 步 ODE、input CFG=2、SC-CFG=1。

**3. 服务器同步、环境和下载**

将本地代码同步到你已有的服务器仓库，不同步 checkpoint、results 或本地临时文件。以下所有命令都在服务器仓库根目录执行；不要在 Mac 上执行环境/下载命令。

```bash
cd /你的服务器路径/diffusion_baseline
export DLB_ROOT="$PWD"
export PYTHONDONTWRITEBYTECODE=1
conda create -n dlb-elf python=3.11 -y
conda activate dlb-elf
export ELF_PYTHON="$CONDA_PREFIX/bin/python"

python scripts/prepare_elf.py source
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r envs/elf-requirements.txt
python -m pip check
python -m unittest discover -s tests -p 'test_elf_standalone.py' -v

python scripts/prepare_elf.py assets --download-workers 1 --download-attempts 6 &&
python scripts/prepare_elf.py data
mkdir -p results/elf/environment
python -m pip freeze > results/elf/environment/pip-freeze.txt
nvidia-smi -q > results/elf/environment/nvidia-smi.txt
```

建议先用 H200 或支持 bf16 的 L40/L40S 单卡，generation batch=8，timing batch=1。驱动需能运行 cu124；以服务器实际 CUDA smoke 为准，不能只根据系统安装的 toolkit 版本判断。显存不足可把 generation/evaluation batch 从 8 调到 4 或 1，并记录这个变化。不要因为 OOM 改变长度、精度或采样步数后仍沿用原结果名。

使用单独的 `dlb-elf` 环境是因为官方 PyTorch 分支要求 `transformers<4.45`，不应直接复用现有 FLM/LangFlow 环境。独立 inference runner 不构建 optimizer，不需要安装 Muon 或登录 WandB。它直接 strict-load checkpoint 的 EMA 权重；T5 与 ELF 都固定为 eval 模式。

下载后生成 `data/elf/assets.json`，包含实际文件 SHA256。脚本在离线运行前校验使用到的文件；源码必须位于锁定提交且没有改动。不要编辑 `upstreams/elf`。所有新增配置都通过 wrapper 参数传入。

`prepare_elf.py data` 生成六个带 manifest 的 JSONL：两份作者 validation，以及 WMT14/XSum 各自原始 validation/test。每行保留原始 source/reference 和稳定 id。作者 Arrow 的条件 token IDs 会额外保留，以便做发布配置校验；若原始文本列不存在，脚本明确报错，不用截断 token 解码结果冒充 reference。

**第 3 步恢复：HF 429 / assets.json 不存在（2026-09-11）**

已遇到的服务器日志是：源码 checkout、PyTorch/依赖安装、`pip check` 和 7 个原有 CPU 测试全部成功；第一次查询 ELF OWT 模型 revision 时，HF 返回 `429 Too Many Requests`，详细信息为 `maximum time in concurrency queue reached`。这是下载端的限流/并发排队失败，不能据此判断模型 revision 不存在，也不是 CUDA 安装错误。

Hub 0.36 将这个 HTTP 错误包装成 `LocalEntryNotFoundError`。因为所有资源成功下载后才发布 `data/elf/assets.json`，随后单独执行 `data` 会再次报清单缺失。这是同一次下载失败的后果。旧测试里打印的 `primary timing requires --batch-size 1` 等 argparse error 是预期的拒绝测试，末尾 `OK` 表示测试通过；新版已捕获这些预期输出，减少误解。

同步更新后的 `scripts/prepare_elf.py`、`scripts/elf_common.py` 和测试文件后，保留已建环境、源码和下载缓存，从下载阶段继续：

```bash
cd ~/diffusion_baseline
conda activate dlb-elf
export DLB_ROOT="$PWD"
export ELF_PYTHON="$CONDA_PREFIX/bin/python"
export PYTHONDONTWRITEBYTECODE=1

# 可选但建议：尚未登录 HF 时，交互式输入自己的只读 token。
# 已登录可跳过；不要把 token 写进脚本、日志或聊天。
hf auth login

python -m unittest discover -s tests -p 'test_elf_standalone.py' -v &&
python scripts/prepare_elf.py assets --download-workers 1 --download-attempts 6 &&
python scripts/prepare_elf.py data
```

新脚本默认单文件下载 worker；遇到 429/500/502/503/504，连同嵌套在 cache exception 内的错误，最多尝试 6 次。默认间隔为 60/120/240/300/300 秒；响应给出更长的 `Retry-After` 或 `RateLimit` 重置时间时尊重该时间。认证失败、revision 不存在、磁盘错误直接退出，不会无限重试。减少文件并发不能直接消除这次 revision API 的排队问题，退避重试才是对该错误的恢复措施。

出现 `OK: all ... ELF assets verified; wrote data/elf/assets.json` 后才会进入 `data`。如果重试仍耗尽，稍后重跑同一条下载命令，或检查服务器共享出口/代理是否存在其他大量 HF 请求。不要并行启动多个相同下载脚本；不必删除缓存、手工创建 assets.json、重建 Conda 环境或重装 PyTorch。HF 登录可以避免未认证请求共享 IP 配额，但不保证解决服务端拥塞。[HF 官方限流说明](https://huggingface.co/docs/hub/rate-limits)

**第 3 步恢复：`Feature type 'List' not found`（2026-09-11）**

若日志已出现 `OK: all 10 ELF assets verified; wrote data/elf/assets.json`，说明下载阶段已成功。之后 `datasets.load_from_disk` 报 `Feature type 'List' not found`，是作者 Arrow 数据的 HF Features 元数据包含 `_type: List`，而固定的 `datasets==3.6.0` 尚不支持该类型。此时无需再次下载资源。

导出脚本现改为使用 PyArrow IPC 读取官方 Arrow 分片，按 `state.json` 中的分片顺序保留 source、完整 reference 和条件 token IDs；原始 validation/test Parquet 也直接用 PyArrow 读取。只跳过 HF Features 的版本相关反序列化，不改动任何 snapshot 文件、token 内容或下载清单，原有 SHA256 校验仍执行。[Arrow IPC 官方文档](https://arrow.apache.org/docs/python/ipc.html)

在服务器拉取修复后，只重跑数据导出：

```bash
cd ~/diffusion_baseline
git pull --ff-only origin main
conda activate dlb-elf
export DLB_ROOT="$PWD"
export ELF_PYTHON="$CONDA_PREFIX/bin/python"
export PYTHONDONTWRITEBYTECODE=1

python -m unittest discover -s tests -p 'test_elf_standalone.py' -v &&
python scripts/prepare_elf.py data
```

预期看到 6 条 `OK: exported ... rows to ...jsonl`，分别对应两个任务各自的 official-validation、validation、test。Arrow 已作为现有 datasets 的依赖安装，无需升级 datasets、transformers 或重装环境；也不要手工将 snapshot 的 `List` 改成 `Sequence`，否则会破坏下载校验。导出全部成功后再进入第 4 步。

**4. 先做服务器 smoke 和公开 checkpoint 校验**

选择一个 GPU。生成和 timing 都会先执行一个独立、不计时的检查：验证每个模型输入数值有限、观察实际前向次数，条件任务还检查每次 forward 的前缀 latent 是否正确固定。该检查后重置 RNG，不改变正式样本的随机种子。

```bash
# 先按单 H200 清单确认卡可用并设置 CUDA_VISIBLE_DEVICES；不要默认占用物理 GPU 0。
ELF_STEPS="1 32" bash scripts/run_elf_suite.sh smoke owt
ELF_STEPS="1 64" ELF_SPLIT=official-validation bash scripts/run_elf_suite.sh smoke wmt14
ELF_STEPS="1 64" ELF_SPLIT=official-validation bash scripts/run_elf_suite.sh smoke xsum

# 1000 样本的公开设置校验，放在独立目录。
export ELF_RESULTS="$DLB_ROOT/results/elf-sanity"
ELF_COUNT=1000 ELF_STEPS=32 bash scripts/run_elf_suite.sh generate owt
ELF_STEPS=32 bash scripts/run_elf_suite.sh evaluate owt
ELF_COUNT=1000 ELF_STEPS=64 ELF_SPLIT=official-validation bash scripts/run_elf_suite.sh generate wmt14
ELF_STEPS=64 ELF_SPLIT=official-validation bash scripts/run_elf_suite.sh evaluate wmt14
ELF_COUNT=1000 ELF_STEPS=64 ELF_SPLIT=official-validation bash scripts/run_elf_suite.sh generate xsum
ELF_STEPS=64 ELF_SPLIT=official-validation bash scripts/run_elf_suite.sh evaluate xsum
unset ELF_RESULTS
```

论文参考点：OWT 32 步 PPL≈24.1、entropy≈5.15；test 的 WMT14 BLEU=26.4，XSum R1/R2/RL=36.0/12.2/27.8。它们是定位错误的参考，不是通过测试的硬编码目标。作者 README 的 validation 参考值与 test 不同；GPU 精度、种子、样本数和本文去除 reference 编码的实现也需要在复现记录中注明。

若偏差很大，先检查 EMA、checkpoint 任务、SC-CFG/input CFG、gamma、tokenizer、EOS、样本数和 split，再跑多个 seed；不要为了追近论文数值在 test 上调参。脚本保留全部预测，空输出不会被静默删除：PPL 遇到少于两个 GPT-2 tokens 的样本会保存 `valid=false` 并报错；BLEU/ROUGE 保留空预测参与评分。

**5. OWT unconditional 正式多步曲线**

先跑核心点确认质量/耗时范围，再决定是否启动最贵的长步数格点。

```bash
ELF_STEPS="8 16 32 64" bash scripts/run_elf_suite.sh generate owt
ELF_STEPS="8 16 32 64" bash scripts/run_elf_suite.sh evaluate owt
ELF_STEPS="8 16 32 64" bash scripts/run_elf_suite.sh timing owt

ELF_STEPS="1 2 4 128 256 512 1024" bash scripts/run_elf_suite.sh generate owt
ELF_STEPS="1 2 4 128 256 512 1024" bash scripts/run_elf_suite.sh evaluate owt
ELF_STEPS="1 2 4 128 256 512 1024" bash scripts/run_elf_suite.sh timing owt
```

输出位置类似 `results/elf/quality/owt/unconditional/steps_32/seed_42/compile_0/`。每个目录只能创建一次，避免重跑覆盖结果；已存在时会拒绝。失败后检查对应目录和日志，用新 `ELF_RESULTS` 路径重新运行失败点，而不是批量覆盖所有结果。极少步 PPL 无效导致 evaluate 循环停止时，保留无效记录，单独指定剩余步数继续。

**可比性：** 原生 ELF 长度 1024 指 T5 tokens，旧 OWT 模型长度 1024 指 GPT-2 tokens。本文把生成文本重新用同一个 GPT-2 tokenizer 分词，PPL/entropy 都以最多 1024 GPT-2 tokens 评测，保存输出实际长度与截断比例。不要把 T5 native entropy 和旧 GPT-2 entropy直接相减。主表必须列出 native tokenizer、canvas、实际 GPT-2 输出长度；固定 native canvas 的 seconds/sample 是系统比较，不代表相同字节量或相同 GPT-2 tokens 的生成速度。

**2026-09-13：对用户旧 OWT baseline 表的代码核查**

“ELF 指标合理”只说明发布配置的结果量级合理，尚不证明与旧表所有设置完全相同。旧表头写“sequence length=1024、tokenizer=GPT-2”，不适用于 ELF 的生成端。可将 ELF 放入注明差异的公开权重比较表；若表头声称所有方法固定生成 1024 GPT-2 tokens，则必须修改表头/增加 tokenizer 与输出长度列，或建立另外的统一协议结果表。

| 项目 | 当前旧 baseline 路径 | 当前 ELF 路径 | 判断 |
|---|---|---|---|
| 生成任务 | OWT 无条件生成 | OWT 无条件生成，不输入 reference/prompt | 任务类型一致 |
| 样本数、seed | 正式矩阵 1024、42 | 正式 1024、42 | 一致；不同采样器/batch 的随机数流不因此相同 |
| PPL scorer/tokenizer | 锁定 GPT-2 Large / GPT-2 | 相同锁定版本 | 一致 |
| PPL 实现 | `compute_gen_ppl`，总 NLL/有效 next-token 数后取 exp，上限1024、右截断 | 同一函数和默认上限 | 评分函数一致，不代表输入文本处理一致 |
| 生成长度 | 1024 GPT-2 tokens | 1024 T5 tokens | 不一致；统一 scorer 不能将生成长度自动变成相同 |
| 解码 | FLM/Duo/MDLM 共用 capture 直接 `batch_decode(result)`，没有统一的首 EOS 截断；LangFlow wrapper 使用 `skip_special_tokens=True` | 首 EOS 起清除，再 `skip_special_tokens=True` | 原有不同方法之间也需逐项核实，不把旧表视为已统一 EOS |
| entropy | 对保存的生成 token IDs 求逐样本自然对数 unigram 熵；OWT 不排除任何 ID，保留 BOS/EOS | 对处理后的文本重新 GPT-2 分词，取前1024，逐样本求同一熵公式 | 单位和公式一致，token 范围/特殊 token 策略不同 |
| sampler | FLM Euler、gamma=0；Duo ancestral；MDLM DDPM + noise removal 等各自配置 | SDE、logit-normal、SC-CFG=3；8/16 gamma=2，32=1.5，64=1 | 各自采样配置比较，不是同 solver/noise/guidance 的消融 |
| 计算预算 | 同名 steps 不保证相同 forward 次数/单次成本 | N 步还含一次最终神经网络解码，共 N+1 次 ELF forward | 不能从步数直接推导加速比 |
| 训练条件 | 各发布 checkpoint / 本地复现 checkpoint | ELF-B OWT 公布权重，T5 表征配置 | 没有控制统一训练预算与表示方式 |

依据：仓库 `evaluation/evaluate.py`、`evaluation/generative_perplexity.py`、`src/dlb/adapters/capture.py`、`adapters/sample_langflow.py`、各模型 adapter，以及 `scripts/run_elf.py` / `scripts/evaluate_elf.py`。它们证明当前代码行为；截图中每组数字具体由哪个历史版本和 checkpoint 产生，仍须其 manifest 才能确认。

数值本身并非明显异常：[ELF 官方发布说明](https://github.com/lillian039/ELF/blob/b29d8833609e9ab7f67cd9da39435ac5cea04837/README.md)给出的 ELF-B 32 步 SDE 参考为 24.1/5.15，本次为 24.3517/5.1651。它支持“发布配置复现量级合理”，不能据此消除上述可比性差异。

对差距的解释应区分证据与假设：

- 旧表 FLM 的少步数点（如2步 PPL=7.74、entropy=0.89）说明很低的 PPL 可以同时伴随高度集中的词频，不能仅按 PPL 排质量。当前 ELF entropy≈5 不属于这种极端低熵表现，但也不是语义多样性或覆盖度的充分证明。
- ELF 32 步 entropy≈5.165，低于旧表32步的 FLM≈5.72、Duo≈5.57、MDLM≈5.69。引导采样与分布集中可能贡献较低 PPL，需要匹配评测口径和引导消融量化，不能宣称已经证明各因素占比。
- 8 步 ELF≈70.95/5.25，而截图 FLM 粗体≈449.15/5.21，unigram 熵接近时 PPL 仍差很多，因此也不能把全部提升归结为 entropy 下降。方法/权重/采样器的差异可能带来实际少步数优势，仍需统一评测核验。
- 原先32步 sanity 的 GPT-2 长度均值约946、最短828、超1024比例0.002；该日志不支持“全靠生成几个简单词获得低 PPL”。这是1000条 sanity 的长度记录，不能替代四个正式格子的长度统计；它也不能量化长度处理对 PPL 差距的贡献。
- 截图同一格同时有两组数值（如 MDLM 1024步 42.36/5.30 与105.15/5.63），应明确论文/复现/不同配置的来源后选择比较对象。ELF64步不能放入旧表的1024步列。

下一步先从现有 artifacts 做统一文本处理的重评分：明确 BOS、首个非开头 EOS、特殊 token 和 GPT-2 长度上限，保留短/空输出与长度统计，不按得分筛掉样本；对各方法采用同一套函数，将新分数另存。与旧 native-token entropy 并行保留以便追溯。固定长度对照可另做公共 GPT-2 前缀窗口，但不足窗口的样本必须报告，不能默默删除；这属于新的评分协议，不覆盖现有分数。SC-CFG=1 与3的 ELF 32步对照可用于估计引导影响，不能仅凭原生不同方法的分数差归因。当前未实施或声称已得到这些重评分/消融结果；timing 仍暂缓。

ELF 原生采用首个 EOS 截断；旧 baseline 若使用固定完整 canvas，需另外统一 EOS/解码/截断策略后重评估文本，才可作严格 matched-quality 比较。至少画出 PPL–entropy–time 三者的关系，不能只看 PPL 更低就认定质量更好。

**6. OWT conditional：沿用提示文本的零样本扩展**

这里不是 WMT14/XSum 的条件训练复现。OWT 权重只接受零样本 clean-prefix projection，结果标记为 `zero_shot_ood_native_projection`。

先使用旧 benchmark 的环境验证现有 OWT prompts；若尚未生成，按旧数据流程完成 OWT 预处理，再执行：

```bash
# 在原先安装了 dlb/data 依赖的 bootstrap 环境中；已有 prompts 也会核验。
python scripts/build_conditional_prompts.py --root "$DLB_ROOT" --dataset owt
python scripts/verify_conditional_prompts.py --root "$DLB_ROOT" --dataset owt

conda activate dlb-elf
export ELF_PYTHON="$CONDA_PREFIX/bin/python"
python scripts/prepare_elf.py prefix
ELF_STEPS="1 32" bash scripts/run_elf_suite.sh smoke owt-prefix
ELF_STEPS="8 16 32 64" bash scripts/run_elf_suite.sh generate owt-prefix
ELF_STEPS="8 16 32 64" bash scripts/run_elf_suite.sh evaluate owt-prefix
ELF_STEPS="8 16 32 64" bash scripts/run_elf_suite.sh timing owt-prefix
```

需要完整曲线时，再跑其余 1/2/4/128/256/512/1024 步，命令格式相同。MAUVE 可在主结果生成后单独补充：

```bash
python scripts/evaluate_elf.py \
  --run results/elf/quality/owt-prefix/c64-text-t5/steps_32/seed_42/compile_0 \
  --mauve
```

`c64_text_t5_v1` 的 source/reference 来自同一批旧 prompts，但只共享**文本**：不把 GPT-2 IDs 送入 T5，不用旧 schema 假称前缀是 64 个 T5 tokens。ELF 的 clean source embeddings 每步固定，最终 prefix IDs 也恢复为观测值；输出保留 suffix 和原始 source。编码器只接收 source，reference 不进入模型，避免 contextual encoder 泄漏目标文本。

质量评测取生成 suffix 的前 64 个 GPT-2 tokens；不足 64 的样本保留实际长度并报告比例。CPPL 将 prefix/suffix 分开分词再拼接，prefix targets 全部 mask；同样计算 reference CPPL。entropy 报 completion-0 和全部 completions 两项；grouped Self-BLEU 只用前 256 prompts 的五个续写。

它不是原 `c64_zs_v1` 的直接新增一行：tokenizer、native canvas 和边界分词都不同。若要与已有方法严格共表，需要把已有方法的预测也转成相同 source/suffix 文本，使用完全相同的 `c64_text_t5_v1` 质量评测口径重算；不要拿旧 CPPL CSV 原样拼接。本文的汇总不会自动混入旧表。

**7. WMT14/XSum：验证集扫步数，测试集冻结评测**

先用原始公开 validation 全集建立新的可比基线。这与作者发布的 pretokenized validation sanity check 分目录保存。默认 `ELF_COUNT=0` 表示完整输入，source 按模型上限截断，reference 使用完整原文。

```bash
ELF_STEPS="1 2 4 8 16 32 64" ELF_SPLIT=validation bash scripts/run_elf_suite.sh generate wmt14
ELF_STEPS="1 2 4 8 16 32 64" ELF_SPLIT=validation bash scripts/run_elf_suite.sh evaluate wmt14
ELF_STEPS="1 2 4 8 16 32 64" ELF_SPLIT=validation bash scripts/run_elf_suite.sh timing wmt14

ELF_STEPS="1 2 4 8 16 32 64" ELF_SPLIT=validation bash scripts/run_elf_suite.sh generate xsum
ELF_STEPS="1 2 4 8 16 32 64" ELF_SPLIT=validation bash scripts/run_elf_suite.sh evaluate xsum
ELF_STEPS="1 2 4 8 16 32 64" ELF_SPLIT=validation bash scripts/run_elf_suite.sh timing xsum
```

BLEU 使用官方实现的 lowercase=true、effective_order=true，保存 SacreBLEU signature。ROUGE 使用 stemmer=true、逐样本 F1 的算术平均，按百分制报告，SEM 不是跨随机种子的标准差。

先固定论文标准 64 步作为主 baseline；从 validation 曲线中再选择 1–2 个质量接近但更快的配置。选定后可用 `run_elf.py` 明确传 `--steps/--cfg/--sc-cfg/--gamma/--sampler`，并用新目录名保存。冻结配置后再跑 test；下面是**预先固定 64 步**的命令，不代表已经从 validation 找到了其他最优点：

```bash
ELF_STEPS=64 ELF_SPLIT=test bash scripts/run_elf_suite.sh generate wmt14
ELF_STEPS=64 ELF_SPLIT=test bash scripts/run_elf_suite.sh evaluate wmt14
ELF_STEPS=64 ELF_SPLIT=test bash scripts/run_elf_suite.sh timing wmt14
ELF_STEPS=64 ELF_SPLIT=test bash scripts/run_elf_suite.sh generate xsum
ELF_STEPS=64 ELF_SPLIT=test bash scripts/run_elf_suite.sh evaluate xsum
ELF_STEPS=64 ELF_SPLIT=test bash scripts/run_elf_suite.sh timing xsum
```

公开原始 test 重建不等于已经证实作者 Table 1 的逐样本预处理完全一致。若论文数值无法对齐，需要核对作者 test 版本、过滤、source tokenization 和 reference 处理；未经核对，表述为“官方权重在公开 test 上按本协议重新评测”，不要声称严格复现 Table 1。

**8. timing 的主口径与公平比较**

官方 `generation.py` 的 wall-clock 日志将去噪和 final decode 分开，缺少可靠的首尾 CUDA 同步，且可能混入首次编译。不要用它的 `Generation: ...` 除以样本数作为论文主耗时。

本文复用当前仓库 `dlb.timing.benchmark`：单 GPU、batch=1、5 次 warmup、32 次测量、每次首尾 `torch.cuda.synchronize()` + `perf_counter()`，保存所有原始 durations、均值/中位数/标准差、GPU/驱动/torch/CUDA 以及峰值显存。

- `sampler` / `sampler_cached_condition` 包含噪声与时间网格准备、完整去噪、最后的神经网络解码、argmax、条件投影以及必要张量操作；排除模型加载、CPU tokenizer、文件 IO 和质量评测。
- `encoder_plus_sampler` 还包含每次重新执行 T5 条件编码。源 token tensors 已在 GPU 上；这是包含条件编码的模型推理耗时，**不是包含 CPU tokenizer/H2D/网络的服务端端到端延迟**。
- 最后的 neural decode 必须计时。当前发布版 OWT SC-CFG 为训练时实现，N 步实际 ELF forward 为 **N+1**；WMT14/XSum input CFG=2，实际为 **2N+1**，另外有一次 T5 encoder。32 步 OWT=33 次，64 步条件生成=129 次。脚本在 preflight 实测确认这些次数。
- `compile_0` 为默认 eager，`compile_1` 单独报告。参数/latent fp32、官方 bf16 autocast 及 fp32 heads 都写入 metadata。主表可与旧 `precision=author` 比较，但需注明各作者精度不同；若声称算法本身更快，应补同精度/同编译模式的比较。

可以补 compiled 版本，先检查 compiled 与 eager 的质量差异，再比较排除首次编译的稳态耗时：

```bash
ELF_COMPILE=1 ELF_STEPS="16 32 64" bash scripts/run_elf_suite.sh generate owt
ELF_COMPILE=1 ELF_STEPS="16 32 64" bash scripts/run_elf_suite.sh evaluate owt
ELF_COMPILE=1 ELF_STEPS="16 32 64" bash scripts/run_elf_suite.sh timing owt
```

条件 timing 默认 prompt 0 是为了对齐旧 benchmark 的微基准。主结论再补 0、32、64、128 四个固定输入的差异；尤其 XSum source 长度不同，不应只靠一个短输入判断真实速度：

```bash
for index in 32 64 128; do
  ELF_TIMING_PROMPT="$index" ELF_STEPS=64 ELF_SPLIT=test bash scripts/run_elf_suite.sh timing xsum
done
```

同一张卡上 timing 时不要并行训练/生成/评测；H200 和 L40 的耗时不可直接比。四卡可将独立任务分卡，例：GPU0 OWT、GPU1 WMT14、GPU2 XSum、GPU3 前缀续写。使用 `CUDA_VISIBLE_DEVICES=N` 在独立终端运行，每张卡一个进程。当前脚本不是 torchrun/DDP 入口；官方 conditional evaluator 也没有在各 rank 正确划分 dataloader，不建议直接 NGPU=4 做条件主实验。

先测 8/32/64 步 batch=1，得到每样本耗时，再估预算：timing 每个点约 `(5+32)×latency`，条件两种口径分别测；另有加载和 preflight 开销。generation 实际预算使用 smoke/小批次 wall time 按 batch 放大，不能拿 batch=1 的延迟当成 batch=8 的实测吞吐。不要在尚未测量前承诺具体小时数。

**9. 你的方法后续怎么比：固定质量容差，比较最小耗时**

这轮先保留 ELF 的标准 64 步条件结果和完整 validation 曲线。你的方法以后需要对应 WMT14/XSum 条件 checkpoint，或在同一 ELF checkpoint 上实现采样加速；仅凭 OWT unconditional checkpoint 不应宣称复现机器翻译/摘要任务。

建议在看 test 前预先确定以下**研究目标容差**，它们不是论文给定标准：

| 任务 | 建议的 matched-quality 约束 | 优化目标 |
|---|---|---|
| OWT | PPL 相对差异不超过 5%，entropy 下降不超过 0.05 nats，同时检查输出长度和重复度 | 最小 seconds/sample |
| WMT14 | BLEU 至少达到 ELF 基线减 0.5 分 | 最小 encoder+sampler latency |
| XSum | R1/R2/RL 分别不低于 ELF 减 0.5/0.3/0.5 分 | 最小 encoder+sampler latency |

对每种方法都在 validation 上选满足容差的最快配置，再在同一 test/设备/精度/编译口径上评测。不是把自己方法最优点与 ELF 故意选差的点相比。报告 `speedup = ELF latency / your latency`，同时给出 quality 差值、NFE、checkpoint 参数量、source/target 长度和原始 timing 分布。没有实测前不能保证你的方法会快。

对最终关键点补 seed=42、43、44，每个 seed 用相同 inputs、配置、batch 和独立输出目录：

```bash
for seed in 43 44; do
  ELF_SEED="$seed" ELF_STEPS=64 ELF_SPLIT=test bash scripts/run_elf_suite.sh generate wmt14
  ELF_SEED="$seed" ELF_STEPS=64 ELF_SPLIT=test bash scripts/run_elf_suite.sh evaluate wmt14
  ELF_SEED="$seed" ELF_STEPS=64 ELF_SPLIT=test bash scripts/run_elf_suite.sh timing wmt14
done
```

XSum 同样执行；最终可做 paired bootstrap CI。这个 CI/跨种子汇总尚未由当前脚本自动计算，不能把 ROUGE 样本间 SEM 当作速度显著性或跨种子置信区间。

**10. 结果检查与汇总**

```bash
python scripts/summarize_elf.py --results results/elf --output results/elf/summary.csv
```

每个质量目录应有 `request.json`、`samples.jsonl`、`generation.json`、`metrics.json`；每个 timing 目录应有 `request.json` 和 `timing.json`。samples 保留原生 T5 suffix IDs、生成文本、source/reference、prompt/completion IDs；metrics 引用样本和 generation manifest 的 SHA256。timing 保留 32 次原始观测值。

`summary.csv` 是逐 run 清单，质量与 timing 为分开的记录，会显式标出 incomplete/quality_invalid/timing_only；它不是自动证明所有 grid 完成的出版表。合并两类结果前，核对 task、split/input hash、checkpoint、steps、sampler、CFG/SC-CFG/gamma、seed、precision、compile、GPU 和长度一致。现有严格 aggregate 不接受这个独立格式，这是有意保留的边界。

未来要正式注册 ELF，可在经过服务器 smoke 后再增加 `elf` adapter、T5 tokenizer/dataset contract、source/checkpoint registry 和新矩阵计数测试。此次不要先改旧 tokenizer 合约、硬凑旧 schema 或自动重跑现有全部方法。

**来源与核查依据**

- 用户提供的 ELF.pdf：第 6 页任务定义与模型规模，第 9 页 Table 1，第 19 页采样，第 24 页 D.2 参数。
- [官方 PyTorch README](https://github.com/lillian039/ELF/tree/pytorch_elf)：checkpoint 清单、GPU 精度设置、validation/test 区别。
- [锁定采样代码](https://github.com/lillian039/ELF/blob/b29d8833609e9ab7f67cd9da39435ac5cea04837/src/utils/sampling_utils.py)、[生成与解码](https://github.com/lillian039/ELF/blob/b29d8833609e9ab7f67cd9da39435ac5cea04837/src/utils/generation_utils.py)、[官方指标](https://github.com/lillian039/ELF/blob/b29d8833609e9ab7f67cd9da39435ac5cea04837/src/utils/metrics_utils.py)。
- [WMT14 原始数据](https://huggingface.co/datasets/wmt/wmt14)、[XSum 原始数据](https://huggingface.co/datasets/EdinburghNLP/xsum)。具体提交及下载文件见 `artifacts/elf_lock.json`。

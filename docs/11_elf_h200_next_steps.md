# DS210029：数据准备完成后，用单张 H200 跑 ELF

**最新状态：已确认原始条件输入缺少 EOS 并修复。请优先按 [文档 13 的恢复步骤](13_elf_eos_fix.md) 继续 WMT14/XSum 的 validation/test，使用新的 `elf-h200-gpu2-eos-v2` 结果根目录。** OWT 与前缀步骤仍可使用；不要整体重跑已经完成的格子。文档 12 是诊断期间的临时安排，现已被替代。

这份步骤从你 **2026-09-11 已完成数据导出** 的状态开始。按 A → B → C → D → E → F → G → H 顺序执行。所有命令都在服务器 `~/diffusion_baseline`；不再执行 Conda 创建、pip 安装、源码下载或 `prepare_elf.py assets/data`。模型推理均使用同一张 H200、单进程、generation batch=8；正式 timing 固定 batch=1。

**2026-09-12 sanity 日志更新：C/D 已成功完成，不再执行。** 先阅读 D1 的指标判读；如果换了 shell，按 B 恢复环境，然后执行 E。F 尚未准备时可以先执行 G 的前三个任务；前缀 timing 已单列为可选后续命令。

**A. 你已经完成什么，还需要跑什么**

已完成：`dlb-elf` 环境、锁定 ELF 源码、10 项模型/数据资源下载及校验、以下 6 个 JSONL 导出。

| 数据文件 | 行数 | 用途 |
|---|---:|---|
| wmt14-official-validation | 3000 | 作者预处理版本的 sanity check |
| wmt14-validation | 3000 | 扫步数、选择配置 |
| wmt14-test | 3003 | 冻结配置后的正式测试 |
| xsum-official-validation | 11332 | 作者预处理版本的 sanity check |
| xsum-validation | 11332 | 扫步数、选择配置 |
| xsum-test | 11334 | 冻结配置后的正式测试 |

最新日志还确认完成：物理 GPU 2 的 CUDA bf16 检查、OWT/WMT14/XSum smoke、三项 1000 样本 sanity 生成和质量评测。尚未收到正式全集质量、timing、OWT 前缀续写 prompts 的结果。OWT unconditional 无需下载训练集；OWT 前缀续写需要服务器上原 benchmark 的 held-out prompts，放在 F 单独准备。

本轮实验清单如下。先完成“核心”，再补扩展；不要上来直接运行最贵的 1024 步。

| 阶段 | 实验 | 步数 | 样本数 | 目的 |
|---|---|---|---:|---|
| C | OWT/WMT14/XSum smoke | OWT 1/32；另两项 1/64 | 每格 2 | 确认加载、采样、条件编码能运行 |
| D | 官方配置 sanity | OWT 32；另两项 64 | 每格 1000 | 先检查质量大致合理 |
| E1 | OWT unconditional 核心 | 8/16/32/64 | 每格 1024 | PPL、entropy 曲线 |
| E2 | WMT14/XSum validation 核心 | 8/16/32/64 | 全集 3000/11332 | BLEU/ROUGE 随步数变化 |
| F | OWT 零样本前缀续写核心 | 8/16/32/64 | 每格 2048 个续写 | CPPL、entropy、Self-BLEU |
| G | 上述核心配置 timing | 相同步数 | 每格 5 warmup+32 repeats | 同一 H200 的独立延迟 |
| H1 | WMT14/XSum test 标准点 | 64 | 全集 3003/11334 | 正式 baseline 质量与 timing |
| H2 | 扩展曲线 | OWT/前缀 1/2/4/128/256/512/1024；条件任务 1/2/4 | 同正式设置 | 补齐此前要求的 grid |

本轮无需训练你的方法；先保存 ELF 基线。WMT14 是德译英，XSum 是摘要。

**B. 固定使用物理 GPU 2（2026-09-12 更新）**

你已确认当前空闲的是 **2 号 H200 NVL**，本轮所有实验固定使用物理 GPU 2。此前 9 月 11 日快照里的占用情况已过时，不再按物理 GPU 3 安排。正式 timing 期间仍保持该卡独占，不与其他生成、评测或训练进程并行。

先拉取文档更新，并记录开始前的 2 号卡状态：

```bash
cd ~/diffusion_baseline
git pull --ff-only origin main
nvidia-smi -i 2
```

下面通过物理 GPU 2 的 UUID 绑定进程，所有后续命令自动沿用该卡。本轮结果使用 `results/elf-h200-gpu2`，先前目录保持原样；无需重新下载或预处理。

长任务建议放在 tmux 中：若服务器已安装 tmux，先 `tmux new -s elf-h200`，然后在其中运行后续命令；离开按 Ctrl+B 再按 D，回来执行 `tmux attach -t elf-h200`。没有 tmux 时可先在当前 SSH 终端做 smoke，长任务前再安排不会断线的运行方式。

```bash
cd ~/diffusion_baseline
conda activate dlb-elf
export DLB_ROOT="$PWD"
export ELF_PYTHON="$CONDA_PREFIX/bin/python"
export PYTHONDONTWRITEBYTECODE=1

# 这里的 2 指 nvidia-smi 显示的物理卡号。
export ELF_PHYSICAL_GPU=2
export ELF_GPU_UUID="$(nvidia-smi -i "$ELF_PHYSICAL_GPU" --query-gpu=uuid --format=csv,noheader)"
export CUDA_VISIBLE_DEVICES="$ELF_GPU_UUID"

# 使用新的固定目录，避免与之前尝试过的结果混在一起。
export ELF_RESULTS="$DLB_ROOT/results/elf-h200-gpu2"
export ELF_BATCH_SIZE=8
export ELF_EVAL_BATCH_SIZE=8
export ELF_SEED=42
export ELF_COMPILE=0
unset ELF_STEPS ELF_COUNT ELF_SPLIT ELF_TIMING_PROMPT
mkdir -p "$ELF_RESULTS/logs" "$ELF_RESULTS/environment"
nvidia-smi -i "$ELF_GPU_UUID" > "$ELF_RESULTS/environment/gpu-before.txt"
printf '%s\n' "physical_gpu=$ELF_PHYSICAL_GPU" "uuid=$ELF_GPU_UUID" > "$ELF_RESULTS/environment/gpu-selection.txt"
python -m pip freeze > "$ELF_RESULTS/environment/pip-freeze.txt"

# 日志保存在文件中；pipefail 让 tee 不掩盖实验失败。
set -o pipefail
runlog() {
  local label="$1"
  shift
  "$@" 2>&1 | tee "$ELF_RESULTS/logs/$label.log"
}
```

CUDA 接受 GPU UUID 作为 `CUDA_VISIBLE_DEVICES`；这种绑定避免把物理卡号与框架枚举顺序混淆。进程只见一张卡，所以 Python 显示 `cuda:0` 是正常的，它对应刚选定的物理 H200，不是机器的物理 GPU 0。[NVIDIA 环境变量说明](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)

进行一次小型 CUDA 检查：

```bash
python - <<'PY'
import os, torch
print('selected UUID:', os.environ['CUDA_VISIBLE_DEVICES'])
print('torch:', torch.__version__, 'wheel CUDA:', torch.version.cuda)
print('visible GPUs:', torch.cuda.device_count())
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
print('logical cuda:0:', torch.cuda.get_device_name(0))
assert 'H200' in torch.cuda.get_device_name(0)
assert torch.cuda.is_bf16_supported()
x = torch.randn(256, 256, device='cuda', dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
print('CUDA bf16 check passed:', tuple(y.shape))
PY
```

你的驱动为 575.57.08；保留已安装的 `torch 2.5.1+cu124` 即可，不需要为了 `nvidia-smi` 顶部显示 CUDA 12.9 改装 cu129。驱动支持版本与 wheel 内的 CUDA 运行时不是同一个字段；最终以这次实际 CUDA 检查和 smoke 为准。[NVIDIA 兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

每次更换 SSH/tmux shell，需要重新执行环境设置和 `runlog` 定义。不要在同一 GPU 同时开多个实验终端；下面的命令全部串行。

**C. 先跑 3 组 smoke，不先测正式 timing**

以下小块使用子 shell 的 `set -e`，任何一项失败都会停止这一块；不要看到报错后直接进入下一节。

```bash
(
  set -e
  runlog smoke_owt env ELF_STEPS="1 32" bash scripts/run_elf_suite.sh smoke owt
  runlog smoke_wmt14 env ELF_STEPS="1 64" ELF_SPLIT=official-validation bash scripts/run_elf_suite.sh smoke wmt14
  runlog smoke_xsum env ELF_STEPS="1 64" ELF_SPLIT=official-validation bash scripts/run_elf_suite.sh smoke xsum
)
```

成功标志：每格进度到 `2/2 samples`、无 traceback、对应目录出现 `generation.json` 和 `samples.jsonl`。例如：

```text
results/elf-h200-gpu2/smoke/owt/unconditional/steps_32/seed_42/compile_0/
results/elf-h200-gpu2/smoke/wmt14/official-validation/steps_64/seed_42/compile_0/
results/elf-h200-gpu2/smoke/xsum/official-validation/steps_64/seed_42/compile_0/
```

smoke 是功能检查，不凭两个样本判断论文质量；1 步生成很差也不等于实现错误。若失败，把对应 `logs/smoke_*.log` 的 traceback 发来。首次模型加载与文件校验期间可能暂时没有采样进度，不要因此再开第二份进程。

**D. 1000 样本 sanity：确认质量合理，再花时间扫全集**

这一节单独存到 `results/elf-h200-gpu2-sanity`，不会覆盖后面的正式 1024 样本或完整 validation。

```bash
(
  set -e
  export ELF_RESULTS="$DLB_ROOT/results/elf-h200-gpu2-sanity"
  mkdir -p "$ELF_RESULTS/logs"
  runlog owt32_generate env ELF_COUNT=1000 ELF_STEPS=32 bash scripts/run_elf_suite.sh generate owt
  runlog owt32_evaluate env ELF_STEPS=32 bash scripts/run_elf_suite.sh evaluate owt
  runlog wmt64_generate env ELF_COUNT=1000 ELF_STEPS=64 ELF_SPLIT=official-validation bash scripts/run_elf_suite.sh generate wmt14
  runlog wmt64_evaluate env ELF_STEPS=64 ELF_SPLIT=official-validation bash scripts/run_elf_suite.sh evaluate wmt14
  runlog xsum64_generate env ELF_COUNT=1000 ELF_STEPS=64 ELF_SPLIT=official-validation bash scripts/run_elf_suite.sh generate xsum
  runlog xsum64_evaluate env ELF_STEPS=64 ELF_SPLIT=official-validation bash scripts/run_elf_suite.sh evaluate xsum
)
```

`generate` 只生成并保存样本；`evaluate` 才计算指标。评测成功会打印 JSON 并保存 `metrics.json`。OWT 32 步论文参考 PPL≈24、entropy≈5.15；WMT14 BLEU 和 XSum ROUGE 的论文 test 参考分别为 26.4、36.0/12.2/27.8。这些只是量级参照，当前 sanity 使用 1000 条 official-validation，不能当作 test 全集结果。明显偏离时先检查日志与样本，不根据 test 分数反复调参。

括号结束后，`ELF_RESULTS` 自动回到 B 设置的 `results/elf-h200-gpu2`，不需要手动 unset。

**D1. 本次 sanity 日志判读：主指标合理，继续正式实验**

三组均完成 `1000/1000 samples` 并打印评测 JSON，没有 traceback。终端先后打印 OWT、WMT14、XSum；条件任务的通用评测器都会打印 BLEU 和 ROUGE，应按任务读取主指标：

| 任务 | 本次主指标 | 前文记录的论文参考 | 判读 |
|---|---|---|---|
| OWT，32 步 | PPL **24.4239**；entropy **5.1671 nats** | 24.1；5.15 | PPL 高约 1.34%，entropy 高约 0.017，量级接近 |
| WMT14，64 步 | BLEU **25.7906** | 26.4 | 低约 0.61 BLEU；当前是前 1000 条 official-validation，不是论文 test |
| XSum，64 步 | ROUGE-1/2/L **36.3126 / 12.3464 / 28.1115** | 36.0 / 12.2 / 27.8 | 量级接近；不能据此声称 test 超过论文 |

XSum 的 BLEU=8.0842 是附带指标，不是截图中 XSum 的评价列；WMT14 的 ROUGE=59.24/33.29/53.85 也不应拿去对比 XSum。OWT 的 entropy 是每条文本的经验 unigram 熵，PPL 是 GPT-2 Large 的预测困惑度，两者不是同一个概率分布，所以不要求 `entropy = ln(PPL)`。ROUGE 的 `_sem` 是此次 1000 条样本之间的标准误，不是多个随机种子的标准差，不能与论文误差条直接判定显著性。

两条 tokenizer 警告均未导致本次失败：

- `clean_up_tokenization_spaces` 是所装 Transformers 版本的未来默认值提示，本次不用升级或重装环境。
- `1037 > 1024` 来自评测脚本先对完整文本分词以统计原始长度；`above_1024_tokens_fraction=0.002` 表示 1000 条里有 2 条超过 GPT-2 的 1024 上限。实际 PPL 调用 `compute_gen_ppl` 时明确设置 `truncation=True, max_length=1024`，entropy 同样取前 1024 个 GPT-2 tokens。样本没有被丢弃，超长部分不计分，结果中已记录右截断策略。

长度还可做交叉检查：原始平均长度 946.265 对应总计 946265 tokens；减去每条不计首 token 的 1000 个目标以及截断掉的 21 tokens，得到日志中的 `valid_token_count=945244`。所有三项 `empty_fraction=0`，OWT 最短也有 828 个 GPT-2 tokens，未见空输出或极短输出退化的迹象。

**结论：日志未显示需要修复的生成或计分错误，保留已有 sanity，不改采样参数、不重跑 C/D，进入 E 的正式步数扫描。** 当前没有正式 timing，不能从 `8/1000 samples` 的进度或生成总用时推断单样本延迟。此次日志判读仅说明 sanity 合理，正式 test 和速度结论仍待测量。

**E. 核心正式质量实验：OWT → WMT14 → XSum**

先 OWT，后条件任务；同一张 H200 连续串行运行。显式固定 OWT 为 1024 条、条件任务为完整 validation，按每个步数先生成再评测，便于及时看到对应质量：

```bash
(
  set -e
  for steps in 8 16 32 64; do
    runlog "owt_s${steps}_generate" env ELF_COUNT=1024 ELF_STEPS="$steps" bash scripts/run_elf_suite.sh generate owt
    runlog "owt_s${steps}_evaluate" env ELF_STEPS="$steps" bash scripts/run_elf_suite.sh evaluate owt
  done
  for task in wmt14 xsum; do
    for steps in 8 16 32 64; do
      runlog "${task}_s${steps}_generate" env ELF_COUNT=0 ELF_STEPS="$steps" ELF_SPLIT=validation bash scripts/run_elf_suite.sh generate "$task"
      runlog "${task}_s${steps}_evaluate" env ELF_STEPS="$steps" ELF_SPLIT=validation bash scripts/run_elf_suite.sh evaluate "$task"
    done
  done
)
```

预期每个 OWT 格子 1024 样本，WMT14 格子 3000 样本，XSum 格子 11332 样本。核心质量结果共 12 格。XSum 总 canvas 为 1088 T5 tokens，完整 validation 比 1000 样本 sanity 更耗时；不要把没有新终端输出等同于卡死。

查看任意一个完成的指标文件：

```bash
cat "$ELF_RESULTS/quality/owt/unconditional/steps_32/seed_42/compile_0/metrics.json"
cat "$ELF_RESULTS/quality/wmt14/validation/steps_64/seed_42/compile_0/metrics.json"
cat "$ELF_RESULTS/quality/xsum/validation/steps_64/seed_42/compile_0/metrics.json"
```

**F. OWT 前缀续写：另外准备 prompts，再跑核心实验**

刚导出的六个数据文件不含 OWT 前缀 prompts。先检查已有文件：

```bash
ls -l data/conditional/owt-c64/prompts.jsonl data/manifests/conditional-owt-c64.json
```

若文件已存在，直接在 bootstrap 环境验证；若不存在但已有原 benchmark 的 OWT 预处理数据，先构建再验证：

```bash
conda activate dlb-bootstrap
# 仅在 prompts 尚未准备时执行这一行：
python scripts/build_conditional_prompts.py --root "$DLB_ROOT" --dataset owt
python scripts/verify_conditional_prompts.py --root "$DLB_ROOT" --dataset owt
conda activate dlb-elf
export ELF_PYTHON="$CONDA_PREFIX/bin/python"
python scripts/prepare_elf.py prefix
```

如果构建提示缺少 OWT processed 数据，先完成 [旧数据准备流程](02_datasets.md)；这不会妨碍你先完成 E/G/H 中不依赖前缀的三个任务。不要用 WMT14 或 XSum 来替代 OWT prompts。

```bash
(
  set -e
  runlog prefix_smoke env ELF_STEPS="1 32" bash scripts/run_elf_suite.sh smoke owt-prefix
  runlog prefix_core_generate env ELF_STEPS="8 16 32 64" bash scripts/run_elf_suite.sh generate owt-prefix
  runlog prefix_core_evaluate env ELF_STEPS="8 16 32 64" bash scripts/run_elf_suite.sh evaluate owt-prefix
)
```

smoke 使用 2 个 prompts，每个 5 个续写，共 10 条；正式每格 1024 个主续写，加前 256 prompts 各 4 个额外续写，共 2048 条。默认得到 CPPL、entropy、grouped Self-BLEU；MAUVE 后续可补，不是继续主流程的前提。

**G. 核心 timing：同一张 H200 独占时单独运行**

测量前确认你自己的 generation/evaluate 均已结束，并重新检查所选卡，不能用 10:47 的旧快照代替：

```bash
nvidia-smi -i "$ELF_GPU_UUID"
```

如果仍有其他计算进程，等其结束或取得独占时段。就绪后执行：

```bash
(
  set -e
  nvidia-smi -i "$ELF_GPU_UUID" > "$ELF_RESULTS/environment/gpu-before-timing.txt"
  runlog owt_core_timing env ELF_STEPS="8 16 32 64" bash scripts/run_elf_suite.sh timing owt
  runlog wmt_core_timing env ELF_STEPS="8 16 32 64" ELF_SPLIT=validation bash scripts/run_elf_suite.sh timing wmt14
  runlog xsum_core_timing env ELF_STEPS="8 16 32 64" ELF_SPLIT=validation bash scripts/run_elf_suite.sh timing xsum
  nvidia-smi -i "$ELF_GPU_UUID" > "$ELF_RESULTS/environment/gpu-after-timing.txt"
)
```

只有完成 F 的 prefix 准备后，再单独测前缀任务；同样保持 GPU 独占：

```bash
(
  set -e
  nvidia-smi -i "$ELF_GPU_UUID" > "$ELF_RESULTS/environment/gpu-before-prefix-timing.txt"
  runlog prefix_core_timing env ELF_STEPS="8 16 32 64" bash scripts/run_elf_suite.sh timing owt-prefix
  nvidia-smi -i "$ELF_GPU_UUID" > "$ELF_RESULTS/environment/gpu-after-prefix-timing.txt"
)
```

也在另一个终端观察 GPU 是否被新任务占用；前后两张快照不能证明整个计时窗口独占。若发生干扰，此次 timing 标为受干扰，之后用新结果目录重测，不能据此算加速比。

每个格子固定 batch=1、5 次 warmup、32 次重复；无需将 `ELF_BATCH_SIZE=8` 改为 1，timing 入口会强制传 1。保存位置例：

```text
results/elf-h200-gpu2/timing/owt/unconditional/steps_32/seed_42/compile_0/prompt_0/timing.json
results/elf-h200-gpu2/timing/xsum/validation/steps_64/seed_42/compile_0/prompt_0/timing.json
```

主要读取 `results.sampler.seconds_per_sample`；条件任务读取 `results.sampler_cached_condition.seconds_per_sample` 和 `results.encoder_plus_sampler.seconds_per_sample`，后者包含 T5 条件编码。它们包含最终神经网络解码，不含加载、CPU tokenizer、磁盘 IO 和指标计算。不要把 generation 的总 wall time 当作正式延迟。

**H. 正式 test、扩展曲线和多 seed**

H1：先固定论文标准 64 步条件配置，不在 test 上选择参数。使用完整 test 集：

```bash
(
  set -e
  runlog wmt_test_generate env ELF_STEPS=64 ELF_SPLIT=test bash scripts/run_elf_suite.sh generate wmt14
  runlog wmt_test_evaluate env ELF_STEPS=64 ELF_SPLIT=test bash scripts/run_elf_suite.sh evaluate wmt14
  runlog xsum_test_generate env ELF_STEPS=64 ELF_SPLIT=test bash scripts/run_elf_suite.sh generate xsum
  runlog xsum_test_evaluate env ELF_STEPS=64 ELF_SPLIT=test bash scripts/run_elf_suite.sh evaluate xsum
  # 仍须满足 G 的独占条件。
  runlog wmt_test_timing env ELF_STEPS=64 ELF_SPLIT=test bash scripts/run_elf_suite.sh timing wmt14
  runlog xsum_test_timing env ELF_STEPS=64 ELF_SPLIT=test bash scripts/run_elf_suite.sh timing xsum
)
```

WMT14 应为 3003 条，XSum 应为 11334 条。若还要报告更少步但质量接近的点，先在 validation 上选择并记录冻结配置，再指定那些步数跑 test，不自动把某个低步数点叫作最优。

H2：核心完成后补扩展，每个格子都生成、评测、计时。OWT/前缀比核心多 7 个步数，条件任务多 3 个：

```bash
(
  set -e
  for task in owt owt-prefix; do
    runlog "${task}_extra_generate" env ELF_STEPS="1 2 4 128 256 512 1024" bash scripts/run_elf_suite.sh generate "$task"
    runlog "${task}_extra_evaluate" env ELF_STEPS="1 2 4 128 256 512 1024" bash scripts/run_elf_suite.sh evaluate "$task"
    runlog "${task}_extra_timing" env ELF_STEPS="1 2 4 128 256 512 1024" bash scripts/run_elf_suite.sh timing "$task"
  done
  for task in wmt14 xsum; do
    runlog "${task}_extra_generate" env ELF_STEPS="1 2 4" ELF_SPLIT=validation bash scripts/run_elf_suite.sh generate "$task"
    runlog "${task}_extra_evaluate" env ELF_STEPS="1 2 4" ELF_SPLIT=validation bash scripts/run_elf_suite.sh evaluate "$task"
    runlog "${task}_extra_timing" env ELF_STEPS="1 2 4" ELF_SPLIT=validation bash scripts/run_elf_suite.sh timing "$task"
  done
)
```

如果 F 未就绪，先把第一行循环中的 `owt-prefix` 去掉，之后再补它。如果低步数空输出导致 PPL 无效，脚本会保留 `valid=false` 并停止；这是可能的质量结果，不能删除空样本抬高指标。确认是哪一格后，用 `ELF_STEPS` 单独指定剩余未评测格子继续；**不要把生成循环整体重跑**，因为已有输出目录会被拒绝。

H3：第一次完整流程成功后才补关键点 seed 43/44 或 compiled 版本，避免现在一次启动过多配置。比如补两种条件任务的标准点：

```bash
(
  set -e
  for seed in 43 44; do
    for task in wmt14 xsum; do
      runlog "${task}_test_seed${seed}_generate" env ELF_SEED="$seed" ELF_STEPS=64 ELF_SPLIT=test bash scripts/run_elf_suite.sh generate "$task"
      runlog "${task}_test_seed${seed}_evaluate" env ELF_SEED="$seed" ELF_STEPS=64 ELF_SPLIT=test bash scripts/run_elf_suite.sh evaluate "$task"
      runlog "${task}_test_seed${seed}_timing" env ELF_SEED="$seed" ELF_STEPS=64 ELF_SPLIT=test bash scripts/run_elf_suite.sh timing "$task"
    done
  done
)
```

**I. 结果位置、汇总与中断恢复**

```bash
python scripts/summarize_elf.py \
  --results "$ELF_RESULTS" --output "$ELF_RESULTS/summary.csv"
```

| 目录 | 内容 |
|---|---|
| `results/elf-h200-gpu2/logs/` | 每个步骤的终端日志 |
| `results/elf-h200-gpu2/smoke/` | 小样本功能检查，不能当正式质量结果 |
| `results/elf-h200-gpu2/quality/` | 正式 samples、generation manifest、metrics |
| `results/elf-h200-gpu2/timing/` | 原始重复测量、均值/中位数/标准差 |
| `results/elf-h200-gpu2/environment/` | 物理 GPU UUID 映射、环境和 GPU 快照 |
| `results/elf-h200-gpu2-sanity/` | 1000 样本预检查，与主实验分开 |

核心 E+F 的主质量格子共 16 个，相应 G timing 共 16 个；H1 另加 2 个 test 质量格子和 2 个 timing 格子。某项没跑就应留作 missing，不用别的 split 或 sanity 数据补数。summary 是逐 run 清单，质量和 timing 分开列，smoke 也可能显示 incomplete；不是完成度证明或自动排好的论文表。

失败目录不会被覆盖。若某次失败且目录已存在，先保留日志，用新的 `ELF_RESULTS` 目录（例如 `results/elf-h200-gpu2-retry1`）只重跑失败点；评测时要指向相同新目录。已经成功生成但尚未评测的点，只执行 evaluate；已经有质量结果而缺 timing，只执行 timing。不要在同一根目录里改变 seed/batch/compile 或样本数后覆写同一格。

如果 shell 断开后要恢复，先重新设置 B 中的变量和 `runlog` 函数，查询当前 GPU 占用，再从未完成的阶段继续。本次更新基于你提供的服务器终端日志，没有在服务器代跑；已确认 C/D 的运行结果，后续全集质量和实际 timing 仍待执行。

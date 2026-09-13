# 2026-09-13：ELF 已完成结果与剩余质量实验

本页是当前执行入口，取代故障期间重复跑 64 步或检查 OWT 是否完成的安排。所有实验在服务器执行；**timing 按用户要求暂缓，本页没有 timing 命令**。无需重新下载或安装；当前 EOS 修复代码可直接使用。

**已完成 12 个正式质量配置，全部报告 valid=true**

| 步数 | OWT PPL | OWT entropy (nats) | WMT14 BLEU | XSum R1 / R2 / RL |
|---:|---:|---:|---:|---|
| 8 | 70.9471 | 5.2528 | 24.1705 | 34.3705 / 11.1567 / 26.6864 |
| 16 | 32.1761 | 5.1609 | 25.3318 | 35.3485 / 11.7578 / 27.2921 |
| 32 | 24.3517 | 5.1651 | 26.1424 | 35.8402 / 12.1714 / 27.6562 |
| 64 | 19.1247 | 5.0774 | 26.6741 | 36.0270 / 12.3522 / 27.8165 |

OWT 每格 1024 条；WMT14 每格 3000 条 validation；XSum 每格 11332 条 validation。条件任务全部 empty_fraction=0。额外的 OWT 32 步、1000 条 sanity 单独保留，不算正式第 5 个配置。数字来自用户贴出的指标清单；没有在本地接触服务器结果文件。

OWT 四格在 `results/elf-h200-gpu2`，条件八格在 `results/elf-h200-gpu2-eos-v2`。因此第三处 `summarize_elf.py` 打印八条记录是正确的，只扫描了条件根目录。不要把两个目录移动或混并来凑数量。

OWT 64 步 PPL 更低，但 entropy 比 32 步低约 0.0877 nats；不能只看 PPL 判定整体最好，而且 OWT 的 gamma 随配置变化（8/16 为 2，32 为 1.5，64 为 1），这不是只改变步数的单变量消融。XSum 32→64 步的 R1/R2/RL 差为 0.1868/0.1808/0.1603；WMT14 BLEU 差为 0.5317，略超过前文建议的 0.5 BLEU 容差，不能将它表述为已满足该容差。没有 timing，不能声称任何速度倍数。

**后续顺序**

1. 补 OWT 的 1/2/4 步。
2. 固定两项条件任务的 32/64 步，跑 test 全集；64 是标准点，32 是较少步数的对比点，不声称两者质量等价。
3. 补 OWT 的 128/256/512/1024 步，完成 11 点无条件曲线。
4. 补 WMT14/XSum 的 1/2/4 步 validation，完成 7 点曲线。
5. 准备 OWT 前缀数据并跑 8/16/32/64 步；核心通过后补其余 7 个步数。

先跑这些质量实验；后续与旧 baseline 对照时统一 GPT-2 scorer/tokenizer、EOS 和截断口径，并列出实际生成长度。ELF 的 1024 个原生 T5 tokens 不等于旧模型的 1024 个 GPT-2 tokens。前缀的 `c64_text_t5_v1` 也须与其他方法使用一致的文本边界和评分规则。

**A．每个新 shell 先设置环境**

确认同一 GPU 上没有你自己仍在运行的实验，下面各块串行执行。其他人的小进程可能拖慢质量运行；正式速度测量另约独占时段。

```bash
cd ~/diffusion_baseline
conda activate dlb-elf
export DLB_ROOT="$PWD"
export ELF_PYTHON="$CONDA_PREFIX/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export ELF_PHYSICAL_GPU=2
export ELF_GPU_UUID="$(nvidia-smi -i "$ELF_PHYSICAL_GPU" --query-gpu=uuid --format=csv,noheader)"
export CUDA_VISIBLE_DEVICES="$ELF_GPU_UUID"
export ELF_RESULTS="$DLB_ROOT/results/elf-h200-gpu2-eos-v2"
export ELF_BATCH_SIZE=8 ELF_EVAL_BATCH_SIZE=8 ELF_SEED=42 ELF_COMPILE=0
unset ELF_STEPS ELF_COUNT ELF_SPLIT ELF_TIMING_PROMPT
mkdir -p "$ELF_RESULTS/logs"
set -o pipefail
runlog() {
  local label="$1"
  shift
  "$@" 2>&1 | tee "$ELF_RESULTS/logs/$label.log"
}
nvidia-smi -i "$ELF_GPU_UUID"
```

**B．补 OWT 少步数质量**

子 shell 临时切回原 OWT 根目录，结束后恢复条件根目录。每步 1024 条，batch=8，seed=42。

```bash
(
  set -e
  export ELF_RESULTS="$DLB_ROOT/results/elf-h200-gpu2"
  mkdir -p "$ELF_RESULTS/logs"
  for steps in 1 2 4; do
    runlog "owt_s${steps}_generate" env ELF_COUNT=1024 ELF_STEPS="$steps" \
      bash scripts/run_elf_suite.sh generate owt
    runlog "owt_s${steps}_evaluate" env ELF_STEPS="$steps" \
      bash scripts/run_elf_suite.sh evaluate owt
  done
)
```

低步数可能产生极短或空输出。若 evaluate 保存 `valid=false` 并提示少于两个 GPT-2 tokens，保留为该配置的退化结果，不删样本来计算更好 PPL。本块会停止；查看日志后只指定剩余步数继续，不重复已生成的格子。若是其他异常（文件损坏、OOM 等），先修复。OWT 的 1/2/4 及大于 64 步配置是项目扩展，不是作者已报告的复现点。

**C．条件任务 test：32/64 步，共 4 格**

本次在查看 test 前固定 32/64 步；不根据 test 分数反复选参数。WMT14 每格 3003 条，XSum 每格 11334 条。使用修复后的 EOS 规则、原文完整 reference；这仍是公开 test 按本协议重评，并未证明与论文全部预处理完全相同。

```bash
(
  set -e
  for task in wmt14 xsum; do
    for steps in 32 64; do
      runlog "${task}_test_s${steps}_generate" \
        env ELF_COUNT=0 ELF_STEPS="$steps" ELF_SPLIT=test \
        bash scripts/run_elf_suite.sh generate "$task"
      runlog "${task}_test_s${steps}_evaluate" \
        env ELF_STEPS="$steps" ELF_SPLIT=test \
        bash scripts/run_elf_suite.sh evaluate "$task"
    done
  done
)
```

**D．补 OWT 大步数质量**

```bash
(
  set -e
  export ELF_RESULTS="$DLB_ROOT/results/elf-h200-gpu2"
  mkdir -p "$ELF_RESULTS/logs"
  for steps in 128 256 512 1024; do
    runlog "owt_s${steps}_generate" env ELF_COUNT=1024 ELF_STEPS="$steps" \
      bash scripts/run_elf_suite.sh generate owt
    runlog "owt_s${steps}_evaluate" env ELF_STEPS="$steps" \
      bash scripts/run_elf_suite.sh evaluate owt
  done
)
```

这些配置明显更耗时。若预算有限，可以先跑 128/256，把 512/1024 留待补齐并明确标记缺失；不要用已有 64 步结果填充它们。完成后 OWT 正式曲线为 11 个步数，每个配置独立记录有效性。

**E．条件任务补 1/2/4 步 validation，共 6 格**

```bash
(
  set -e
  for task in wmt14 xsum; do
    for steps in 1 2 4; do
      runlog "${task}_val_s${steps}_generate" \
        env ELF_COUNT=0 ELF_STEPS="$steps" ELF_SPLIT=validation \
        bash scripts/run_elf_suite.sh generate "$task"
      runlog "${task}_val_s${steps}_evaluate" \
        env ELF_STEPS="$steps" ELF_SPLIT=validation \
        bash scripts/run_elf_suite.sh evaluate "$task"
    done
  done
)
```

**F．OWT 前缀续写：独立准备和执行**

先在 bootstrap 环境检查旧 benchmark prompts，只有不存在时才构建。已有 OWT processed 数据是构建前提；如果缺少则按文档 02 准备，其他实验不依赖它。

```bash
(
  set -e
  conda activate dlb-bootstrap
  if [ ! -f "$DLB_ROOT/data/conditional/owt-c64/prompts.jsonl" ]; then
    python scripts/build_conditional_prompts.py --root "$DLB_ROOT" --dataset owt
  fi
  python scripts/verify_conditional_prompts.py --root "$DLB_ROOT" --dataset owt
  conda activate dlb-elf
  export ELF_PYTHON="$CONDA_PREFIX/bin/python"
  python scripts/prepare_elf.py prefix
  runlog prefix_smoke env ELF_STEPS="1 32" bash scripts/run_elf_suite.sh smoke owt-prefix
  for steps in 8 16 32 64; do
    runlog "prefix_s${steps}_generate" env ELF_COUNT=1024 ELF_STEPS="$steps" \
      bash scripts/run_elf_suite.sh generate owt-prefix
    runlog "prefix_s${steps}_evaluate" env ELF_STEPS="$steps" \
      bash scripts/run_elf_suite.sh evaluate owt-prefix
  done
)
```

每个正式格子为 1024 个主续写，加前 256 prompts 的 4 个额外续写，共 2048 条；主指标 CPPL、entropy、grouped Self-BLEU。核心无异常后，使用同一 generate/evaluate 循环，将 steps 改为 `1 2 4 128 256 512 1024` 补完整前缀曲线。首次只跑核心，若出现无效极短输出按 B 的规则保留。

**G．分别汇总两个根目录**

```bash
python scripts/summarize_elf.py \
  --results "$DLB_ROOT/results/elf-h200-gpu2" \
  --output "$DLB_ROOT/results/elf-h200-gpu2/summary.csv"
python scripts/summarize_elf.py \
  --results "$DLB_ROOT/results/elf-h200-gpu2-eos-v2" \
  --output "$DLB_ROOT/results/elf-h200-gpu2-eos-v2/summary.csv"
```

旧根目录还可能含 EOS 修复前的 WMT14 低分结果和 smoke；论文质量表应筛选其 OWT 正式行。条件任务仅用 EOS 修复根目录，并区分 validation/test；两份 CSV 都是清单，不是自动合并好的论文表。成功格子不要重跑；已生成未评测只 evaluate，未完成目录保留并在新重试根目录处理。

# ELF 条件输入 EOS 修复与正式实验恢复

本页取代文档 12 中“原始 validation/test 暂停、先改跑作者全集”的临时安排。现有 JSONL、权重、环境无需重新准备；修复发生在 runner 的条件分词阶段，结果使用新目录 `results/elf-h200-gpu2-eos-v2`。

**诊断已经确认什么**

用户提交的服务器 audit 显示：WMT14 的 2992 条唯一文本对、XSum 的全部 11332 条，作者保存的条件 IDs 都严格等于 `raw_no_special_tokens + [EOS]`。EOS ID 为 1（`</s>`）。WMT14 剩余 8 条是重复文本对，旧诊断没有唯一配对；新诊断通过包含重复次数的完整记录比较覆盖它们，不会删除重复样本。

XSum 的 source ID **219、3834、4715、5677、8445** 是空白文本；旧路径返回 `[]`，作者路径是 `[1]`，所以原始实验在准备含第 219 行的 batch 时中断，而作者 sanity 能通过。EOS 属于输入编码格式，并非补写文章内容；这五条仍保留完整 reference 并参与评测。

WMT14 旧输入缺 EOS 的协议错误已确认，是质量下降的重要解释；是否足以恢复 BLEU，要以修正后实测为准，不能声称已经恢复 26 分。此前依据官方 JSONL loader 的无特殊 token 写法没有复现发布 Arrow 的编码，现根据实际发布数据修正。

**修复规则**

- WMT14/XSum 原始文本：完整分词后追加一个 EOS，再截到 source 上限 64/1024；不能先截断再强行补 EOS，也不能预留一个位置替换长文本的最后一个正文 token。
- 已有 `condition_input_ids`：原样使用，不再添加 EOS；已有 official-validation 生成结果不受影响。
- OWT unconditional 与 OWT 前缀续写：保留原协议。前缀续写不是已结束的 source 文本，不能套用这次 EOS 改动。
- `request.json`、`generation.json`、`timing.json` 记录 `condition_tokenization_policy=released_ids_or_raw_plus_eos_then_source_cap_v2`，汇总 CSV 也保留该字段。旧结果缺这个字段，不能仅按 task/steps 与新结果混用。

**1．同步代码并绑定物理 GPU 2**

```bash
cd ~/diffusion_baseline
git pull --ff-only origin main
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
mkdir -p "$ELF_RESULTS/logs" "$ELF_RESULTS/environment"
set -o pipefail
runlog() {
  local label="$1"
  shift
  "$@" 2>&1 | tee "$ELF_RESULTS/logs/$label.log"
}
nvidia-smi -i "$ELF_GPU_UUID" > "$ELF_RESULTS/environment/gpu-before.txt"
```

**2．CPU 检查实际修复路径，再跑 64 步 validation 全集**

更新后的诊断会保留旧 `raw` 统计用于对照，所以其中仍显示 5 个空列表是预期的。真正判定修复的是新增的 `corrected_runner_comparison_including_duplicates`：`empty_condition_ids=[]`、`ready_for_raw_validation=true`。完整 token 比较和截断后比较会分别报告，并检查顺序是否一致。它调用 runner 共用的分词函数，不是仅对 EOS 做一个假设性比较。

以下整块先做 CPU 检查，通过后才加载 GPU 模型。两项分别生成全部 3000/11332 条；参考答案不参与条件编码。

```bash
(
  set -e
  runlog input_audit_v2 python scripts/audit_elf_inputs.py \
    --output "$ELF_RESULTS/input-audit-v2.json"
  python - <<'PY'
import json, os
from pathlib import Path
r = json.loads((Path(os.environ['ELF_RESULTS']) / 'input-audit-v2.json').read_text())
assert r['schema'] == 'elf-condition-audit-v2'
for task in ('wmt14', 'xsum'):
    c = r['tasks'][task]['corrected_runner_comparison_including_duplicates']
    print(task, c)
    assert c['ready_for_raw_validation'], f'{task}: 修复后的条件与作者数据仍不一致，请保留报告'
print('条件输入核验通过，开始 64 步 validation 全集')
PY
  for task in wmt14 xsum; do
    runlog "${task}_val_s64_generate" \
      env ELF_COUNT=0 ELF_STEPS=64 ELF_SPLIT=validation \
      bash scripts/run_elf_suite.sh generate "$task"
    runlog "${task}_val_s64_evaluate" \
      env ELF_STEPS=64 ELF_SPLIT=validation \
      bash scripts/run_elf_suite.sh evaluate "$task"
  done
)
```

先看 WMT14 BLEU 和 XSum ROUGE 是否回到先前 sanity 的合理量级；全集与前 1000 条允许有抽样差异。如果仍明显异常，先提供这两项指标和 audit v2，不继续扩展扫描。无需再次跑 1000 条 sanity 或重复作者全集；如果作者全集已启动，先让它完成或正常停止，勿在同一 GPU 同时启动本节。

**3．64 步质量确认后，重跑原始 validation 的 8/16/32 步**

旧 WMT14 的 13–14 分结果使用了错误的 source 编码，不能只重算指标，必须重新生成；旧 XSum 的部分 `.tmp` 不能拼接进新结果。全部旧文件保留，不删除、不覆盖。

```bash
(
  set -e
  for task in wmt14 xsum; do
    for steps in 8 16 32; do
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

**4．单卡独占时测修复后的条件 timing**

固定 batch=1、5 次 warmup、32 次 repeats，默认第 0 条输入；不与生成/指标评测并行。旧原始输入的 timing 也使用了不同条件长度，不能替代修复后的 timing。

```bash
nvidia-smi -i "$ELF_GPU_UUID"
(
  set -e
  nvidia-smi -i "$ELF_GPU_UUID" > "$ELF_RESULTS/environment/gpu-before-timing.txt"
  for task in wmt14 xsum; do
    runlog "${task}_val_timing" \
      env ELF_STEPS="8 16 32 64" ELF_SPLIT=validation \
      bash scripts/run_elf_suite.sh timing "$task"
  done
  nvidia-smi -i "$ELF_GPU_UUID" > "$ELF_RESULTS/environment/gpu-after-timing.txt"
)
```

同时保留 `sampler_cached_condition` 和 `encoder_plus_sampler` 的 `seconds_per_sample`，后者包含 T5 条件编码；两者均不含加载、CPU tokenizer、IO 和指标计算。这是单输入重复测量，不是全集平均服务耗时。OWT 的质量/已有 timing 不受本次修复影响；若 OWT timing 尚未跑，可在独占时用本目录单独运行 `env ELF_STEPS="8 16 32 64" bash scripts/run_elf_suite.sh timing owt`，质量继续引用旧目录。

**5．恢复 test：先冻结标准 64 步，使用同一 EOS 规则**

validation 检查确认后可执行完整 WMT14 test 3003 条、XSum test 11334 条。此次 EOS 规则由作者 validation 实证确定，再应用于 test；没有作者 test token 文件用于逐行对照，因此报告“官方权重在公开 test 上按验证过的 source EOS 规则重评”，不宣称已确认论文全部 test 预处理相同。空 source 仍保留，完整 reference 不截断。

```bash
(
  set -e
  for task in wmt14 xsum; do
    runlog "${task}_test_s64_generate" \
      env ELF_COUNT=0 ELF_STEPS=64 ELF_SPLIT=test \
      bash scripts/run_elf_suite.sh generate "$task"
    runlog "${task}_test_s64_evaluate" \
      env ELF_STEPS=64 ELF_SPLIT=test \
      bash scripts/run_elf_suite.sh evaluate "$task"
    # 此时同样要求 GPU 独占。
    runlog "${task}_test_s64_timing" \
      env ELF_STEPS=64 ELF_SPLIT=test \
      bash scripts/run_elf_suite.sh timing "$task"
  done
)
python scripts/summarize_elf.py \
  --results "$ELF_RESULTS" --output "$ELF_RESULTS/summary.csv"
```

更少步数的 test 配置先根据 validation 冻结，再补测。OWT 前缀和扩展步数仍按文档 11 的 F/H2 实施；原始条件任务的相关命令使用本页的新结果根目录。任何已经成功的格子都不要原样重复 generate；若只缺指标则只 evaluate，若只缺 timing 则只 timing。

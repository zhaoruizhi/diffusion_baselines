# ELF 正式条件实验中断：诊断和继续运行

2026-09-12 最新日志显示：WMT14 连续完成了若干个 3000 样本格子，BLEU 为 13.4014、14.0021、14.1667；随后 XSum 在 `216/11332 samples` 后报 `ValueError: Empty condition`。这份恢复步骤优先于文档 11 中使用原始 `validation/test` 的条件实验命令。

**已确认的问题与尚未确认的原因**

- XSum 中断发生在下一批（零基索引 216–223）准备输入时：至少一行得到空的条件 token 列表。仅凭日志不能确定确切行号或是空字符串、空白字符还是分词后为空；新诊断会列出全部空条件 ID。
- `1069 > 512` 是 T5 tokenizer 的长度提示，不是这次异常。当前 ELF XSum 配置将 source 截到 1024，总 canvas 为 1088。不要把 source 上限改成 512 来消除提示，那会改变实验。新预检查使用 `verbose=False` 做完整分词，再沿用原有模型截断上限。
- WMT14 的 BLEU≈13–14 明显低于之前 official-validation sanity 的 25.79，需调查原始文本与作者条件 IDs 的差异。此次日志没有格子的步数标签；按此前循环顺序，最后完成的是 64 步，但应以 `request.json` 为准。不能把该降幅直接归为正常采样误差，也不能凭日志断定唯一原因是 EOS。
- 旧 `prepare` 对官方输入使用 `condition_input_ids`，对原始输入使用 `add_special_tokens=False` 重新分词。这与锁定版本的[官方 JSONL loader](https://github.com/lillian039/ELF/blob/b29d8833609e9ab7f67cd9da39435ac5cea04837/src/utils/data_utils.py#L121-L137)一致，但不能据此证明与发布的 Arrow tokens 等价。作者预处理 validation 才是[发布配置默认使用的数据](https://github.com/lillian039/ELF/blob/b29d8833609e9ab7f67cd9da39435ac5cea04837/README.md)。

此前从 sanity 直接切换原始 validation 全集的安排，缺少逐样本输入一致性核验。这次先补核验，并用已下载的作者 validation 全集建立 ELF 基线。原始 validation/test 暂停，不删空样本、不添加虚构 source、不猜测性插入 EOS。已有低分格子保留为待调查的原始输入协议结果；作者数据结果单独存放，不混成一条曲线。

**1．更新代码、恢复单张物理 GPU 2 环境**

新版本在加载 ELF 权重前扫描本次全部条件输入，空条件会报告数量和行号并提前退出，避免再生成几百条才失败。有效输入的 tokens、截断、采样器和指标没有改变。OWT 已成功完成的质量结果无需重跑。

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
export ELF_RESULTS="$DLB_ROOT/results/elf-h200-gpu2-official-v1"
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

失败的原始 XSum 目录包含 `samples.jsonl.tmp`，不是完整结果，也不能直接从第 217 条拼接续跑：脚本没有保存该位置的完整 RNG 状态。保留该目录，本次使用新的根目录和 `official-validation` split；不要移动或删除旧结果来伪装恢复。

**2．CPU 诊断，并核对之前各格子的任务与步数**

不下载数据、不加载模型权重、不占 GPU 推理。读取服务器现有 JSONL、manifest 和本地 T5 tokenizer：

```bash
runlog input_audit python scripts/audit_elf_inputs.py \
  --output "$ELF_RESULTS/input-audit.json"

python scripts/summarize_elf.py \
  --results "$DLB_ROOT/results/elf-h200-gpu2" \
  --output "$DLB_ROOT/results/elf-h200-gpu2/summary-before-recovery.csv"

python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['DLB_ROOT']) / 'results/elf-h200-gpu2/quality'
for path in sorted(root.rglob('request.json')):
    request = json.loads(path.read_text())
    metrics = path.with_name('metrics.json')
    m = json.loads(metrics.read_text()) if metrics.exists() else {}
    print(request['task'], (request.get('input') or {}).get('split', 'unconditional'),
          'steps=', request['steps'], 'valid=', m.get('valid'),
          'BLEU=', m.get('metrics', {}).get('bleu'), str(path.parent))
PY
```

报告重点：`raw.empty_condition_ids`、`official.empty_condition_ids`、按 source/reference 文本配对后的 `exact_ids_after_source_cap`、`raw_plus_eos_equals_official`、`mismatch_examples`。匹配按完整文本对进行，不假设两个数据集同一行号就是同一样本；无法匹配或重复匹配的行也会计数。EOS 对比只是诊断，不会修改 token。将 `logs/input_audit.log` 提供给我，用它确定原始 validation/test 的后续预处理修复。

**3．先跑作者 validation 全集的 64 步标准点**

先检查诊断成功且作者输入全部非空；通过后运行 WMT14 3000 条、XSum 11332 条。之前 sanity 是独立目录的 1000 条，这里是全集，不会覆盖它。

```bash
(
  set -e
  python - <<'PY'
import json, os
from pathlib import Path
r = json.loads((Path(os.environ['ELF_RESULTS']) / 'input-audit.json').read_text())
for task in ('wmt14', 'xsum'):
    assert r['tasks'][task]['official_conditions_nonempty'], (
        task, r['tasks'][task]['official']['empty_condition_ids'])
print('Official validation conditions are nonempty; generation will recheck input hashes.')
PY
  for task in wmt14 xsum; do
    runlog "${task}_official_s64_generate" \
      env ELF_COUNT=0 ELF_STEPS=64 ELF_SPLIT=official-validation \
      bash scripts/run_elf_suite.sh generate "$task"
    runlog "${task}_official_s64_evaluate" \
      env ELF_STEPS=64 ELF_SPLIT=official-validation \
      bash scripts/run_elf_suite.sh evaluate "$task"
  done
)
```

先看 64 步的 BLEU/ROUGE 是否回到 sanity 的量级。全集与前 1000 条分数可以不同，不设置为了“通过”而调参的分数门槛；如果仍明显偏低，先提供这两个指标与 audit 日志，暂不跑第 4 步的全量扫描。若 preflight 仍报错，保留错误信息，不绕过它。这里的结果应标为 **official-validation 全集**，不能称 test 结果。

**4．标准点合理后，补 8/16/32 步质量**

```bash
(
  set -e
  for task in wmt14 xsum; do
    for steps in 8 16 32; do
      runlog "${task}_official_s${steps}_generate" \
        env ELF_COUNT=0 ELF_STEPS="$steps" ELF_SPLIT=official-validation \
        bash scripts/run_elf_suite.sh generate "$task"
      runlog "${task}_official_s${steps}_evaluate" \
        env ELF_STEPS="$steps" ELF_SPLIT=official-validation \
        bash scripts/run_elf_suite.sh evaluate "$task"
    done
  done
)
```

**5．同一张 H200 独占时测 timing**

不要与 generation/evaluate 并行；确认 GPU 2 没有其他计算任务。条件 timing 与本次质量统一使用 `official-validation`，固定 batch=1，5 次预热、32 次测量，第 0 条输入。OWT 不受这次条件输入问题影响。

```bash
nvidia-smi -i "$ELF_GPU_UUID"
(
  set -e
  nvidia-smi -i "$ELF_GPU_UUID" > "$ELF_RESULTS/environment/gpu-before-timing.txt"
  runlog owt_core_timing env ELF_STEPS="8 16 32 64" \
    bash scripts/run_elf_suite.sh timing owt
  for task in wmt14 xsum; do
    runlog "${task}_official_core_timing" \
      env ELF_STEPS="8 16 32 64" ELF_SPLIT=official-validation \
      bash scripts/run_elf_suite.sh timing "$task"
  done
  nvidia-smi -i "$ELF_GPU_UUID" > "$ELF_RESULTS/environment/gpu-after-timing.txt"
)
python scripts/summarize_elf.py \
  --results "$ELF_RESULTS" --output "$ELF_RESULTS/summary.csv"
```

OWT 质量仍在旧根目录，以上 OWT timing 在新根目录；汇总时按任务/步数/seed/配置匹配，保留两边 provenance。条件任务看 `results.encoder_plus_sampler.seconds_per_sample`（含 T5 编码）及 `results.sampler_cached_condition.seconds_per_sample`（缓存编码）；OWT 看 `results.sampler.seconds_per_sample`。这些不含模型加载、CPU 分词、指标计算和 IO；第 0 条重复测量不代表全数据集平均服务延迟。

**6．后续实验**

OWT 前缀实验仍可按文档 11 的 F 准备和运行，不受 WMT14/XSum 输入问题影响。原始 validation/test 的正式恢复取决于本次 audit 的实证结果；不能将作者 validation 冒充 test，也不能删除空数据后继续声称原始全集。现在先取得可信的作者 validation 质量与 timing 基线，再完成原始集协议核验和 test。
